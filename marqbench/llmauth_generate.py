"""
llmauth_generate.py — Generation sweep runner for MARQ-Bench (Paper 2).

Calls language models to author rule sets and records every run with the
provenance needed to reproduce or audit it. Provider-agnostic: backends are
thin adapters behind one interface, so the model panel can change without
touching the sweep logic.

DESIGNED FOR COLAB
------------------
The full sweep is 640 generations. A Colab session will not survive it. This
module therefore appends one JSON line per run to a file on Drive, flushed
immediately, and skips runs already present on restart. Reconnect, re-run the
cell, and it continues from where it stopped.

RAW CAPTURE IS THE POINT
------------------------
The complete, unparsed response text is stored before anything touches it.
When a parser bug surfaces in month three, the raw corpus is what saves you
from regenerating everything. Never store only parsed rules.

FAILURES ARE DATA
-----------------
A run that errors, refuses, or returns nothing is retried up to three times and
then recorded as a null result with its reason. It is never replaced with a
fresh seed. Silently retrying until success would bias the sample toward
well-behaved models and inflate every quality rate.

ANTI-TAUTOLOGY BOUNDARY
-----------------------
Authoring side. Must not import llmauth_ir or any evaluation module. Parsing
and scoring happen later, in notebook N3.

Author: Ramesh Babu Kallam
License: MIT
"""

from __future__ import annotations

import json
import os
import random
import re
import time
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

from llmauth_prompts import PromptBundle

__all__ = [
    "GENERATE_VERSION", "GenerationResult", "RunKey", "SweepPlan",
    "Backend", "MockBackend", "AnthropicBackend", "OpenAIBackend",
    "GeminiBackend",
    "HuggingFaceBackend", "RunLog", "run_sweep", "estimate_cost",
    "preflight_backends", "list_anthropic_models", "resolve_hf_revision",
    "NON_RETRYABLE_ERRORS",
]

GENERATE_VERSION = "1.0.0"
MAX_ATTEMPTS = 3
DEFAULT_MAX_TOKENS = 4096
BASE_BACKOFF_SECONDS = 2.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------
# Run identity and results
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class RunKey:
    """Uniquely identifies one cell of the sweep. Used for resume."""
    corpus_id: str
    condition: str
    model_id: str
    seed: int
    temperature: float
    variant: str = "main"      # "main" | "no_warning" | "temp0"

    def as_str(self) -> str:
        return (f"{self.corpus_id}|{self.condition}|{self.model_id}"
                f"|{self.seed}|{self.temperature}|{self.variant}")


@dataclass
class GenerationResult:
    """One generation run. Serialized as a single JSON line."""
    key: dict[str, Any]
    raw_response: str | None
    ok: bool
    failure_reason: str = ""
    attempts: int = 1
    model_version: str = ""
    prompt_sha256: str = ""
    prompt_version: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    latency_seconds: float | None = None
    started_at_utc: str = ""
    finished_at_utc: str = ""
    request_params: dict[str, Any] = field(default_factory=dict)
    generate_version: str = GENERATE_VERSION

    def to_json_line(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, default=str)


# --------------------------------------------------------------------------
# Backend interface
# --------------------------------------------------------------------------

@dataclass
class BackendResponse:
    text: str
    model_version: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    applied_params: dict[str, Any] = field(default_factory=dict)
    """What the backend ACTUALLY sent, which may differ from what was asked.

    Newer models deprecate `temperature`, and several APIs have no `seed`
    parameter at all. Recording the request as issued rather than as intended
    keeps the run log honest — a log claiming temperature=0.7 when the field
    was never sent would silently misdescribe the experiment.
    """


class Backend(Protocol):
    """Adapter for one model. Keep these thin — no retry, no logging."""

    model_id: str

    def generate(
        self,
        system: str,
        user: str,
        *,
        temperature: float,
        seed: int,
        max_tokens: int,
    ) -> BackendResponse: ...


class MockBackend:
    """Deterministic fake for testing the sweep without spending anything."""

    def __init__(self, model_id: str = "mock-1", fail_seeds: Sequence[int] = ()):
        self.model_id = model_id
        self.fail_seeds = set(fail_seeds)
        self.calls = 0

    def generate(self, system, user, *, temperature, seed, max_tokens):
        self.calls += 1
        if seed in self.fail_seeds:
            raise RuntimeError(f"simulated backend failure on seed {seed}")
        rng = random.Random(seed)
        lo = rng.choice([0, 1, 18])
        payload = {"rules": [{
            "rule_id": f"mock_{seed}", "column": "age",
            "dimension": "validity", "predicate_type": "range",
            "parameters": {"min": lo, "max": 95},
            "severity": "reject", "rationale": "plausible adult age",
        }]}
        return BackendResponse(
            text=json.dumps(payload),
            model_version=f"{self.model_id}-v0",
            input_tokens=len(user) // 4,
            output_tokens=len(json.dumps(payload)) // 4,
            applied_params={"temperature": temperature,
                            "temperature_applied": True,
                            "seed_applied": True,
                            "max_tokens": max_tokens},
        )


class AnthropicBackend:
    """Adapter for the Anthropic Messages API.

    Two API facts this handles explicitly, because both affect what the run log
    is allowed to claim:

    1. **Some models reject `temperature`.** Newer models return
       400 "`temperature` is deprecated for this model". On the first such
       error the backend records that the model does not accept the parameter
       and reissues without it, for the rest of the session.
    2. **There is no `seed` parameter.** Sampling variation across seeds is
       therefore run-to-run API nondeterminism, not seeded sampling. That is
       still a valid measurement for RQ3 — arguably a more realistic one — but
       it must be described accurately in the paper, so `applied_params`
       records `seed_applied: False`.
    """

    def __init__(self, model_id: str, api_key: str | None = None):
        import anthropic
        self.model_id = model_id
        self._anthropic = anthropic
        self._client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )
        self._temperature_supported: bool | None = None   # None = not yet known

    def _build_kwargs(self, system, user, temperature, max_tokens):
        kw: dict[str, Any] = {
            "model": self.model_id,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if self._temperature_supported is not False:
            kw["temperature"] = temperature
        return kw

    def generate(self, system, user, *, temperature, seed, max_tokens):
        kw = self._build_kwargs(system, user, temperature, max_tokens)
        try:
            resp = self._client.messages.create(**kw)
        except self._anthropic.BadRequestError as exc:
            if "temperature" not in str(exc).lower() or self._temperature_supported is False:
                raise
            print(f"  [backend] {self.model_id} rejects `temperature`; "
                  "reissuing without it and recording that in provenance")
            self._temperature_supported = False
            kw.pop("temperature", None)
            resp = self._client.messages.create(**kw)
        else:
            if self._temperature_supported is None:
                self._temperature_supported = True

        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        return BackendResponse(
            text=text,
            model_version=getattr(resp, "model", self.model_id),
            input_tokens=getattr(resp.usage, "input_tokens", None),
            output_tokens=getattr(resp.usage, "output_tokens", None),
            applied_params={
                "temperature": temperature if self._temperature_supported else None,
                "temperature_applied": bool(self._temperature_supported),
                "seed_applied": False,          # the API has no seed parameter
                "max_tokens": max_tokens,
            },
        )


class OpenAIBackend:
    """Adapter for the OpenAI Chat Completions API."""

    def __init__(self, model_id: str, api_key: str | None = None):
        from openai import OpenAI
        self.model_id = model_id
        self._client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self._temperature_supported: bool | None = None
        self._seed_supported: bool | None = None

    def generate(self, system, user, *, temperature, seed, max_tokens):
        kw = {
            "model": self.model_id,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if self._temperature_supported is not False:
            kw["temperature"] = temperature
        if self._seed_supported is not False:
            kw["seed"] = seed

        for _ in range(3):          # drop at most one unsupported field per pass
            try:
                resp = self._client.chat.completions.create(**kw)
                break
            except Exception as exc:
                msg = str(exc).lower()
                if "temperature" in msg and "temperature" in kw:
                    print(f"  [backend] {self.model_id} rejects `temperature`; "
                          "reissuing without it")
                    self._temperature_supported = False
                    kw.pop("temperature")
                    continue
                if "seed" in msg and "seed" in kw:
                    print(f"  [backend] {self.model_id} rejects `seed`; "
                          "reissuing without it")
                    self._seed_supported = False
                    kw.pop("seed")
                    continue
                raise
        else:
            raise RuntimeError("exhausted parameter fallbacks")

        if self._temperature_supported is None:
            self._temperature_supported = "temperature" in kw
        if self._seed_supported is None:
            self._seed_supported = "seed" in kw

        usage = getattr(resp, "usage", None)
        return BackendResponse(
            text=resp.choices[0].message.content or "",
            model_version=getattr(resp, "model", self.model_id),
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
            applied_params={
                "temperature": temperature if self._temperature_supported else None,
                "temperature_applied": bool(self._temperature_supported),
                "seed_applied": bool(self._seed_supported),
                "max_tokens": max_tokens,
            },
        )


class GeminiBackend:
    """Adapter for the Google Gemini API.

    Included to give the model panel a second vendor and a genuinely
    independent architecture, which is the most likely reviewer request for a
    study whose panel is otherwise single-vendor.

    FREE TIER
    ---------
    Gemini offers a free tier with no card required, but the quotas have
    changed repeatedly and published figures disagree. Do not hard-code an
    assumption about them. This adapter instead:

      - paces requests to a configurable requests-per-minute budget;
      - on a 429, honours the server's retry delay if it supplies one, and
        otherwise backs off exponentially;
      - raises a clear DailyQuotaExhausted when the daily cap is hit, so the
        sweep stops cleanly rather than burning attempts.

    Because the sweep is resumable, a restrictive daily cap simply means the
    160 runs complete across several days at no cost.

    DATA USE
    --------
    Google's terms state that free-tier prompts may be used to improve their
    models; paid tier and Vertex AI do not. Every prompt in this study is
    derived from public corpora and contains no proprietary content, so this is
    disclosable rather than disqualifying — but it MUST be disclosed in the
    paper's methods. Verify the current terms before running.
    """

    class DailyQuotaExhausted(RuntimeError):
        """Raised when the daily request cap is reached. Resume tomorrow."""

    def __init__(self, model_id: str, api_key: str | None = None,
                 requests_per_minute: float = 10.0):
        from google import genai
        self.model_id = model_id
        self._genai = genai
        self._client = genai.Client(
            api_key=api_key or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GEMINI_API_KEY"))
        self.min_interval = 60.0 / max(requests_per_minute, 0.1)
        self._last_call = 0.0

    def _pace(self):
        wait = self.min_interval - (time.time() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.time()

    @staticmethod
    def _retry_delay(exc) -> float | None:
        """Seconds the server asked us to wait, if it said."""
        m = re.search(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)",
                      str(exc))
        return float(m.group(1)) if m else None

    def generate(self, system, user, *, temperature, seed, max_tokens):
        from google.genai import types

        cfg = types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        for attempt in range(1, 6):
            self._pace()
            try:
                resp = self._client.models.generate_content(
                    model=self.model_id, contents=user, config=cfg)
                break
            except Exception as exc:  # noqa: BLE001
                msg = str(exc)
                is_429 = "429" in msg or "RESOURCE_EXHAUSTED" in msg.upper()
                # 503 UNAVAILABLE means the model is busy, not that we are over
                # quota. It is transient and must be retried — treating it as
                # fatal silently truncates a run and leaves gaps that look like
                # results.
                is_503 = "503" in msg or "UNAVAILABLE" in msg.upper()
                if not (is_429 or is_503):
                    raise
                if is_503:
                    delay = min(self.min_interval * 2 ** attempt, 60.0)
                    print(f"  [gemini] 503 model busy, waiting {delay:.0f}s "
                          f"(attempt {attempt}/5)")
                    time.sleep(delay)
                    continue
                if "PerDay" in msg or "per day" in msg.lower():
                    raise self.DailyQuotaExhausted(
                        "Gemini daily free-tier quota exhausted. The run log is "
                        "resumable — re-run the sweep cell tomorrow and it will "
                        "continue from where it stopped."
                    ) from exc
                delay = self._retry_delay(exc) or (self.min_interval * 2 ** attempt)
                print(f"  [gemini] rate limited, waiting {delay:.0f}s "
                      f"(attempt {attempt}/5)")
                time.sleep(delay)
        else:
            raise RuntimeError("Gemini: exhausted rate-limit retries")

        text = getattr(resp, "text", "") or ""
        usage = getattr(resp, "usage_metadata", None)
        return BackendResponse(
            text=text,
            model_version=getattr(resp, "model_version", None) or self.model_id,
            input_tokens=getattr(usage, "prompt_token_count", None),
            output_tokens=getattr(usage, "candidates_token_count", None),
            applied_params={"temperature": temperature,
                            "temperature_applied": True,
                            "seed_applied": False,   # no seed parameter
                            "max_tokens": max_tokens,
                            "requests_per_minute": 60.0 / self.min_interval},
        )


class HuggingFaceBackend:
    """Adapter for a locally hosted open-weight model.

    Pin by commit hash, not tag. A tag can be moved; a hash cannot, and the
    reproducibility claim depends on it.
    """

    def __init__(self, model_id: str, revision: str, load_in_4bit: bool = True,
                 attn_implementation: str = "sdpa", max_input_tokens: int = 12000):
        import os as _os
        # Reduce allocator fragmentation; must be set before CUDA initialises.
        _os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model_id = model_id
        self.revision = revision
        self.load_in_4bit = load_in_4bit
        self.max_input_tokens = max_input_tokens
        self._torch = torch
        if len(revision) < 40:
            print(f"  [backend] WARNING: revision {revision!r} is not a full commit "
                  "hash. Tags and branch names move; pin by hash (see "
                  "resolve_hf_revision).")
        self._tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
        kwargs: dict[str, Any] = {
            "revision": revision,
            "device_map": "auto",
            # Memory-efficient attention. Without this, some configurations
            # fall back to eager attention, which materialises the full
            # seq_len x seq_len matrix per layer — a ~5k-token prompt then
            # tries to allocate over 12 GiB and dies on a 16 GB card.
            "attn_implementation": attn_implementation,
        }
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        try:
            self._model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        except (ValueError, ImportError) as exc:
            print(f"  [backend] {attn_implementation} unavailable ({exc}); "
                  "falling back to eager attention — expect higher memory use")
            kwargs["attn_implementation"] = "eager"
            self._model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        self._model.eval()

    def generate(self, system, user, *, temperature, seed, max_tokens):
        self._torch.manual_seed(seed)
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        prompt = self._tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tok(prompt, return_tensors="pt").to(self._model.device)
        n_in = int(inputs["input_ids"].shape[1])

        # Never truncate. A shortened prompt is a different experimental
        # condition, and silently running A5-minus-half-its-samples would
        # corrupt the study far worse than a recorded failure does.
        if n_in > self.max_input_tokens:
            raise RuntimeError(
                f"prompt is {n_in} tokens, above max_input_tokens="
                f"{self.max_input_tokens}. Refusing to truncate, because that "
                "would silently alter the information condition. Use a smaller "
                "model or a larger GPU for this cell."
            )

        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self._tok.pad_token_id or self._tok.eos_token_id,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.95

        try:
            with self._torch.inference_mode():
                out = self._model.generate(**inputs, **gen_kwargs)
            text = self._tok.decode(out[0][n_in:], skip_special_tokens=True)
            n_out = int(out.shape[1] - n_in)
        finally:
            # Release the KV cache before the next run. Without this the
            # allocator holds several GB across calls and the sweep OOMs part
            # way through even though each individual run would fit.
            inputs = None
            out = None
            import gc
            gc.collect()
            self._torch.cuda.empty_cache()
        return BackendResponse(
            text=text,
            model_version=f"{self.model_id}@{self.revision}",
            input_tokens=n_in,
            output_tokens=n_out,
            applied_params={"temperature": temperature,
                            "temperature_applied": temperature > 0,
                            "seed_applied": True,      # torch.manual_seed
                            "max_tokens": max_tokens,
                            "revision": self.revision,
                            "quantization": "4bit-nf4" if self.load_in_4bit else "fp16"},
        )


# --------------------------------------------------------------------------
# Cost
# --------------------------------------------------------------------------

# USD per 1M tokens. Update at sweep time and record what you used —
# published prices change and the paper reports actual spend.
PRICING: dict[str, tuple[float, float]] = {}


def estimate_cost(model_id: str, in_tok: int | None, out_tok: int | None) -> float | None:
    if model_id not in PRICING or in_tok is None or out_tok is None:
        return None
    pin, pout = PRICING[model_id]
    return (in_tok / 1e6) * pin + (out_tok / 1e6) * pout


# --------------------------------------------------------------------------
# Append-only, resumable run log
# --------------------------------------------------------------------------

class RunLog:
    """JSON-lines log on Drive. Append-only, flushed per write, resumable."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._completed = self._scan()

    def _scan(self) -> set[str]:
        done: set[str] = set()
        if not self.path.exists():
            return done
        with open(self.path, "r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    done.add(RunKey(**rec["key"]).as_str())
                except (json.JSONDecodeError, KeyError, TypeError):
                    # A session killed mid-write can leave one truncated line.
                    # Skip it rather than crash; it will simply be re-run.
                    print(f"  [runlog] skipping unreadable line {line_no}")
        return done

    def is_done(self, key: RunKey) -> bool:
        return key.as_str() in self._completed

    def append(self, result: GenerationResult) -> None:
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(result.to_json_line() + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        self._completed.add(RunKey(**result.key).as_str())

    def __len__(self) -> int:
        return len(self._completed)

    def purge_failures(self, *, reason: str) -> int:
        """Remove FAILED runs so they will be re-attempted. Returns count removed.

        Failures are data (see module header) and must never be silently
        re-seeded away. But a failure caused by a configuration or adapter bug
        — a rejected API parameter, a missing key — is not a finding about the
        model, and leaving it in the log would permanently suppress that cell.

        This makes the distinction explicit and auditable: a `reason` is
        required, removed records are archived to a sidecar file rather than
        deleted, and the purge itself is appended to a purge log. Successful
        runs are never touched.

        Do NOT use this to retry genuine refusals or malformed outputs. Those
        are results.
        """
        if not reason or not reason.strip():
            raise ValueError(
                "purge_failures requires a reason; it is recorded in the audit "
                "trail and will appear in the paper's deviation log"
            )
        if not self.path.exists():
            return 0

        kept, removed = [], []
        for line in open(self.path, "r", encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue                     # truncated line: drop
            (kept if rec.get("ok") else removed).append(rec)

        if removed:
            archive = self.path.with_suffix(".purged.jsonl")
            with open(archive, "a", encoding="utf-8") as fh:
                for rec in removed:
                    fh.write(json.dumps({**rec, "_purge_reason": reason,
                                         "_purged_at_utc": _utc_now()},
                                        default=str) + "\n")
            tmp = self.path.with_suffix(".jsonl.tmp")
            with open(tmp, "w", encoding="utf-8") as fh:
                for rec in kept:
                    fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            os.replace(tmp, self.path)

            with open(self.path.with_suffix(".purgelog.jsonl"), "a",
                      encoding="utf-8") as fh:
                fh.write(json.dumps({"at_utc": _utc_now(), "reason": reason,
                                     "removed": len(removed),
                                     "kept": len(kept)}) + "\n")

        self._completed = self._scan()
        return len(removed)

    def summary(self) -> dict[str, Any]:
        """Counts and spend so far. Cheap enough to call between cells."""
        n_ok = n_fail = 0
        cost = 0.0
        by_model: dict[str, int] = {}
        if self.path.exists():
            for line in open(self.path, encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                n_ok += bool(rec.get("ok"))
                n_fail += not rec.get("ok")
                cost += rec.get("cost_usd") or 0.0
                m = rec.get("key", {}).get("model_id", "?")
                by_model[m] = by_model.get(m, 0) + 1
        return {"runs": n_ok + n_fail, "ok": n_ok, "failed": n_fail,
                "cost_usd": round(cost, 4), "by_model": by_model}


# --------------------------------------------------------------------------
# Sweep
# --------------------------------------------------------------------------

@dataclass
class SweepPlan:
    """One planned run: the key plus the prompt to send."""
    key: RunKey
    prompt: PromptBundle


# Errors that will never succeed on retry. A wrong model id, a bad key, or a
# revoked permission is a configuration fault, not a transient failure —
# retrying three times with backoff just wastes time and money on every run.
NON_RETRYABLE_ERRORS = (
    "NotFoundError", "AuthenticationError", "PermissionDeniedError",
    "UnprocessableEntityError", "InvalidRequestError",
)


def preflight_backends(
    backends: dict[str, "Backend"],
    *,
    max_tokens: int = 16,
) -> dict[str, str]:
    """Send one tiny request per backend to prove it works. Returns {id: status}.

    Run this before any sweep. A wrong model id costs one cheap call to detect
    here, versus hundreds of failed runs and a polluted run log to detect the
    hard way.
    """
    results: dict[str, str] = {}
    for model_id, backend in backends.items():
        try:
            resp = backend.generate(
                "Reply with the single word: ok",
                "ok",
                temperature=0.0, seed=1, max_tokens=max_tokens,
            )
            version = resp.model_version or model_id
            results[model_id] = f"OK   {version}"
        except Exception as exc:  # noqa: BLE001
            results[model_id] = f"FAIL {type(exc).__name__}: {str(exc)[:120]}"
    return results


def resolve_hf_revision(model_id: str, token: str | None = None) -> str:
    """Return the current commit SHA for a Hugging Face repo's main branch.

    Pin by this, never by a tag or branch name. `main` moves; a commit hash
    does not. A study whose subject is reproducibility cannot cite a moving
    reference for its own models.

    Resolve once, paste the returned hash into your backend registration as a
    literal, and record it in provenance. Do not call this inside the sweep —
    that would re-resolve to whatever is current at run time and defeat the
    purpose.
    """
    from huggingface_hub import HfApi
    info = HfApi(token=token).model_info(model_id)
    return info.sha


def list_anthropic_models(api_key: str | None = None) -> list[str]:
    """List model ids available to this API key. Useful when a 404 says a model
    id is unrecognised — model identifiers are retired over time."""
    import anthropic
    client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
    return [m.id for m in client.models.list(limit=100).data]


def _execute_one(
    backend: Backend,
    plan: SweepPlan,
    max_tokens: int,
    sleep_between: float,
) -> GenerationResult:
    started = _utc_now()
    t0 = time.time()
    last_error = ""

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            resp = backend.generate(
                plan.prompt.system,
                plan.prompt.user,
                temperature=plan.key.temperature,
                seed=plan.key.seed,
                max_tokens=max_tokens,
            )
            if not (resp.text or "").strip():
                raise ValueError("empty response body")
            return GenerationResult(
                key=asdict(plan.key),
                raw_response=resp.text,
                ok=True,
                attempts=attempt,
                model_version=resp.model_version,
                prompt_sha256=plan.prompt.sha256,
                prompt_version=plan.prompt.prompt_version,
                input_tokens=resp.input_tokens,
                output_tokens=resp.output_tokens,
                cost_usd=estimate_cost(
                    plan.key.model_id, resp.input_tokens, resp.output_tokens
                ),
                latency_seconds=round(time.time() - t0, 3),
                started_at_utc=started,
                finished_at_utc=_utc_now(),
                request_params={
                    "requested": {"temperature": plan.key.temperature,
                                  "seed": plan.key.seed,
                                  "max_tokens": max_tokens},
                    "applied": resp.applied_params,
                },
            )
        except Exception as exc:  # noqa: BLE001 — any failure is data
            last_error = f"{type(exc).__name__}: {exc}"
            if type(exc).__name__ in NON_RETRYABLE_ERRORS:
                # Configuration fault. Retrying cannot help; fail fast so the
                # sweep surfaces it on the first run instead of the last.
                print(f"  [backend] {plan.key.model_id}: {type(exc).__name__} — "
                      "not retrying (configuration error, not transient)")
                break
            if attempt < MAX_ATTEMPTS:
                time.sleep(BASE_BACKOFF_SECONDS * (2 ** (attempt - 1)))
            else:
                traceback.print_exc(limit=1)

    # Exhausted. Record the failure; do NOT substitute another seed.
    return GenerationResult(
        key=asdict(plan.key),
        raw_response=None,
        ok=False,
        failure_reason=last_error,
        attempts=attempt,
        prompt_sha256=plan.prompt.sha256,
        prompt_version=plan.prompt.prompt_version,
        latency_seconds=round(time.time() - t0, 3),
        started_at_utc=started,
        finished_at_utc=_utc_now(),
        request_params={
            "requested": {"temperature": plan.key.temperature,
                          "seed": plan.key.seed,
                          "max_tokens": max_tokens},
            "applied": None,       # the call never succeeded
        },
    )


def run_sweep(
    plans: Iterable[SweepPlan],
    backends: dict[str, Backend],
    log: RunLog,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    sleep_between: float = 0.0,
    progress_every: int = 10,
) -> dict[str, Any]:
    """Execute planned runs, skipping any already in the log.

    Safe to interrupt and re-run. Returns a summary dict.
    """
    plans = list(plans)
    todo = [p for p in plans if not log.is_done(p.key)]
    skipped = len(plans) - len(todo)
    print(f"[sweep] {len(plans)} planned, {skipped} already done, {len(todo)} to run")

    done = 0
    for plan in todo:
        backend = backends.get(plan.key.model_id)
        if backend is None:
            raise KeyError(f"no backend registered for {plan.key.model_id!r}")
        result = _execute_one(backend, plan, max_tokens, sleep_between)
        log.append(result)
        done += 1
        if done % progress_every == 0 or done == len(todo):
            s = log.summary()
            print(f"[sweep] {done}/{len(todo)} | ok={s['ok']} "
                  f"failed={s['failed']} spend=${s['cost_usd']:.3f}")
        if sleep_between:
            time.sleep(sleep_between)

    return {"planned": len(plans), "skipped": skipped,
            "executed": done, **log.summary()}
