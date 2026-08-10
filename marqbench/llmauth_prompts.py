"""
llmauth_prompts.py — Prompt construction for MARQ-Bench (Paper 2).

Builds the frozen system prompt and the four information-ladder payloads
(A2-A5). Everything here is deterministic: identical inputs produce a
byte-identical prompt and therefore an identical SHA-256, so a prompt can be
reconstructed from its hash years later.

ANTI-TAUTOLOGY BOUNDARY — READ BEFORE EDITING
---------------------------------------------
This module MUST NOT import llmauth_ir, llmauth_corruption, or any evaluation
module. It is the authoring side of the wall; those are the scoring side. An
automated test asserts the absence of those imports.

Consequence: the output JSON schema below is DUPLICATED from llmauth_ir rather
than imported. That duplication is intentional. `test_llmauth_prompts.py`
imports both modules and asserts the vocabularies stay in sync, so drift is
caught by a test rather than prevented by an import.

WHAT MAY AND MAY NOT ENTER A PROMPT
-----------------------------------
May:  column names, dtypes, published documentation, census statistics stated
      as observed facts, sampled rows.
May not: the sentinel registry, the protected-attribute list, the failure
      taxonomy, injection specifications, retention targets, or any
      interpretation of what a value "means".

The census reports that 81.7% of `pdays` values are -1. It never says that -1
encodes "not previously contacted". The first is a statistic; the second is the
answer. `assert_no_leakage` enforces this on generated census text.

Author: Ramesh Babu Kallam
License: MIT
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

__all__ = [
    "SYSTEM_PROMPT", "SYSTEM_PROMPT_NO_WARNING", "OUTPUT_SCHEMA_BLOCK",
    "DIMENSION_VOCAB", "PREDICATE_VOCAB", "SEVERITY_VOCAB",
    "ColumnFacts", "CorpusFacts", "PromptBundle",
    "build_payload", "build_prompt", "sample_rows",
    "assert_no_leakage", "LeakageError",
]

PROMPT_VERSION = "1.0.0"
CONDITIONS = ("A2", "A3", "A4", "A5")
SAMPLE_ROWS_N = 20
TOP_K_LEVELS = 15


# --------------------------------------------------------------------------
# Controlled vocabularies — duplicated from llmauth_ir by design (see header)
# --------------------------------------------------------------------------

DIMENSION_VOCAB = (
    "completeness", "validity", "consistency",
    "uniqueness", "accuracy", "timeliness",
)
PREDICATE_VOCAB = (
    "not_null", "in_set", "range", "regex",
    "unique", "cross_column", "type",
)
SEVERITY_VOCAB = ("reject", "quarantine", "warn")


OUTPUT_SCHEMA_BLOCK = """\
Emit a single JSON object with this exact shape:

{
  "rules": [
    {
      "rule_id": "string, unique within this response",
      "column": "string, must be one of the columns listed above",
      "dimension": "one of: completeness | validity | consistency | uniqueness | accuracy | timeliness",
      "predicate_type": "one of: not_null | in_set | range | regex | unique | cross_column | type",
      "parameters": {},
      "severity": "one of: reject | quarantine | warn",
      "rationale": "one sentence explaining why a record failing this rule is defective"
    }
  ]
}

Parameters by predicate_type:
  not_null      {}
  in_set        {"allowed": [...]}                 non-empty list of permitted values
  range         {"min": <number>, "max": <number>} at least one of min/max
  regex         {"pattern": "<regex>"}             must fullmatch a passing value
  unique        {}
  cross_column  {"other_column": "<name>", "op": "< | <= | > | >= | == | !="}
  type          {"expected": "int | float | string"}
"""


# --------------------------------------------------------------------------
# System prompts — FROZEN. Any edit invalidates every prompt hash on record.
# --------------------------------------------------------------------------

_WARNING_CLAUSE = (
    " Prefer rules that would reject records a domain expert would consider "
    "erroneous, and avoid rules that would reject records that are merely "
    "unusual or that use a documented encoding convention."
)

_SYSTEM_BASE = (
    "You are a data quality engineer. Given a description of a table, author "
    "validation rules that identify genuinely defective records. Emit only "
    "JSON conforming to the provided schema, with no prose outside the JSON. "
    "Do not author rules for columns not listed."
)

SYSTEM_PROMPT = _SYSTEM_BASE + _WARNING_CLAUSE
SYSTEM_PROMPT_NO_WARNING = _SYSTEM_BASE


# --------------------------------------------------------------------------
# Leakage guard
# --------------------------------------------------------------------------

class LeakageError(RuntimeError):
    """Raised when generated prompt text interprets rather than reports."""


# Terms that would tell the author what a value MEANS rather than how often it
# occurs. Applied to census text this module generates. NOT applied to
# published documentation (condition A3), which is quoted verbatim on purpose.
_FORBIDDEN_TERMS = (
    "sentinel", "placeholder", "encoded missing", "encodes missing",
    "missing code", "missing value code", "means not", "indicates missing",
    "stands for", "should be treated as", "is actually", "represents missing",
    "do not reject", "do not flag", "valid despite", "not a defect",
    "protected attribute", "fairness", "ground truth", "injected",
)
_FORBIDDEN_RE = re.compile("|".join(re.escape(t) for t in _FORBIDDEN_TERMS), re.I)


def assert_no_leakage(text: str, *, where: str = "census") -> None:
    """Fail loudly if generated text interprets values instead of reporting them."""
    hit = _FORBIDDEN_RE.search(text)
    if hit:
        raise LeakageError(
            f"interpretive term {hit.group(0)!r} found in {where} text; "
            "the census may report frequencies but must not explain meanings"
        )


# --------------------------------------------------------------------------
# Corpus facts (independent of llmauth_ir by design)
# --------------------------------------------------------------------------

@dataclass
class ColumnFacts:
    """Observed facts about one column, from the AUTHORING SPLIT only.

    `top_levels` MUST be populated for every column, numeric ones included.
    Modal concentration is the single most load-bearing statistic in the
    census: `pdays` has 559 distinct values, so a cardinality-gated profiler
    would emit no level list for it and the census would convey only
    "range -1 to 871, p50 = -1". That understates the fact that 81.70% of
    values are exactly -1 — which is the entire signal condition A4 is
    supposed to supply. Omitting it silently weakens the A4 treatment and
    would bias the study against H2.
    """
    name: str
    dtype: str                       # "int" | "float" | "string" | "bool" | "timestamp"
    doc: str = ""                    # published documentation, condition A3
    null_rate: float | None = None
    distinct_count: int | None = None
    min_value: Any = None
    max_value: Any = None
    quantiles: dict[str, Any] = field(default_factory=dict)   # {"p01":..,"p50":..}
    top_levels: list[tuple[Any, int]] = field(default_factory=list)  # (value, count)


@dataclass
class CorpusFacts:
    corpus_id: str
    row_count: int
    columns: list[ColumnFacts]
    table_doc: str = ""              # corpus-level prose, condition A3

    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]


@dataclass
class PromptBundle:
    """A fully assembled prompt plus the provenance needed to reproduce it."""
    system: str
    user: str
    condition: str
    corpus_id: str
    prompt_version: str = PROMPT_VERSION
    warning_clause: bool = True
    sample_seed: int | None = None

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(
                {
                    "system": self.system,
                    "user": self.user,
                    "prompt_version": self.prompt_version,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

    def to_messages(self) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user},
        ]

    def provenance(self) -> dict[str, Any]:
        return {
            "prompt_sha256": self.sha256,
            "prompt_version": self.prompt_version,
            "condition": self.condition,
            "corpus_id": self.corpus_id,
            "warning_clause": self.warning_clause,
            "sample_seed": self.sample_seed,
        }


# --------------------------------------------------------------------------
# Serializers — each block is deterministic given its inputs
# --------------------------------------------------------------------------

def _fmt(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def serialize_schema(facts: CorpusFacts) -> str:
    """A2 payload: column names and types only. No statistics, no prose."""
    lines = [f"Table: {facts.corpus_id}", "", "Columns:"]
    for c in facts.columns:
        lines.append(f"  {c.name}: {c.dtype}")
    return "\n".join(lines)


def serialize_dictionary(facts: CorpusFacts) -> str:
    """A3 addition: published documentation, quoted verbatim.

    Deliberately NOT leakage-checked. The published dictionaries for all four
    corpora do describe their encoding conventions — UCI documents that
    `pdays = -1` means the client was not previously contacted; TLC documents
    `RatecodeID = 99` as Null/unknown. Condition A3 exists precisely to test
    whether supplying that documentation is sufficient. Sanitising it would
    destroy the condition.
    """
    lines: list[str] = []
    if facts.table_doc:
        lines += ["Table description:", facts.table_doc.strip(), ""]
    documented = [c for c in facts.columns if c.doc]
    if documented:
        lines.append("Column descriptions (from published documentation):")
        for c in documented:
            lines.append(f"  {c.name}: {c.doc.strip()}")
    return "\n".join(lines)


def serialize_census(facts: CorpusFacts) -> str:
    """A4 addition: observed statistics, stated as frequencies only.

    Reports that a value occurs and how often. Never why it occurs or what it
    means. Passed through assert_no_leakage before return.
    """
    lines = [
        f"Profile computed on an authoring sample of {facts.row_count:,} rows.",
        "All figures below are observed counts and frequencies.",
        "",
    ]
    for c in facts.columns:
        parts: list[str] = []
        if c.null_rate is not None:
            parts.append(f"null {c.null_rate * 100:.2f}%")
        if c.distinct_count is not None:
            parts.append(f"{c.distinct_count} distinct")
        if c.min_value is not None or c.max_value is not None:
            parts.append(f"range {_fmt(c.min_value)} to {_fmt(c.max_value)}")
        if c.quantiles:
            q = ", ".join(f"{k}={_fmt(v)}" for k, v in sorted(c.quantiles.items()))
            parts.append(f"quantiles {q}")
        lines.append(f"  {c.name} ({c.dtype}): " + "; ".join(parts))

        if c.top_levels:
            total = facts.row_count or 1
            shown = c.top_levels[:TOP_K_LEVELS]
            lines.append("      most frequent values:")
            for value, count in shown:
                lines.append(
                    f"        {value!r}: {count:,} ({count / total * 100:.2f}%)"
                )
            if len(c.top_levels) > TOP_K_LEVELS:
                lines.append(
                    f"        ... {len(c.top_levels) - TOP_K_LEVELS} further "
                    "values not shown"
                )
    text = "\n".join(lines)
    assert_no_leakage(text, where="census")
    return text


def sample_rows(
    rows: Sequence[dict[str, Any]],
    n: int = SAMPLE_ROWS_N,
    seed: int = 20260807,
) -> list[dict[str, Any]]:
    """Deterministic sample. Same seed and input always yield the same rows."""
    if len(rows) <= n:
        return list(rows)
    rng = random.Random(seed)
    return [rows[i] for i in sorted(rng.sample(range(len(rows)), n))]


def serialize_samples(rows: Sequence[dict[str, Any]]) -> str:
    """A5 addition: sampled rows as JSON lines, keys in schema order."""
    lines = [f"{len(rows)} sampled rows from the authoring split:"]
    for r in rows:
        lines.append("  " + json.dumps(r, default=str, sort_keys=False))
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------

_TASK_INSTRUCTION = (
    "Author validation rules for the table described above. Return only the "
    "JSON object."
)


def build_payload(
    condition: str,
    facts: CorpusFacts,
    rows: Sequence[dict[str, Any]] | None = None,
    *,
    sample_seed: int = 20260807,
) -> str:
    """Assemble the user message body for one information condition.

    A2  schema
    A3  schema + published documentation
    A4  schema + census
    A5  schema + census + sampled rows
    """
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; expected one of {CONDITIONS}")

    blocks = [serialize_schema(facts)]

    if condition == "A3":
        doc = serialize_dictionary(facts)
        if not doc.strip():
            raise ValueError(
                f"condition A3 requested for {facts.corpus_id} but no "
                "documentation is present; A3 without docs is silently A2"
            )
        blocks.append(doc)

    if condition in ("A4", "A5"):
        blocks.append(serialize_census(facts))

    if condition == "A5":
        if not rows:
            raise ValueError("condition A5 requires sampled rows")
        blocks.append(serialize_samples(sample_rows(rows, seed=sample_seed)))

    blocks.append(OUTPUT_SCHEMA_BLOCK)
    blocks.append(_TASK_INSTRUCTION)
    return "\n\n".join(b.strip() for b in blocks if b.strip())


def build_prompt(
    condition: str,
    facts: CorpusFacts,
    rows: Sequence[dict[str, Any]] | None = None,
    *,
    warning_clause: bool = True,
    sample_seed: int = 20260807,
) -> PromptBundle:
    """Build a complete, hashable prompt bundle for one generation run."""
    return PromptBundle(
        system=SYSTEM_PROMPT if warning_clause else SYSTEM_PROMPT_NO_WARNING,
        user=build_payload(condition, facts, rows, sample_seed=sample_seed),
        condition=condition,
        corpus_id=facts.corpus_id,
        warning_clause=warning_clause,
        sample_seed=sample_seed if condition == "A5" else None,
    )
