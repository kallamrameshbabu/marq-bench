"""Tests for llmauth_generate. Run: python3 test_llmauth_generate.py"""
import ast
import json
import tempfile
from pathlib import Path

import llmauth_generate as G
import llmauth_prompts as P

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


G.BASE_BACKOFF_SECONDS = 0.0  # keep the suite fast

facts = P.CorpusFacts("bank_marketing", 9042, [
    P.ColumnFacts("age", "int", null_rate=0.0, distinct_count=70,
                  min_value=18, max_value=95, top_levels=[(32, 400), (31, 390)]),
    P.ColumnFacts("poutcome", "string", null_rate=0.0, distinct_count=4,
                  top_levels=[("unknown", 7400), ("failure", 980)]),
])


def make_plans(model_id="mock-1", seeds=(1, 2, 3), condition="A4"):
    out = []
    for s in seeds:
        out.append(G.SweepPlan(
            key=G.RunKey("bank_marketing", condition, model_id, s, 0.7),
            prompt=P.build_prompt(condition, facts),
        ))
    return out


print("\n=== 1. Anti-tautology boundary ===")
tree = ast.parse(open(G.__file__).read())
imported = set()
for n in ast.walk(tree):
    if isinstance(n, ast.Import):
        imported |= {a.name.split(".")[0] for a in n.names}
    elif isinstance(n, ast.ImportFrom) and n.module:
        imported.add(n.module.split(".")[0])
check("generate does not import llmauth_ir", "llmauth_ir" not in imported)
check("generate does not import evaluation modules",
      not (imported & {"llmauth_taxonomy", "llmauth_corruption"}))

print("\n=== 2. Basic sweep ===")
with tempfile.TemporaryDirectory() as td:
    log = G.RunLog(Path(td) / "runs.jsonl")
    be = {"mock-1": G.MockBackend("mock-1")}
    res = G.run_sweep(make_plans(), be, log, progress_every=99)
    check("all runs executed", res["executed"] == 3, str(res["executed"]))
    check("all runs ok", res["ok"] == 3)
    check("log length matches", len(log) == 3)

    rec = json.loads(open(log.path).readline())
    check("raw_response captured verbatim", "rules" in rec["raw_response"])
    check("prompt hash recorded", len(rec["prompt_sha256"]) == 64)
    check("model version recorded", rec["model_version"] == "mock-1-v0")
    check("timestamps recorded", rec["started_at_utc"] and rec["finished_at_utc"])
    check("requested params recorded",
          rec["request_params"]["requested"]["seed"] in (1, 2, 3))
    check("applied params recorded separately",
          rec["request_params"]["applied"]["temperature_applied"] is True)
    check("token counts recorded", rec["input_tokens"] > 0)

print("\n=== 3. Resume after simulated Colab disconnect ===")
with tempfile.TemporaryDirectory() as td:
    path = Path(td) / "runs.jsonl"
    plans = make_plans(seeds=(1, 2, 3, 4, 5))

    log1 = G.RunLog(path)
    be1 = G.MockBackend("mock-1")
    G.run_sweep(plans[:2], {"mock-1": be1}, log1, progress_every=99)
    check("first session ran 2", be1.calls == 2, str(be1.calls))

    # session dies; new RunLog object rescans the file from disk
    log2 = G.RunLog(path)
    be2 = G.MockBackend("mock-1")
    res = G.run_sweep(plans, {"mock-1": be2}, log2, progress_every=99)
    check("resume skips completed runs", res["skipped"] == 2, str(res["skipped"]))
    check("resume executes only the remainder", be2.calls == 3, str(be2.calls))
    check("total runs correct", res["runs"] == 5, str(res["runs"]))

print("\n=== 4. Truncated final line survives ===")
with tempfile.TemporaryDirectory() as td:
    path = Path(td) / "runs.jsonl"
    log = G.RunLog(path)
    G.run_sweep(make_plans(seeds=(1, 2)), {"mock-1": G.MockBackend()},
                log, progress_every=99)
    with open(path, "a") as fh:
        fh.write('{"key": {"corpus_id": "bank_mark')  # killed mid-write
    log2 = G.RunLog(path)
    check("truncated line skipped, not fatal", len(log2) == 2, str(len(log2)))

print("\n=== 5. Failures are recorded, never re-seeded ===")
with tempfile.TemporaryDirectory() as td:
    log = G.RunLog(Path(td) / "runs.jsonl")
    be = G.MockBackend("mock-1", fail_seeds=[2])
    res = G.run_sweep(make_plans(seeds=(1, 2, 3)), {"mock-1": be}, log,
                      progress_every=99)
    check("failed run still logged", res["runs"] == 3, str(res["runs"]))
    check("failure counted", res["failed"] == 1, str(res["failed"]))
    check("successes counted", res["ok"] == 2)

    recs = [json.loads(l) for l in open(log.path) if l.strip()]
    bad = [r for r in recs if not r["ok"]][0]
    check("failed run has null raw_response", bad["raw_response"] is None)
    check("failure reason recorded", "simulated backend failure" in bad["failure_reason"])
    check("retried MAX_ATTEMPTS times", bad["attempts"] == G.MAX_ATTEMPTS)
    check("failed seed preserved, not substituted", bad["key"]["seed"] == 2)
    check("failed run records requested params",
          bad["request_params"]["requested"]["seed"] == 2)
    seeds = sorted(r["key"]["seed"] for r in recs)
    check("no duplicate or replacement seeds", seeds == [1, 2, 3], str(seeds))

print("\n=== 6. Run keys and variants ===")
k1 = G.RunKey("bank_marketing", "A4", "m", 1, 0.7)
k2 = G.RunKey("bank_marketing", "A4", "m", 1, 0.0)
k3 = G.RunKey("bank_marketing", "A4", "m", 1, 0.7, variant="no_warning")
check("temperature is part of run identity", k1.as_str() != k2.as_str())
check("variant is part of run identity", k1.as_str() != k3.as_str())
check("identical keys collide as intended",
      k1.as_str() == G.RunKey("bank_marketing", "A4", "m", 1, 0.7).as_str())

print("\n=== 7. Cost accounting ===")
check("unknown model returns None", G.estimate_cost("nope", 1000, 500) is None)
G.PRICING["priced-model"] = (3.0, 15.0)
c = G.estimate_cost("priced-model", 1_000_000, 1_000_000)
check("cost computed from per-1M pricing", abs(c - 18.0) < 1e-9, str(c))
check("missing token counts return None",
      G.estimate_cost("priced-model", None, 500) is None)

print("\n=== 8. Missing backend fails loudly ===")
with tempfile.TemporaryDirectory() as td:
    log = G.RunLog(Path(td) / "runs.jsonl")
    try:
        G.run_sweep(make_plans(model_id="unregistered"), {}, log)
        check("unregistered backend raises", False)
    except KeyError:
        check("unregistered backend raises", True)

print("\n=== 9. purge_failures is auditable ===")
with tempfile.TemporaryDirectory() as td:
    path = Path(td) / "runs.jsonl"
    log = G.RunLog(path)
    G.run_sweep(make_plans(seeds=(1, 2, 3)),
                {"mock-1": G.MockBackend("mock-1", fail_seeds=[2])},
                log, progress_every=99)
    try:
        log.purge_failures(reason="  ")
        check("purge requires a reason", False)
    except ValueError:
        check("purge requires a reason", True)
    n = log.purge_failures(reason="adapter bug: API rejected a parameter")
    check("removes only failures", n == 1 and len(log) == 2, f"n={n} len={len(log)}")
    check("archives removed records", path.with_suffix(".purged.jsonl").exists())
    check("writes an audit entry", path.with_suffix(".purgelog.jsonl").exists())
    res = G.run_sweep(make_plans(seeds=(1, 2, 3)),
                      {"mock-1": G.MockBackend("mock-1")}, log, progress_every=99)
    check("only the purged cell re-runs",
          res["executed"] == 1 and res["skipped"] == 2,
          f"exec={res['executed']} skip={res['skipped']}")

print("\n" + "=" * 60)
print("ALL CHECKS PASSED" if not FAILS else f"FAILURES: {FAILS}")
print("=" * 60)
