"""Tests for llmauth_checkpoint. Run: python3 test_llmauth_checkpoint.py"""
import tempfile, json
from pathlib import Path
import pandas as pd
from llmauth_checkpoint import Checkpoint

FAILS = []
def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond: FAILS.append(name)

print("\n=== 1. Build once, reuse thereafter ===")
with tempfile.TemporaryDirectory() as td:
    ck = Checkpoint(td, verbose=False)
    calls = {"n": 0}
    def build():
        calls["n"] += 1
        return {"answer": 42}
    v1, c1 = ck.step("thing", "json", build)
    v2, c2 = ck.step("thing", "json", build)
    v3, c3 = ck.step("thing", "json", build)
    check("build ran exactly once", calls["n"] == 1, str(calls["n"]))
    check("first call not cached", c1 is False)
    check("later calls cached", c2 and c3)
    check("value survives round-trip", v3 == {"answer": 42})

print("\n=== 2. Delete file -> rebuilds (the stated workflow) ===")
with tempfile.TemporaryDirectory() as td:
    ck = Checkpoint(td, verbose=False)
    calls = {"n": 0}
    def build():
        calls["n"] += 1
        return {"v": calls["n"]}
    ck.step("x", "json", build)
    ck.step("x", "json", build)
    check("cached after first build", calls["n"] == 1)
    ck.path_for("x", "json").unlink()          # user deletes it by hand in Drive
    v, cached = ck.step("x", "json", build)
    check("rebuilds after manual delete", calls["n"] == 2 and not cached)
    check("new value returned", v == {"v": 2})
    ck.invalidate("x", "json")
    ck.step("x", "json", build)
    check("invalidate() also forces rebuild", calls["n"] == 3)
    check("force=True forces rebuild",
          ck.step("x", "json", build, force=True)[1] is False and calls["n"] == 4)

print("\n=== 3. Atomic writes ===")
with tempfile.TemporaryDirectory() as td:
    ck = Checkpoint(td, verbose=False)
    def bad_build():
        raise RuntimeError("build blew up")
    try:
        ck.step("half", "json", bad_build)
    except RuntimeError:
        pass
    check("failed build leaves no artifact", not ck.exists("half", "json"))
    check("failed build leaves no .tmp",
          not list(Path(td).glob("*.tmp")), str(list(Path(td).glob("*.tmp"))))

print("\n=== 4. Metadata sidecar ===")
with tempfile.TemporaryDirectory() as td:
    ck = Checkpoint(td, verbose=False)
    ck.step("m", "json", lambda: {"a": 1}, code_version="census-1.0.0")
    i = ck.info("m", "json")
    check("records write time", bool(i.written_at_utc))
    check("records code version", i.code_version == "census-1.0.0")
    check("records content hash", len(i.content_sha256 or "") == 64)
    check("reports missing artifact", ck.info("nope", "json").exists is False)

print("\n=== 5. Kinds ===")
with tempfile.TemporaryDirectory() as td:
    ck = Checkpoint(td, verbose=False)
    df = pd.DataFrame({"a": [1, 2, 3], "b": list("xyz")})
    back, _ = ck.step("frame", "parquet", lambda: df)
    back2, cached = ck.step("frame", "parquet", lambda: df)
    check("parquet round-trip", cached and back2.equals(df))
    rows = [{"i": i} for i in range(3)]
    _, _ = ck.step("lines", "jsonl", lambda: rows)
    back3, cached3 = ck.step("lines", "jsonl", lambda: rows)
    check("jsonl round-trip", cached3 and back3 == rows)
    _, _ = ck.step("note", "text", lambda: "hello")
    check("text round-trip", ck.step("note", "text", lambda: "hello")[0] == "hello")

print("\n=== 6. Guardrails ===")
with tempfile.TemporaryDirectory() as td:
    ck = Checkpoint(td, verbose=False)
    for fn, label in [
        (lambda: ck.path_for("a", "bogus"), "unknown kind rejected"),
        (lambda: ck.path_for("sub/dir", "json"), "path separator rejected"),
        (lambda: ck.clear_all(), "clear_all needs confirmation"),
    ]:
        try:
            fn(); check(label, False)
        except (ValueError, RuntimeError):
            check(label, True)
    ck.step("keep", "json", lambda: {})
    n = ck.clear_all(confirm="yes, delete everything")
    check("clear_all works with confirmation", n > 0 and not ck.exists("keep", "json"))

print("\n" + "=" * 60)
print("ALL CHECKS PASSED" if not FAILS else f"FAILURES: {FAILS}")
print("=" * 60)
