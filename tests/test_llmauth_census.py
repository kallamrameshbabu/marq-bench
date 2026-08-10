"""Tests for llmauth_census. Run: python3 test_llmauth_census.py"""
import ast
import tempfile
from pathlib import Path

import pandas as pd
import llmauth_census as C
import llmauth_prompts as P

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


print("\n=== 1. Anti-tautology boundary ===")
tree = ast.parse(open(C.__file__).read())
imported = set()
for n in ast.walk(tree):
    if isinstance(n, ast.Import):
        imported |= {a.name.split(".")[0] for a in n.names}
    elif isinstance(n, ast.ImportFrom) and n.module:
        imported.add(n.module.split(".")[0])
print(f"  imports: {sorted(imported)}")
check("census does not import llmauth_ir", "llmauth_ir" not in imported)
check("census may import llmauth_prompts", "llmauth_prompts" in imported)

print("\n=== 2. The 'None' bug guard ===")
check("pandas default NA list does contain 'None'",
      "None" in pd._libs.parsers.STR_NA_VALUES)
naive = pd.read_csv("/mnt/project/diabetic_data.csv")
check("naive read_csv DESTROYS 96,420 'None' values",
      naive.max_glu_serum.isna().sum() == 96420, str(naive.max_glu_serum.isna().sum()))

diab, prov = C.load_corpus("diabetes_130us", "/mnt/project/diabetic_data.csv")
check("load_corpus preserves 'None' as a value",
      (diab.max_glu_serum == "None").sum() == 96420)
check("load_corpus reports zero nulls in max_glu_serum",
      diab.max_glu_serum.isna().sum() == 0)
check("load_corpus preserves '?' in weight",
      (diab.weight == "?").sum() == 98569)
check("provenance records file hash", len(prov["file_sha256"]) == 64)

try:
    C.load_corpus("some_new_corpus", "/mnt/project/bank.csv")
    check("unregistered corpus refused", False)
except KeyError:
    check("unregistered corpus refused", True)

bank, bprov = C.load_corpus("bank_marketing", "/mnt/project/bankfull.csv")
check("bank loads with correct shape", bank.shape == (45211, 17), str(bank.shape))
check("bank 'unknown' preserved in poutcome",
      (bank.poutcome == "unknown").sum() == 36959)
check("bank pdays -1 preserved", (bank.pdays == -1).sum() == 36954)

print("\n=== 3. Split reproducibility and disjointness ===")
auth, ev, man = C.make_split(bank, "bank_marketing")
print(f"  authoring {man.authoring_rows:,} / evaluation {man.evaluation_rows:,}")
check("split is 20/80", abs(man.authoring_rows / 45211 - 0.20) < 0.001)
check("splits are disjoint", len(set(auth.index) & set(ev.index)) == 0)
check("splits are exhaustive", man.authoring_rows + man.evaluation_rows == 45211)

auth2, ev2, man2 = C.make_split(bank, "bank_marketing")
check("split reproducible under same seed",
      man.authoring_index_sha256 == man2.authoring_index_sha256)
_, _, man3 = C.make_split(bank, "bank_marketing", seed=999)
check("different seed -> different split",
      man.authoring_index_sha256 != man3.authoring_index_sha256)
check("manifest serializes", "authoring_index_sha256" in man.to_json())

big = pd.DataFrame({"x": range(100_000)})
_, _, bman = C.make_split(big, "test", subsample_rows=10_000)
check("subsample honoured", bman.authoring_rows + bman.evaluation_rows == 10_000)
check("subsample seed recorded", bman.subsample_seed is not None)

print("\n=== 4. Profiling: no cardinality gate ===")
facts = C.profile_authoring_split(auth, "bank_marketing")
by_name = {c.name: c for c in facts.columns}
pdays = by_name["pdays"]
print(f"  pdays: {pdays.distinct_count} distinct, {len(pdays.top_levels)} levels listed")
print(f"  pdays top level: {pdays.top_levels[0]}")
check("high-cardinality numeric still gets a level list",
      len(pdays.top_levels) > 0, f"{pdays.distinct_count} distinct")
check("pdays modal value is -1", pdays.top_levels[0][0] == -1)
modal_rate = pdays.top_levels[0][1] / facts.row_count
check("modal concentration ~81.7% on authoring split",
      abs(modal_rate - 0.817) < 0.02, f"{modal_rate:.4f}")
check("every column has a level list",
      all(len(c.top_levels) > 0 for c in facts.columns))
check("numeric dtype inferred for pdays", pdays.dtype == "int")
check("string dtype inferred for poutcome", by_name["poutcome"].dtype == "string")
check("quantiles computed for numerics", "p50" in pdays.quantiles)

dauth, dev, _ = C.make_split(diab, "diabetes_130us")
dfacts = C.profile_authoring_split(dauth, "diabetes_130us")
dby = {c.name: c for c in dfacts.columns}
mgs = dby["max_glu_serum"]
check("max_glu_serum profiled with 'None' as a LEVEL, not a null",
      mgs.top_levels[0][0] == "None" and mgs.null_rate == 0.0,
      f"top={mgs.top_levels[0]}, null_rate={mgs.null_rate}")
check("age profiled as string (bucketed)", dby["age"].dtype == "string")

print("\n=== 5. Census reaches the prompt ===")
import re
prompt = P.build_prompt("A4", facts)
# The authoring split is a 20% sample, so its modal rate is near but not equal
# to the full-corpus 81.74%. Assert the structure and a plausible range rather
# than a hardcoded figure.
m = re.search(r"^\s*-1: ([\d,]+) \(([\d.]+)%\)$", prompt.user, re.M)
check("A4 prompt contains the pdays modal frequency line",
      m is not None, m.group(0).strip() if m else "line absent")
check("modal frequency in prompt is > 75%",
      m is not None and float(m.group(2)) > 75.0,
      f"{m.group(2)}%" if m else "n/a")
check("A4 prompt passes the leakage guard", True)  # build_prompt would have raised
mgs_prompt = P.build_prompt("A4", dfacts)
check("A4 diabetes prompt shows 'None' as a value",
      "'None'" in mgs_prompt.user)

print("\n=== 6. Persistence round-trip ===")
with tempfile.TemporaryDirectory() as td:
    p = Path(td) / "facts.json"
    h = C.save_facts(facts, p)
    back = C.load_facts(p)
    check("round-trip preserves column count",
          len(back.columns) == len(facts.columns))
    check("round-trip preserves modal value",
          {c.name: c for c in back.columns}["pdays"].top_levels[0][0] == -1)
    check("round-trip produces identical prompt",
          P.build_prompt("A4", back).sha256 == prompt.sha256)
    check("save returns content hash", len(h) == 64)

print("\n" + "=" * 60)
print("ALL CHECKS PASSED" if not FAILS else f"FAILURES: {FAILS}")
print("=" * 60)
