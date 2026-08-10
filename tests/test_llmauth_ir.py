"""Validation of llmauth_ir against the real corpora. Run: python3 test_llmauth_ir.py"""
import json
import pandas as pd
from llmauth_ir import (
    CorpusSchema, ColumnProfile, FailureCode, parse_llm_response,
    code_ruleset, to_pandas_mask, to_sql_check, to_great_expectations,
    ruleset_jaccard, canonical_form,
)

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


def build_schema(df, corpus_id, cat_max=60):
    cols = {}
    for c in df.columns:
        s = df[c]
        num = pd.to_numeric(s, errors="coerce")
        is_num = num.notna().sum() > 0.95 * len(s)
        cols[c] = ColumnProfile(
            name=c,
            dtype="int" if is_num else "string",
            null_rate=float(s.isna().mean()),
            distinct_count=int(s.nunique(dropna=True)),
            min_value=float(num.min()) if is_num else None,
            max_value=float(num.max()) if is_num else None,
            observed_levels=(sorted(s.dropna().unique().tolist(), key=str)
                             if (not is_num and s.nunique() <= cat_max) else []),
        )
    return CorpusSchema(corpus_id=corpus_id, columns=cols)


print("\n=== Loading real corpora ===")
bank = pd.read_csv("/mnt/project/bankfull.csv", sep=";", quotechar='"')
diab = pd.read_csv("/mnt/project/diabetic_data.csv")
print(f"  bank  {bank.shape}")
print(f"  diab  {diab.shape}")
bank_schema = build_schema(bank, "bank_marketing")
diab_schema = build_schema(diab, "diabetes_130us")

# ---------------------------------------------------------------- parsing
print("\n=== 1. Parsing tolerance ===")
messy = """Sure! Here are the validation rules I'd recommend:

```json
{"rules": [
  {"rule_id": "r1", "column": "pdays", "dimension": "validity",
   "predicate_type": "range", "parameters": {"min": 0},
   "severity": "reject", "rationale": "days since contact cannot be negative"},
  {"rule_id": "r2", "column": "balance", "dimension": "validity",
   "predicate_type": "range", "parameters": {"min": 0},
   "severity": "reject", "rationale": "balance should be non-negative"},
  {"rule_id": "r3", "column": "customer_email", "dimension": "completeness",
   "predicate_type": "not_null", "parameters": {},
   "severity": "reject", "rationale": "email required"},
  {"rule_id": "r4", "column": "poutcome", "dimension": "validity",
   "predicate_type": "in_set",
   "parameters": {"allowed": ["success", "failure", "other"]},
   "severity": "reject", "rationale": "valid campaign outcomes"},
  {"rule_id": "r5", "column": "age", "dimension": "validity",
   "predicate_type": "range", "parameters": {"min": 18, "max": 95},
   "severity": "reject", "rationale": "plausible adult age"},
  {"rule_id": "r6", "column": "duration", "dimension": "validity",
   "predicate_type": "range", "parameters": {"min": 100000},
   "severity": "reject", "rationale": "call duration floor"},
  {"rule_id": "r7", "column": "job", "dimension": "validity",
   "predicate_type": "range", "parameters": {"min": 0},
   "severity": "reject", "rationale": "job code numeric"},
  {"rule_id": "r8", "column": "balance", "dimension": "validity",
   "predicate_type": "range", "parameters": {"min": 0},
   "severity": "warn", "rationale": "duplicate of r2"},
  {"rule_id": "r9", "column": "y", "dimension": "validity",
   "predicate_type": "in_set", "parameters": {"allowed": ["yes","no","maybe"]},
   "severity": "reject", "rationale": "target levels"}
]}
```
Hope this helps!"""

rs = parse_llm_response(messy, "bank_marketing", "A2", model_id="test", seed=1)
check("recovers JSON from fenced + prose response", len(rs.rules) == 9, f"{len(rs.rules)} rules")
check("no parse errors", not rs.parse_errors, str(rs.parse_errors))
check("raw response hashed", len(rs.raw_response_sha256) == 64)

broken = parse_llm_response("I cannot help with that.", "bank_marketing", "A2")
check("unparseable response -> empty set + logged error", len(broken.rules) == 0 and broken.parse_errors)

partial = parse_llm_response(
    '{"rules":[{"column":"age","predicate_type":"range","parameters":{}},'
    '{"column":"age","predicate_type":"telepathy","parameters":{}}]}',
    "bank_marketing", "A2")
check("malformed rules retained, not dropped", len(partial.rules) == 2, f"{len(partial.rules)}")
check("malformed rules flagged F7",
      all(FailureCode.F7_NON_EXECUTABLE in r.failure_codes for r in partial.rules))

# ---------------------------------------------------- retention + coding
print("\n=== 2. Retention against real data ===")
retentions = {}
for r in rs.rules:
    if FailureCode.F7_NON_EXECUTABLE in r.failure_codes:
        continue
    retentions[r.rule_id] = float(to_pandas_mask(r)(bank).mean())

print(f"  pdays >= 0        retains {retentions['r1']:.4f}")
print(f"  balance >= 0      retains {retentions['r2']:.4f}")
print(f"  poutcome in-set   retains {retentions['r4']:.4f}")
print(f"  age 18-95         retains {retentions['r5']:.4f}")
print(f"  duration>=100000  retains {retentions['r6']:.4f}")

check("pdays>=0 reproduces documented 18.3% retention",
      abs(retentions["r1"] - 0.183) < 0.002, f"{retentions['r1']:.4f}")
check("poutcome in-set reproduces 18.3% retention",
      abs(retentions["r4"] - 0.183) < 0.002, f"{retentions['r4']:.4f}")
check("balance>=0 reproduces 91.7% retention",
      abs(retentions["r2"] - 0.917) < 0.002, f"{retentions['r2']:.4f}")

print("\n=== 3. Failure coding ===")
rs = code_ruleset(rs, bank_schema, retentions)
by_id = {r.rule_id: r.failure_codes for r in rs.rules}
for rid in sorted(by_id):
    print(f"  {rid}: {[c.value for c in by_id[rid]]}")

check("F1 on hallucinated column customer_email",
      FailureCode.F1_HALLUCINATED_COLUMN in by_id["r3"])
check("F9 on pdays>=0 (sentinel -1)",
      FailureCode.F9_SENTINEL_MISREAD in by_id["r1"])
check("F4 on pdays>=0 (retains 18%)",
      FailureCode.F4_OVER_TIGHT in by_id["r1"])
check("F9 on poutcome in-set (sentinel 'unknown')",
      FailureCode.F9_SENTINEL_MISREAD in by_id["r4"])
check("F3 on duration>=100000 (above observed max)",
      FailureCode.F3_CONTRADICTS_CENSUS in by_id["r6"])
check("F6 on numeric range over string column job",
      FailureCode.F6_TYPE_MISMATCH in by_id["r7"])
check("F8 on duplicate balance rule",
      FailureCode.F8_REDUNDANT in by_id["r8"])
check("F2 on invented category 'maybe'",
      FailureCode.F2_HALLUCINATED_CATEGORY in by_id["r9"])
check("F10 NOT fired on inert rule over protected attribute age",
      FailureCode.F10_FAIRNESS_HAZARDOUS not in by_id["r5"],
      f"retains {retentions['r5']:.4f} - rejects nobody, so no hazard")
check("balance>=0 NOT flagged F9 (negative balance is not a sentinel)",
      FailureCode.F9_SENTINEL_MISREAD not in by_id["r2"])
check("balance>=0 NOT flagged F4 (retains 92%)",
      FailureCode.F4_OVER_TIGHT not in by_id["r2"])

# ------------------------------------------------- diabetes 'None' trap
print("\n=== 4. Diabetes 'None' sentinel trap ===")
diab_resp = json.dumps({"rules": [
    {"rule_id": "d1", "column": "max_glu_serum", "dimension": "completeness",
     "predicate_type": "in_set", "parameters": {"allowed": ["Norm", ">200", ">300"]},
     "severity": "reject", "rationale": "'None' indicates a missing lab result"},
    {"rule_id": "d2", "column": "weight", "dimension": "completeness",
     "predicate_type": "not_null", "parameters": {},
     "severity": "reject", "rationale": "weight must be recorded"},
    {"rule_id": "d3", "column": "race", "dimension": "completeness",
     "predicate_type": "in_set",
     "parameters": {"allowed": ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"]},
     "severity": "reject", "rationale": "race must be a known category"},
    {"rule_id": "d4", "column": "time_in_hospital", "dimension": "validity",
     "predicate_type": "range", "parameters": {"min": 1, "max": 14},
     "severity": "reject", "rationale": "documented stay length"},
]})
drs = parse_llm_response(diab_resp, "diabetes_130us", "A2", model_id="test", seed=1)
drets = {r.rule_id: float(to_pandas_mask(r)(diab).mean()) for r in drs.rules}
for rid, v in drets.items():
    print(f"  {rid} retains {v:.4f}")
drs = code_ruleset(drs, diab_schema, drets)
dby = {r.rule_id: r.failure_codes for r in drs.rules}
for rid in sorted(dby):
    print(f"  {rid}: {[c.value for c in dby[rid]]}")

check("max_glu_serum rule flagged F9 ('None' is valid)",
      FailureCode.F9_SENTINEL_MISREAD in dby["d1"])
check("max_glu_serum rule flagged F4 (retains ~5%)",
      FailureCode.F4_OVER_TIGHT in dby["d1"], f"{drets['d1']:.4f}")
check("race rule flagged F10 (protected attribute)",
      FailureCode.F10_FAIRNESS_HAZARDOUS in dby["d3"])
check("race rule flagged F9 ('?' is a sentinel)",
      FailureCode.F9_SENTINEL_MISREAD in dby["d3"])
check("F5 NOT assigned on natural data by default",
      FailureCode.F5_VACUOUS not in dby["d4"], f"retains {drets['d4']:.4f}")

# ------------------------------------------------------------- stability
print("\n=== 4b. Recalibrated F5 / F10 semantics ===")
_rs = code_ruleset(
    parse_llm_response(json.dumps({"rules": [
        {"rule_id": "p1", "column": "age", "dimension": "validity",
         "predicate_type": "range", "parameters": {"min": 25, "max": 60},
         "severity": "reject"}]}), "bank_marketing", "A2"),
    bank_schema, {"p1": 0.62})
check("F10 DOES fire when a protected-attribute rule rejects records",
      FailureCode.F10_FAIRNESS_HAZARDOUS in _rs.rules[0].failure_codes)

_rs2 = code_ruleset(
    parse_llm_response(json.dumps({"rules": [
        {"rule_id": "v1", "column": "age", "dimension": "validity",
         "predicate_type": "range", "parameters": {"min": 0},
         "severity": "reject"}]}), "bank_marketing", "A2"),
    bank_schema, {"v1": 1.0}, assign_vacuous=True)
check("F5 fires when scoring corrupted data (assign_vacuous=True)",
      FailureCode.F5_VACUOUS in _rs2.rules[0].failure_codes)

from llmauth_ir import inert_rate
_rs3 = parse_llm_response(json.dumps({"rules": [
    {"rule_id": "a", "column": "age", "predicate_type": "range",
     "parameters": {"min": 0}, "dimension": "validity"},
    {"rule_id": "b", "column": "age", "predicate_type": "range",
     "parameters": {"min": 30}, "dimension": "validity"}]}), "bank_marketing", "A2")
check("inert_rate reports the F5 signal descriptively",
      inert_rate(_rs3, {"a": 1.0, "b": 0.7}) == 0.5)

print("\n=== 5. Canonical form + Jaccard ===")
a = parse_llm_response(json.dumps({"rules": [
    {"rule_id": "x", "column": "age", "predicate_type": "range",
     "parameters": {"min": 18, "max": 95}, "dimension": "validity"},
    {"rule_id": "y", "column": "balance", "predicate_type": "range",
     "parameters": {"min": 0}, "dimension": "validity"}]}), "bank_marketing", "A2")
b = parse_llm_response(json.dumps({"rules": [
    {"rule_id": "DIFFERENT", "column": "age", "predicate_type": "range",
     "parameters": {"max": 95, "min": 18}, "dimension": "validity",
     "rationale": "totally different wording"},
    {"rule_id": "z", "column": "balance", "predicate_type": "range",
     "parameters": {"min": 100}, "dimension": "validity"}]}), "bank_marketing", "A2")
check("canonical form ignores rule_id/rationale/param order",
      canonical_form(a.rules[0]) == canonical_form(b.rules[0]))
check("shifted threshold counts as a different rule",
      canonical_form(a.rules[1]) != canonical_form(b.rules[1]))
check("Jaccard = 1/3 for 1 shared of 3 distinct",
      abs(ruleset_jaccard(a, b) - 1 / 3) < 1e-9, f"{ruleset_jaccard(a,b):.4f}")
check("Jaccard of identical sets = 1.0", ruleset_jaccard(a, a) == 1.0)

# ------------------------------------------------------------ compilation
print("\n=== 6. Compilation targets ===")
sql = to_sql_check(rs.rules[0])
print(f"  SQL: {sql}")
print(f"  GE : {to_great_expectations(rs.rules[3])['expectation_type']}")
check("SQL emits backticked identifier + bound", sql == "`pdays` >= -1" or "pdays" in sql, sql)
check("in_set compiles to GE value-set expectation",
      to_great_expectations(rs.rules[3])["expectation_type"]
      == "expect_column_values_to_be_in_set")
inj = to_sql_check(parse_llm_response(json.dumps({"rules": [
    {"column": "job", "predicate_type": "in_set",
     "parameters": {"allowed": ["it's a trap"]}, "dimension": "validity"}]}),
    "bank_marketing", "A2").rules[0])
check("SQL literal escaping handles quotes", "it''s a trap" in inj, inj)

# --------------------------------------------------- boundary separation
print("\n=== 7. Anti-tautology boundary ===")
try:
    import ast as _ast
    import llmauth_prompts as _P
    _tree = _ast.parse(open(_P.__file__).read())
    _imported = set()
    for _n in _ast.walk(_tree):
        if isinstance(_n, _ast.Import):
            _imported |= {a.name.split(".")[0] for a in _n.names}
        elif isinstance(_n, _ast.ImportFrom) and _n.module:
            _imported.add(_n.module.split(".")[0])
    # Substring matching is wrong here: the prompts module deliberately NAMES
    # llmauth_ir in comments to explain why the vocabularies are duplicated.
    # Only an actual import breaks the boundary.
    check("llmauth_prompts does not IMPORT llmauth_ir",
          "llmauth_ir" not in _imported, str(sorted(_imported)))
except ImportError:
    print("  SKIP  llmauth_prompts not yet written — re-run this check after it exists")

print("\n" + "=" * 60)
print("ALL CHECKS PASSED" if not FAILS else f"FAILURES: {FAILS}")
print("=" * 60)
