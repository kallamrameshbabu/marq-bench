"""Tests for llmauth_prompts. Run: python3 test_llmauth_prompts.py"""
import ast
import pandas as pd
import llmauth_prompts as P
import llmauth_ir as IR

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


# ---------------------------------------------------- boundary + vocab sync
print("\n=== 1. Anti-tautology boundary ===")
tree = ast.parse(open(P.__file__).read())
imported = set()
for node in ast.walk(tree):
    if isinstance(node, ast.Import):
        imported |= {a.name.split(".")[0] for a in node.names}
    elif isinstance(node, ast.ImportFrom) and node.module:
        imported.add(node.module.split(".")[0])
print(f"  imports: {sorted(imported)}")
forbidden = {"llmauth_ir", "llmauth_corruption", "ztlf_corruption", "llmauth_taxonomy"}
check("prompts imports no evaluation module", not (imported & forbidden),
      str(imported & forbidden))

src = open(P.__file__).read()
check("sentinel registry name absent from prompts source", "SENTINEL_REGISTRY" not in src)
check("protected attribute list absent from prompts source",
      "PROTECTED_ATTRIBUTES" not in src)
check("failure codes absent from prompts source",
      not any(f"F{i}_" in src for i in range(1, 11)))

print("\n=== 2. Vocabulary sync with IR (duplication is intentional) ===")
check("dimensions in sync",
      set(P.DIMENSION_VOCAB) == {d.value for d in IR.Dimension})
check("predicates in sync",
      set(P.PREDICATE_VOCAB) == {p.value for p in IR.PredicateType})
check("severities in sync",
      set(P.SEVERITY_VOCAB) == {s.value for s in IR.Severity})

# ------------------------------------------------------------ build facts
print("\n=== 3. Building facts from real Bank data ===")
bank = pd.read_csv("/mnt/project/bankfull.csv", sep=";", quotechar='"')
DOCS = {
    "pdays": "number of days that passed by after the client was last contacted "
             "from a previous campaign (numeric, -1 means client was not "
             "previously contacted)",
    "balance": "average yearly balance, in euros (numeric)",
    "poutcome": "outcome of the previous marketing campaign "
                "(categorical: unknown, other, failure, success)",
}
cols = []
for c in bank.columns:
    s = bank[c]
    num = pd.to_numeric(s, errors="coerce")
    is_num = num.notna().sum() > 0.95 * len(s)
    cols.append(P.ColumnFacts(
        name=c,
        dtype="int" if is_num else "string",
        doc=DOCS.get(c, ""),
        null_rate=float(s.isna().mean()),
        distinct_count=int(s.nunique(dropna=True)),
        min_value=float(num.min()) if is_num else None,
        max_value=float(num.max()) if is_num else None,
        quantiles={"p50": float(num.median())} if is_num else {},
        # top_levels populated for EVERY column, numeric included --
        # modal concentration is what carries the A4 signal.
        top_levels=[(v, int(n)) for v, n in s.value_counts().head(20).items()],
    ))
facts = P.CorpusFacts(corpus_id="bank_marketing", row_count=len(bank),
                      columns=cols, table_doc="Direct marketing campaigns of a "
                                              "Portuguese banking institution.")
rows = bank.head(500).to_dict("records")
print(f"  {len(cols)} columns, {facts.row_count:,} rows")

# --------------------------------------------------------- ladder is nested
print("\n=== 4. Information ladder is strictly nested ===")
p2 = P.build_prompt("A2", facts)
p3 = P.build_prompt("A3", facts)
p4 = P.build_prompt("A4", facts)
p5 = P.build_prompt("A5", facts, rows)
sizes = {k: len(v.user) for k, v in [("A2", p2), ("A3", p3), ("A4", p4), ("A5", p5)]}
print(f"  payload chars: {sizes}")
check("A2 < A3", sizes["A2"] < sizes["A3"])
check("A2 < A4", sizes["A2"] < sizes["A4"])
check("A4 < A5", sizes["A4"] < sizes["A5"])
check("A2 contains no statistics", "%" not in p2.user.split("Parameters by")[0])
check("A5 contains the A4 census block", "Profile computed on" in p5.user)
check("A3 does NOT contain the census", "Profile computed on" not in p3.user)
check("A4 does NOT contain documentation", "previously contacted" not in p4.user)

# ------------------------------------------------------------- determinism
print("\n=== 5. Determinism and hashing ===")
check("identical inputs -> identical hash",
      P.build_prompt("A5", facts, rows).sha256 == p5.sha256)
check("different condition -> different hash", p4.sha256 != p5.sha256)
check("warning ablation changes hash",
      P.build_prompt("A2", facts, warning_clause=False).sha256 != p2.sha256)
check("sample seed changes hash",
      P.build_prompt("A5", facts, rows, sample_seed=999).sha256 != p5.sha256)
check("sample_rows deterministic under fixed seed",
      P.sample_rows(rows, 20, seed=42) == P.sample_rows(rows, 20, seed=42))
check("sample_rows differs under different seed",
      P.sample_rows(rows, 20, seed=42) != P.sample_rows(rows, 20, seed=43))
check("A5 records its sample seed", p5.sample_seed == 20260807)
check("A2 records no sample seed", p2.sample_seed is None)

# ---------------------------------------------------------- leakage guard
print("\n=== 6. Leakage guard ===")
try:
    P.assert_no_leakage("pdays: -1 is a sentinel for no prior contact")
    check("guard catches 'sentinel'", False)
except P.LeakageError:
    check("guard catches 'sentinel'", True)
try:
    P.assert_no_leakage("race is a protected attribute")
    check("guard catches 'protected attribute'", False)
except P.LeakageError:
    check("guard catches 'protected attribute'", True)
try:
    P.assert_no_leakage("pdays: min -1, 81.74% of values are -1")
    check("guard permits pure frequency statements", True)
except P.LeakageError as e:
    check("guard permits pure frequency statements", False, str(e))

check("census block reports -1 frequency without interpreting it",
      "-1" in p4.user and "sentinel" not in p4.user.lower())
check("A4 census surfaces pdays modal concentration at 81.74%",
      "81.74%" in p4.user, "modal -1 frequency must reach the prompt")
check("A4 census surfaces poutcome 'unknown' at 81.75%",
      "81.75%" in p4.user)
check("A3 documentation is NOT sanitised (condition depends on it)",
      "not\n previously contacted" in p3.user or "previously contacted" in p3.user)

# ------------------------------------------------------------ guardrails
print("\n=== 7. Misuse guardrails ===")
for bad, label in [
    (lambda: P.build_payload("A9", facts), "unknown condition rejected"),
    (lambda: P.build_payload("A5", facts), "A5 without rows rejected"),
    (lambda: P.build_payload("A3", P.CorpusFacts("x", 10, [
        P.ColumnFacts("a", "int")])), "A3 without docs rejected"),
]:
    try:
        bad()
        check(label, False)
    except ValueError:
        check(label, True)

print("\n--- A4 census excerpt (pdays) ---")
for line in p4.user.splitlines():
    if "pdays" in line or (line.strip().startswith("-1")):
        print(" ", line)

print("\n" + "=" * 60)
print("ALL CHECKS PASSED" if not FAILS else f"FAILURES: {FAILS}")
print("=" * 60)
