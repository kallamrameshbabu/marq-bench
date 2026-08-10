"""
llmauth_ir.py — Rule intermediate representation for MARQ-Bench (Paper 2).

A single tool-agnostic representation for data quality rules, whatever authored
them. Every rule set — heuristic baseline (A0/A1) or model-generated (A2-A5) —
is parsed into this IR, validated, failure-coded, and compiled through one
enforcement path. Enforcement is thereby held constant and authorship is the
only manipulated variable, which is what Paper 1's tool-equivalence result
licenses.

ANTI-TAUTOLOGY BOUNDARY
-----------------------
The SENTINEL_REGISTRY and PROTECTED_ATTRIBUTES tables below are EVALUATION
ground truth. They encode documented corpus conventions and are used only to
assign failure codes after rules exist. They must never be reachable from
prompt construction. `llmauth_prompts` must not import this module. A test in
`test_llmauth_ir.py` asserts that separation; keep it passing.

Author: Ramesh Babu Kallam
License: MIT
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Callable, Iterable, Sequence

__all__ = [
    "Dimension", "PredicateType", "Severity", "FailureCode",
    "Rule", "RuleSet", "ColumnProfile", "CorpusSchema",
    "parse_llm_response", "validate_static", "validate_empirical",
    "code_ruleset", "canonical_form", "ruleset_jaccard", "inert_rate",
    "to_pandas_mask", "to_spark_expr", "to_sql_check",
    "to_great_expectations", "SENTINEL_REGISTRY", "PROTECTED_ATTRIBUTES",
]

SCHEMA_VERSION = "1.0.0"


# --------------------------------------------------------------------------
# Controlled vocabularies
# --------------------------------------------------------------------------

class Dimension(str, Enum):
    COMPLETENESS = "completeness"
    VALIDITY = "validity"
    CONSISTENCY = "consistency"
    UNIQUENESS = "uniqueness"
    ACCURACY = "accuracy"
    TIMELINESS = "timeliness"


class PredicateType(str, Enum):
    NOT_NULL = "not_null"
    IN_SET = "in_set"
    RANGE = "range"
    REGEX = "regex"
    UNIQUE = "unique"
    CROSS_COLUMN = "cross_column"
    TYPE = "type"


class Severity(str, Enum):
    REJECT = "reject"
    QUARANTINE = "quarantine"
    WARN = "warn"


class FailureCode(str, Enum):
    """Pre-specified taxonomy. Fixed before any output was inspected."""
    F1_HALLUCINATED_COLUMN = "F1"
    F2_HALLUCINATED_CATEGORY = "F2"
    F3_CONTRADICTS_CENSUS = "F3"
    F4_OVER_TIGHT = "F4"
    F5_VACUOUS = "F5"
    F6_TYPE_MISMATCH = "F6"
    F7_NON_EXECUTABLE = "F7"
    F8_REDUNDANT = "F8"
    F9_SENTINEL_MISREAD = "F9"
    F10_FAIRNESS_HAZARDOUS = "F10"


OVER_TIGHT_THRESHOLD = 0.50   # F4: single rule retaining < 50%
VACUOUS_THRESHOLD = 1.0       # F5: single rule retaining 100%

NUMERIC_PREDICATES = {PredicateType.RANGE}
CATEGORICAL_PREDICATES = {PredicateType.IN_SET, PredicateType.REGEX}


# --------------------------------------------------------------------------
# Evaluation ground truth — NOT for prompt construction
# --------------------------------------------------------------------------

SENTINEL_REGISTRY: dict[str, dict[str, list[Any]]] = {
    "bank_marketing": {
        "pdays": [-1],
        "poutcome": ["unknown"],
        "contact": ["unknown"],
        "job": ["unknown"],
        "education": ["unknown"],
    },
    "diabetes_130us": {
        "weight": ["?"],
        "payer_code": ["?"],
        "medical_specialty": ["?"],
        "race": ["?"],
        "diag_1": ["?"], "diag_2": ["?"], "diag_3": ["?"],
        "max_glu_serum": ["None"],
        "A1Cresult": ["None"],
        "gender": ["Unknown/Invalid"],
    },
    "online_retail_ii": {
        "Customer ID": [None],
        "Description": [None],
    },
    # Verified against yellow_tripdata_2026-05.parquet (n=4,090,836), sha256
    # 9aa5a160...b6750. RatecodeID carries BOTH a true null (23.4%) and an
    # encoded "unknown" (99, 3.4%); both are registered because rejecting
    # either is a sentinel misread. payment_type 5/6 are documented sentinels
    # that do not occur in this month — retained so the registry matches the
    # published dictionary rather than one sample; F9 simply never fires on
    # them here, and that absence is reported.
    # NOTE: payment_type 0 (Flex Fare) is a VALID level, not a sentinel.
    # Omitting it from an in_set rule is over-rejection (F4/excess), not F9.
    "nyc_tlc_yellow": {
        "RatecodeID": [99, None],
        "payment_type": [5, 6],
        "passenger_count": [0, None],
    },
}

PROTECTED_ATTRIBUTES: dict[str, list[str]] = {
    "bank_marketing": ["age", "marital", "job", "education"],
    "diabetes_130us": ["race", "gender", "age"],
    "online_retail_ii": ["Country"],
    # payment_type is protected-adjacent on C4: gating on it differentially
    # removes cash-paying riders, who are not evenly distributed across zones.
    "nyc_tlc_yellow": ["PULocationID", "DOLocationID", "payment_type"],
}


# --------------------------------------------------------------------------
# Corpus description
# --------------------------------------------------------------------------

@dataclass
class ColumnProfile:
    """Observed facts about one column, from the AUTHORING SPLIT only."""
    name: str
    dtype: str                      # "int" | "float" | "string" | "bool" | "timestamp"
    null_rate: float = 0.0
    distinct_count: int | None = None
    min_value: Any = None
    max_value: Any = None
    observed_levels: list[Any] = field(default_factory=list)

    @property
    def is_numeric(self) -> bool:
        return self.dtype in {"int", "float"}


@dataclass
class CorpusSchema:
    corpus_id: str
    columns: dict[str, ColumnProfile]

    @property
    def column_names(self) -> set[str]:
        return set(self.columns)

    def sentinels_for(self, column: str) -> list[Any]:
        return SENTINEL_REGISTRY.get(self.corpus_id, {}).get(column, [])

    def is_protected(self, column: str) -> bool:
        return column in PROTECTED_ATTRIBUTES.get(self.corpus_id, [])


# --------------------------------------------------------------------------
# Rule
# --------------------------------------------------------------------------

@dataclass
class Rule:
    rule_id: str
    column: str
    dimension: Dimension
    predicate_type: PredicateType
    parameters: dict[str, Any] = field(default_factory=dict)
    severity: Severity = Severity.REJECT
    rationale: str = ""
    failure_codes: list[FailureCode] = field(default_factory=list)
    parse_note: str = ""

    @property
    def is_executable(self) -> bool:
        return FailureCode.F7_NON_EXECUTABLE not in self.failure_codes

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["dimension"] = self.dimension.value
        d["predicate_type"] = self.predicate_type.value
        d["severity"] = self.severity.value
        d["failure_codes"] = [c.value for c in self.failure_codes]
        return d


@dataclass
class RuleSet:
    """Rules plus the provenance needed to reproduce them."""
    rules: list[Rule]
    corpus_id: str
    condition: str                  # A0..A5
    model_id: str = "heuristic"
    model_version: str = ""
    seed: int | None = None
    temperature: float | None = None
    generated_at_utc: str = ""
    raw_response_sha256: str = ""
    prompt_sha256: str = ""
    parse_errors: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.rules)

    @property
    def executable(self) -> list[Rule]:
        return [r for r in self.rules if r.is_executable]

    def code_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self.rules:
            for c in r.failure_codes:
                counts[c.value] = counts.get(c.value, 0) + 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["schema_version"] = SCHEMA_VERSION
        d["rules"] = [r.to_dict() for r in self.rules]
        return d


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _extract_json_block(text: str) -> str | None:
    """Recover a JSON object from a model response.

    Tolerant of fenced blocks and prose preamble, because refusing to parse a
    recoverable response would silently bias the sample toward well-behaved
    models. Unrecoverable responses become F7, never a dropped row.
    """
    if not text or not text.strip():
        return None
    for candidate in _FENCE.findall(text):
        if candidate.strip():
            return candidate.strip()
    start = text.find("{")
    if start == -1:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _coerce_enum(raw: Any, enum_cls, default):
    if isinstance(raw, str):
        try:
            return enum_cls(raw.strip().lower())
        except ValueError:
            return None
    return default


def parse_llm_response(
    text: str,
    corpus_id: str,
    condition: str,
    *,
    model_id: str = "unknown",
    model_version: str = "",
    seed: int | None = None,
    temperature: float | None = None,
    generated_at_utc: str = "",
    prompt_sha256: str = "",
) -> RuleSet:
    """Parse a raw model response into a RuleSet.

    Never raises and never silently discards. Malformed rules are retained and
    marked F7 so the denominator in every rate statistic stays honest.
    """
    raw_hash = hashlib.sha256((text or "").encode("utf-8")).hexdigest()
    rs = RuleSet(
        rules=[], corpus_id=corpus_id, condition=condition,
        model_id=model_id, model_version=model_version, seed=seed,
        temperature=temperature, generated_at_utc=generated_at_utc,
        raw_response_sha256=raw_hash, prompt_sha256=prompt_sha256,
    )

    block = _extract_json_block(text or "")
    if block is None:
        rs.parse_errors.append("no JSON object found in response")
        return rs
    try:
        payload = json.loads(block)
    except json.JSONDecodeError as exc:
        rs.parse_errors.append(f"JSON decode error: {exc}")
        return rs

    raw_rules = payload.get("rules") if isinstance(payload, dict) else None
    if not isinstance(raw_rules, list):
        rs.parse_errors.append("payload has no 'rules' list")
        return rs

    for i, raw in enumerate(raw_rules):
        rs.rules.append(_parse_one(raw, i))
    return rs


def _parse_one(raw: Any, index: int) -> Rule:
    fallback = Rule(
        rule_id=f"unparsed_{index}", column="", dimension=Dimension.VALIDITY,
        predicate_type=PredicateType.NOT_NULL,
        failure_codes=[FailureCode.F7_NON_EXECUTABLE],
    )
    if not isinstance(raw, dict):
        fallback.parse_note = f"rule {index} is {type(raw).__name__}, not an object"
        return fallback

    notes: list[str] = []
    column = raw.get("column")
    if not isinstance(column, str) or not column.strip():
        fallback.parse_note = f"rule {index} missing 'column'"
        return fallback

    ptype = _coerce_enum(raw.get("predicate_type"), PredicateType, None)
    if ptype is None:
        fallback.column = column.strip()
        fallback.parse_note = f"unknown predicate_type {raw.get('predicate_type')!r}"
        return fallback

    dim = _coerce_enum(raw.get("dimension"), Dimension, None)
    if dim is None:
        dim = Dimension.VALIDITY
        notes.append(f"dimension {raw.get('dimension')!r} unrecognised; defaulted")

    sev = _coerce_enum(raw.get("severity"), Severity, None)
    if sev is None:
        sev = Severity.REJECT
        notes.append("severity defaulted to reject")

    params = raw.get("parameters")
    if not isinstance(params, dict):
        params = {}
        notes.append("parameters missing or malformed")

    rule = Rule(
        rule_id=str(raw.get("rule_id") or f"rule_{index}"),
        column=column.strip(),
        dimension=dim,
        predicate_type=ptype,
        parameters=params,
        severity=sev,
        rationale=str(raw.get("rationale") or ""),
        parse_note="; ".join(notes),
    )
    if not _params_sufficient(rule):
        rule.failure_codes.append(FailureCode.F7_NON_EXECUTABLE)
        rule.parse_note = (rule.parse_note + "; " if rule.parse_note else "") + \
            f"insufficient parameters for {ptype.value}"
    return rule


def _params_sufficient(rule: Rule) -> bool:
    p, t = rule.parameters, rule.predicate_type
    if t is PredicateType.RANGE:
        return ("min" in p or "max" in p) and all(
            isinstance(p[k], (int, float)) for k in ("min", "max") if k in p
        )
    if t is PredicateType.IN_SET:
        return isinstance(p.get("allowed"), list) and len(p["allowed"]) > 0
    if t is PredicateType.REGEX:
        if not isinstance(p.get("pattern"), str):
            return False
        try:
            re.compile(p["pattern"])
        except re.error:
            return False
        return True
    if t is PredicateType.CROSS_COLUMN:
        return isinstance(p.get("other_column"), str) and isinstance(p.get("op"), str)
    if t is PredicateType.TYPE:
        return isinstance(p.get("expected"), str)
    return True  # not_null, unique need no parameters


# --------------------------------------------------------------------------
# Validation and failure coding
# --------------------------------------------------------------------------

def validate_static(rule: Rule, schema: CorpusSchema) -> list[FailureCode]:
    """Codes assignable from the schema alone: F1, F6.

    F10 is NOT assigned here. Touching a protected attribute is not by itself a
    hazard — a rule that rejects nobody harms nobody. F10 requires an observed
    retention below 1.0 and is assigned in validate_empirical.
    """
    codes: list[FailureCode] = []

    if rule.column not in schema.column_names:
        codes.append(FailureCode.F1_HALLUCINATED_COLUMN)
        return codes  # downstream checks need a real column

    prof = schema.columns[rule.column]

    if rule.predicate_type in NUMERIC_PREDICATES and not prof.is_numeric:
        codes.append(FailureCode.F6_TYPE_MISMATCH)
    if rule.predicate_type is PredicateType.REGEX and prof.is_numeric:
        codes.append(FailureCode.F6_TYPE_MISMATCH)

    if rule.predicate_type is PredicateType.CROSS_COLUMN:
        other = rule.parameters.get("other_column")
        if other not in schema.column_names:
            codes.append(FailureCode.F1_HALLUCINATED_COLUMN)

    return codes


def validate_empirical(
    rule: Rule,
    schema: CorpusSchema,
    retention: float | None = None,
    *,
    assign_vacuous: bool = False,
) -> list[FailureCode]:
    """Codes needing the census or an executed retention rate.

    F2, F3, F9 come from the census. F4 and F10 need retention. F5 is opt-in.

    **F5 (vacuous) is off by default.** On natural, uncorrupted data most
    well-formed rules reject nothing — that is the data having no defects of
    that kind, not the rule being defective. Scored against natural data, F5
    fires on the majority of rules and measures nothing useful. It becomes
    meaningful only against a corpus with injected defects, where a rule that
    rejects nothing has failed to catch what it was written to catch. Pass
    ``assign_vacuous=True`` when scoring corrupted data; report the inert rate
    descriptively otherwise.

    **F10 requires actual rejection.** A rule on a protected attribute that
    retains every record removes nobody from any subgroup. Scored without this
    condition, F10 fired on roughly a fifth of all rules and could not support
    the fairness argument.
    """
    codes: list[FailureCode] = []
    if rule.column not in schema.column_names:
        return codes
    prof = schema.columns[rule.column]

    # F3 — threshold outside the observed range entirely
    if rule.predicate_type is PredicateType.RANGE and prof.is_numeric:
        lo, hi = rule.parameters.get("min"), rule.parameters.get("max")
        if lo is not None and prof.max_value is not None and lo > prof.max_value:
            codes.append(FailureCode.F3_CONTRADICTS_CENSUS)
        elif hi is not None and prof.min_value is not None and hi < prof.min_value:
            codes.append(FailureCode.F3_CONTRADICTS_CENSUS)

    # F2 — allowed set invents levels never observed
    if rule.predicate_type is PredicateType.IN_SET and prof.observed_levels:
        allowed = set(map(_hashable, map(_normalise_value,
                                         rule.parameters.get("allowed", []))))
        observed = set(map(_hashable, map(_normalise_value, prof.observed_levels)))
        if allowed - observed:
            codes.append(FailureCode.F2_HALLUCINATED_CATEGORY)

    # F9 — rule rejects a documented sentinel
    if _rejects_sentinel(rule, schema):
        codes.append(FailureCode.F9_SENTINEL_MISREAD)

    # F4 / F5 / F10 — require executed retention
    if retention is not None:
        if retention < OVER_TIGHT_THRESHOLD:
            codes.append(FailureCode.F4_OVER_TIGHT)
        elif assign_vacuous and retention >= VACUOUS_THRESHOLD:
            codes.append(FailureCode.F5_VACUOUS)

        if schema.is_protected(rule.column) and retention < VACUOUS_THRESHOLD \
                and rule.severity in (Severity.REJECT, Severity.QUARANTINE):
            codes.append(FailureCode.F10_FAIRNESS_HAZARDOUS)

    return codes


def _hashable(v: Any) -> Any:
    return json.dumps(v, sort_keys=True) if isinstance(v, (list, dict)) else v


def _rejects_sentinel(rule: Rule, schema: CorpusSchema) -> bool:
    sentinels = schema.sentinels_for(rule.column)
    if not sentinels:
        return False
    t, p = rule.predicate_type, rule.parameters

    if t is PredicateType.NOT_NULL:
        return any(s is None for s in sentinels)

    if t is PredicateType.IN_SET:
        allowed = set(map(_hashable, p.get("allowed", [])))
        return any(_hashable(s) not in allowed for s in sentinels if s is not None)

    if t is PredicateType.RANGE:
        lo, hi = p.get("min"), p.get("max")
        for s in sentinels:
            if not isinstance(s, (int, float)):
                continue
            if (lo is not None and s < lo) or (hi is not None and s > hi):
                return True
        return False

    if t is PredicateType.REGEX:
        try:
            pat = re.compile(p.get("pattern", ""))
        except re.error:
            return False
        return any(
            isinstance(s, str) and not pat.fullmatch(s) for s in sentinels
        )

    return False


def code_ruleset(
    ruleset: RuleSet,
    schema: CorpusSchema,
    retentions: dict[str, float] | None = None,
    *,
    assign_vacuous: bool = False,
) -> RuleSet:
    """Assign all failure codes in place. Idempotent apart from F7 set at parse.

    ``assign_vacuous`` should be True only when scoring against a corpus with
    injected defects. See validate_empirical for why.
    """
    retentions = retentions or {}
    seen: dict[str, str] = {}

    for rule in ruleset.rules:
        existing = [c for c in rule.failure_codes
                    if c is FailureCode.F7_NON_EXECUTABLE]
        codes = list(existing)
        codes += validate_static(rule, schema)
        if FailureCode.F7_NON_EXECUTABLE not in codes:
            codes += validate_empirical(rule, schema, retentions.get(rule.rule_id),
                                        assign_vacuous=assign_vacuous)

        canon = canonical_form(rule)
        if canon in seen:
            codes.append(FailureCode.F8_REDUNDANT)
        else:
            seen[canon] = rule.rule_id

        # preserve declaration order of the taxonomy, drop duplicates
        order = list(FailureCode)
        rule.failure_codes = [c for c in order if c in codes]

    return ruleset


def inert_rate(ruleset: RuleSet, retentions: dict[str, float]) -> float:
    """Fraction of executable rules that reject nothing on this corpus.

    Reported descriptively in place of F5 when scoring natural data. A high
    inert rate is a real and quotable finding — most machine-authored rules
    catch nothing on real corpora — but it is a property of the data as much as
    of the rules, so it is not a failure code.
    """
    ex = [r for r in ruleset.executable if r.rule_id in retentions]
    if not ex:
        return 0.0
    return sum(1 for r in ex if retentions[r.rule_id] >= VACUOUS_THRESHOLD) / len(ex)


# --------------------------------------------------------------------------
# Canonicalisation and set similarity
# --------------------------------------------------------------------------

def _normalise_value(v: Any) -> Any:
    """Collapse numerically-equal values to one representation.

    Models emit `1` for an int column and `1.0` for the same column when nulls
    force it to float64. Without this, canonical_form would call those
    different rules and the stability analysis in RQ3 would report threshold
    churn that is a dtype artefact rather than model behaviour.
    """
    if isinstance(v, bool):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return v


def canonical_form(rule: Rule) -> str:
    """Identity of a rule ignoring rule_id, rationale, and parameter ordering.

    Two rules from different seeds are 'the same rule' iff they gate the same
    column with the same predicate and the same parameters. Thresholds are part
    of identity: a wobbling bound IS a different rule, which is the point of RQ3.
    Numerically-equal values are normalised first, so `1` and `1.0` do not count
    as a threshold change.
    """
    params = {k: _normalise_value(rule.parameters[k]) for k in sorted(rule.parameters)}
    if rule.predicate_type is PredicateType.IN_SET and isinstance(
        params.get("allowed"), list
    ):
        params["allowed"] = sorted(
            (_normalise_value(v) for v in params["allowed"]),
            key=lambda v: (str(type(v)), str(v)),
        )
    return json.dumps(
        {
            "column": rule.column,
            "predicate_type": rule.predicate_type.value,
            "parameters": params,
        },
        sort_keys=True,
    )


def ruleset_jaccard(a: RuleSet, b: RuleSet, executable_only: bool = True) -> float:
    """Jaccard similarity of two rule sets. Two empty sets are identical (1.0)."""
    ra = a.executable if executable_only else a.rules
    rb = b.executable if executable_only else b.rules
    sa = {canonical_form(r) for r in ra}
    sb = {canonical_form(r) for r in rb}
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


# --------------------------------------------------------------------------
# Compilation targets
# --------------------------------------------------------------------------

_EPOCH_SCALES = (
    # (unit, plausible lower bound, plausible upper bound) for years ~1990-2100
    ("s",  6.0e8,  4.2e9),
    ("ms", 6.0e11, 4.2e12),
    ("us", 6.0e14, 4.2e15),
    ("ns", 6.0e17, 4.2e18),
)


def _epoch_unit(value: Any) -> str | None:
    """Guess the epoch unit of a bare integer bound from its magnitude.

    Models express date constraints as epoch integers without stating units —
    seconds for one rule, milliseconds for the next. The IR never specified a
    unit, so a bound of 1_325_376_000 (seconds, end of 2011) was being compared
    against a datetime column coerced to NANOSECONDS (~1.3e18) and rejected
    every row. That turned semantically correct rules into total data loss and
    accounted for a large share of observed zero-retention rule sets.

    Returning None means the magnitude matches no plausible epoch scale, in
    which case the rule is genuinely nonsensical for a date column and is
    scored as written.
    """
    if isinstance(v := value, bool) or not isinstance(v, (int, float)):
        return None
    a = abs(float(v))
    for unit, lo, hi in _EPOCH_SCALES:
        if lo <= a <= hi:
            return unit
    return None


def to_pandas_mask(rule: Rule) -> Callable[[Any], Any]:
    """Return fn(DataFrame) -> boolean Series, True where the record PASSES."""
    col, t, p = rule.column, rule.predicate_type, rule.parameters

    def mask(df):
        import pandas as pd  # local import keeps module importable without pandas
        if col not in df.columns:
            return pd.Series(True, index=df.index)
        s = df[col]
        if t is PredicateType.NOT_NULL:
            return s.notna()
        if t is PredicateType.IN_SET:
            allowed = p.get("allowed", [])
            direct = s.isin(allowed)
            # A float64 column (often float only because nulls forced it) will
            # not match an integer allowed-set under isin. Compare numerically
            # as well, or a correct rule would read as rejecting everything.
            nums = [v for v in allowed if isinstance(v, (int, float))
                    and not isinstance(v, bool)]
            if nums and not pd.api.types.is_bool_dtype(s):
                coerced = pd.to_numeric(s, errors="coerce")
                if coerced.notna().any():
                    direct = direct | coerced.isin([float(v) for v in nums])
            return direct.fillna(False)
        if t is PredicateType.RANGE:
            if pd.api.types.is_datetime64_any_dtype(s):
                # Compare a datetime column against epoch bounds in the unit
                # the bound was plainly written in, rather than silently
                # coercing the column to nanoseconds.
                out = pd.Series(True, index=df.index)
                for key, op in (("min", "ge"), ("max", "le")):
                    bound = p.get(key)
                    if bound is None:
                        continue
                    unit = _epoch_unit(bound)
                    if unit is None:
                        out &= False
                        continue
                    try:
                        ts = pd.to_datetime(bound, unit=unit, utc=True)
                    except (ValueError, OverflowError):
                        out &= False
                        continue
                    series = s
                    if getattr(series.dtype, "tz", None) is None:
                        series = series.dt.tz_localize("UTC")
                    out &= getattr(series, op)(ts)
                return out.fillna(False)

            num = pd.to_numeric(s, errors="coerce")
            out = pd.Series(True, index=df.index)
            if p.get("min") is not None:
                out &= num >= p["min"]
            if p.get("max") is not None:
                out &= num <= p["max"]
            return out.fillna(False)
        if t is PredicateType.REGEX:
            return s.astype(str).str.fullmatch(p.get("pattern", ".*")).fillna(False)
        if t is PredicateType.UNIQUE:
            return ~s.duplicated(keep=False)
        if t is PredicateType.CROSS_COLUMN:
            other, op = p.get("other_column"), p.get("op")
            if other not in df.columns:
                return pd.Series(True, index=df.index)
            ops = {
                "<": s.lt, "<=": s.le, ">": s.gt,
                ">=": s.ge, "==": s.eq, "!=": s.ne,
            }
            return ops[op](df[other]) if op in ops else pd.Series(True, index=df.index)
        if t is PredicateType.TYPE:
            expected = p.get("expected", "")
            if expected in {"int", "float", "numeric"}:
                return pd.to_numeric(s, errors="coerce").notna()
            return pd.Series(True, index=df.index)
        return pd.Series(True, index=df.index)

    return mask


def _lit(v: Any) -> str:
    if v is None:
        return "NULL"
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    if isinstance(v, (int, float)):
        return str(v)
    return "'" + str(v).replace("'", "''") + "'"


def to_sql_check(rule: Rule) -> str:
    """SQL boolean expression, true where the record PASSES.

    Suitable for a Delta Lake CHECK constraint or a Spark filter.
    """
    c, t, p = f"`{rule.column}`", rule.predicate_type, rule.parameters
    if t is PredicateType.NOT_NULL:
        return f"{c} IS NOT NULL"
    if t is PredicateType.IN_SET:
        vals = ", ".join(_lit(v) for v in p.get("allowed", []))
        return f"{c} IN ({vals})"
    if t is PredicateType.RANGE:
        parts = []
        if p.get("min") is not None:
            parts.append(f"{c} >= {_lit(p['min'])}")
        if p.get("max") is not None:
            parts.append(f"{c} <= {_lit(p['max'])}")
        return " AND ".join(parts) if parts else "TRUE"
    if t is PredicateType.REGEX:
        return f"{c} RLIKE {_lit(p.get('pattern', '.*'))}"
    if t is PredicateType.CROSS_COLUMN:
        return f"{c} {p.get('op')} `{p.get('other_column')}`"
    if t is PredicateType.TYPE:
        if p.get("expected") in {"int", "float", "numeric"}:
            return f"CAST({c} AS DOUBLE) IS NOT NULL"
        return "TRUE"
    if t is PredicateType.UNIQUE:
        return "TRUE"  # set-level; enforced by an aggregate, not a row predicate
    return "TRUE"


def to_spark_expr(rule: Rule):
    """Spark Column expression, true where the record PASSES."""
    from pyspark.sql import functions as F  # local import; Spark optional
    if rule.predicate_type is PredicateType.UNIQUE:
        return F.lit(True)  # handled by ztlf enforcement as a windowed check
    return F.expr(to_sql_check(rule))


def to_great_expectations(rule: Rule) -> dict[str, Any]:
    """Great Expectations v3 expectation config."""
    col, t, p = rule.column, rule.predicate_type, rule.parameters
    kw: dict[str, Any] = {"column": col}
    if t is PredicateType.NOT_NULL:
        etype = "expect_column_values_to_not_be_null"
    elif t is PredicateType.IN_SET:
        etype = "expect_column_values_to_be_in_set"
        kw["value_set"] = p.get("allowed", [])
    elif t is PredicateType.RANGE:
        etype = "expect_column_values_to_be_between"
        kw["min_value"] = p.get("min")
        kw["max_value"] = p.get("max")
    elif t is PredicateType.REGEX:
        etype = "expect_column_values_to_match_regex"
        kw["regex"] = p.get("pattern", ".*")
    elif t is PredicateType.UNIQUE:
        etype = "expect_column_values_to_be_unique"
    elif t is PredicateType.CROSS_COLUMN:
        etype = "expect_column_pair_values_a_to_be_greater_than_b"
        kw = {"column_A": col, "column_B": p.get("other_column")}
    else:
        etype = "expect_column_values_to_not_be_null"
    return {
        "expectation_type": etype,
        "kwargs": kw,
        "meta": {"rule_id": rule.rule_id, "dimension": rule.dimension.value},
    }
