"""
llmauth_census.py — Corpus loading, splitting, and profiling for MARQ-Bench.

Produces the CorpusFacts objects that llmauth_prompts turns into A2-A5
payloads. Three jobs:

  1. Load each corpus WITHOUT destroying its encoded-missing conventions.
  2. Split it once into a 20% authoring split and an 80% evaluation split,
     reproducibly, with an archived manifest.
  3. Profile the authoring split only.

ANTI-TAUTOLOGY BOUNDARY
-----------------------
This module is on the AUTHORING side of the wall. It must not import
llmauth_ir or any evaluation module. It may import llmauth_prompts.

THE 'None' BUG — READ THIS
--------------------------
pandas' default NA token list includes the string "None". Reading
diabetic_data.csv with a plain read_csv silently converts 96,420 occurrences
of the literal value "None" in `max_glu_serum` into NaN.

That single default would destroy the study's clearest finding. "None" in that
column is a VALID value meaning the lab test was not administered; it is not
missing data. Converted to NaN, a `not_null` rule on that column stops being a
sentinel misread and becomes correct behaviour, and the whole three-way
discrimination result evaporates.

Every corpus therefore has an explicit read configuration below, and
`load_corpus` refuses to run without one. Never call pd.read_csv directly on
these files.

Author: Ramesh Babu Kallam
License: MIT
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from llmauth_prompts import ColumnFacts, CorpusFacts

__all__ = [
    "CORPUS_READ_CONFIG", "CENSUS_VERSION", "TOP_K",
    "SplitManifest", "load_corpus", "make_split",
    "profile_authoring_split", "save_facts", "load_facts",
]

CENSUS_VERSION = "1.0.0"
TOP_K = 20                 # values listed per column in the census
AUTHORING_FRACTION = 0.20
SPLIT_SEED = 20260807
QUANTILES = (0.01, 0.25, 0.50, 0.75, 0.99)


# --------------------------------------------------------------------------
# Per-corpus read configuration — explicit, never inferred
# --------------------------------------------------------------------------

CORPUS_READ_CONFIG: dict[str, dict[str, Any]] = {
    "bank_marketing": {
        "format": "csv",
        "read_kwargs": {
            "sep": ";", "quotechar": '"',
            "keep_default_na": False, "na_values": [],
        },
        "expected_rows": 45_211,
        "note": "Semicolon-delimited, all fields quoted. No genuine nulls. "
                "'unknown' in job/education/contact/poutcome is a real level; "
                "pdays = -1 is a real value. Nothing is converted to NaN.",
    },
    "diabetes_130us": {
        "format": "csv",
        "read_kwargs": {"keep_default_na": False, "na_values": []},
        "expected_rows": 101_766,
        "note": "CRITICAL: keep_default_na=False. '?' marks missing in "
                "weight/payer_code/medical_specialty/race/diag_*; 'None' in "
                "max_glu_serum and A1Cresult means the test was not "
                "administered and is a VALID value. Default pandas would "
                "convert 96,420 'None' values to NaN and destroy the finding.",
    },
    "online_retail_ii": {
        "format": "excel",
        "read_kwargs": {"sheet_name": None},   # None => read EVERY sheet
        "expected_rows": 1_067_371,
        "note": "CRITICAL: UCI distributes this as online_retail_II.xlsx with "
                "TWO sheets — 'Year 2009-2010' (525,461 rows) and "
                "'Year 2010-2011' (541,910 rows). pd.read_excel(path) with no "
                "sheet_name reads only the FIRST sheet and silently returns "
                "half the corpus with no error. sheet_name=None reads all "
                "sheets; they are concatenated in workbook order. Blank "
                "Customer ID and Description are genuinely absent, so '' maps "
                "to NaN; no other token does.",
    },
    "nyc_tlc_yellow": {
        "format": "parquet",
        "read_kwargs": {},
        "note": "Parquet carries real nulls natively; no token translation. "
                "RatecodeID has BOTH nulls (23.35%) and the encoded value 99 "
                "(3.44%) and both must survive loading intact.",
    },
}


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_corpus(corpus_id: str, path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a corpus using its registered read configuration.

    Returns (dataframe, provenance). Raises for an unregistered corpus rather
    than falling back to pandas defaults, because the defaults are unsafe here.
    """
    if corpus_id not in CORPUS_READ_CONFIG:
        raise KeyError(
            f"no read configuration for {corpus_id!r}. Add one to "
            "CORPUS_READ_CONFIG stating exactly which tokens mean missing. "
            "Do not fall back to pandas defaults."
        )
    cfg = CORPUS_READ_CONFIG[corpus_id]
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if cfg["format"] == "csv":
        df = pd.read_csv(path, **cfg["read_kwargs"])
        sheet_rows = None
    elif cfg["format"] == "parquet":
        df = pd.read_parquet(path, **cfg["read_kwargs"])
        sheet_rows = None
    elif cfg["format"] == "excel":
        # sheet_name=None returns {sheet_name: DataFrame} for EVERY sheet.
        # Concatenating them is the whole point: reading only the first sheet
        # is a silent, error-free way to lose half a corpus.
        sheets = pd.read_excel(path, **cfg["read_kwargs"])
        if isinstance(sheets, pd.DataFrame):        # single-sheet workbook
            df, sheet_rows = sheets, {"<only>": len(sheets)}
        else:
            sheet_rows = {name: len(s) for name, s in sheets.items()}
            df = pd.concat(sheets.values(), ignore_index=True)
    else:
        raise ValueError(f"unsupported format {cfg['format']!r}")

    expected = cfg.get("expected_rows")
    if expected is not None and len(df) != expected:
        raise ValueError(
            f"{corpus_id}: loaded {len(df):,} rows but expected {expected:,}. "
            f"Sheets/parts read: {sheet_rows}. This guard exists because "
            "partial loads produce valid-looking results — investigate before "
            "proceeding, do not relax the expectation to match."
        )

    provenance = {
        "corpus_id": corpus_id,
        "path": str(path),
        "file_sha256": _sha256_file(path),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "format": cfg["format"],
        "read_kwargs": cfg["read_kwargs"],
        "sheet_rows": sheet_rows,
        "census_version": CENSUS_VERSION,
    }
    return df, provenance


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------

@dataclass
class SplitManifest:
    """Reproducible record of how a corpus was subsampled and split."""
    corpus_id: str
    source_rows: int
    subsample_rows: int | None
    subsample_seed: int | None
    authoring_rows: int
    evaluation_rows: int
    authoring_fraction: float
    split_seed: int
    authoring_index_sha256: str
    evaluation_index_sha256: str
    census_version: str = CENSUS_VERSION

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def _index_hash(idx: Sequence[int]) -> str:
    return hashlib.sha256(
        np.asarray(sorted(idx), dtype=np.int64).tobytes()
    ).hexdigest()


def make_split(
    df: pd.DataFrame,
    corpus_id: str,
    *,
    authoring_fraction: float = AUTHORING_FRACTION,
    seed: int = SPLIT_SEED,
    subsample_rows: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, SplitManifest]:
    """Split once into (authoring, evaluation) with an archivable manifest.

    Rules are authored from a profile of the authoring split; every metric is
    computed on the disjoint evaluation split. Without this separation the
    census describes the same rows the rules are later scored on, which is
    leakage a reviewer will find.

    `subsample_rows` is for C4, which is downsampled from 4.09M to ~1M before
    splitting. The subsample seed is recorded so the sample is reproducible.
    """
    source_rows = len(df)
    sub_seed = None
    if subsample_rows is not None and subsample_rows < source_rows:
        sub_seed = seed
        df = df.sample(n=subsample_rows, random_state=sub_seed).sort_index()

    rng = np.random.default_rng(seed)
    positions = rng.permutation(len(df))
    n_auth = int(round(authoring_fraction * len(df)))
    auth_pos, eval_pos = positions[:n_auth], positions[n_auth:]

    authoring = df.iloc[np.sort(auth_pos)].copy()
    evaluation = df.iloc[np.sort(eval_pos)].copy()

    manifest = SplitManifest(
        corpus_id=corpus_id,
        source_rows=source_rows,
        subsample_rows=subsample_rows,
        subsample_seed=sub_seed,
        authoring_rows=len(authoring),
        evaluation_rows=len(evaluation),
        authoring_fraction=authoring_fraction,
        split_seed=seed,
        authoring_index_sha256=_index_hash(authoring.index.tolist()),
        evaluation_index_sha256=_index_hash(evaluation.index.tolist()),
    )
    return authoring, evaluation, manifest


# --------------------------------------------------------------------------
# Profiling
# --------------------------------------------------------------------------

def _infer_dtype(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):
        return "bool"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "timestamp"
    if pd.api.types.is_integer_dtype(s):
        return "int"
    if pd.api.types.is_float_dtype(s):
        return "float"
    # object column that is numeric in substance
    num = pd.to_numeric(s, errors="coerce")
    non_null = s.notna().sum()
    if non_null and num.notna().sum() >= 0.95 * non_null:
        return "float" if (num.dropna() % 1 != 0).any() else "int"
    return "string"


def profile_authoring_split(
    authoring: pd.DataFrame,
    corpus_id: str,
    *,
    docs: dict[str, str] | None = None,
    table_doc: str = "",
    top_k: int = TOP_K,
) -> CorpusFacts:
    """Profile the authoring split into CorpusFacts.

    NO CARDINALITY GATE on the value-frequency list. Every column gets its
    top-k most frequent values, numeric columns included.

    This is deliberate and load-bearing. A conventional profiler emits a level
    list only for low-cardinality columns. `pdays` has 559 distinct values, so
    such a profiler would report only "range -1 to 871, p50 = -1" and the fact
    that 81.74% of values are exactly -1 would never reach the prompt. That
    single omission would hollow out condition A4 and bias the study against
    its own primary hypothesis.
    """
    docs = docs or {}
    columns: list[ColumnFacts] = []

    for name in authoring.columns:
        s = authoring[name]
        dtype = _infer_dtype(s)

        min_v = max_v = None
        quantiles: dict[str, Any] = {}
        if dtype in ("int", "float"):
            num = pd.to_numeric(s, errors="coerce")
            if num.notna().any():
                min_v = float(num.min())
                max_v = float(num.max())
                quantiles = {
                    f"p{int(q * 100):02d}": float(num.quantile(q))
                    for q in QUANTILES
                }

        vc = s.value_counts(dropna=True).head(top_k)
        top_levels = [
            (v.item() if hasattr(v, "item") else v, int(n))
            for v, n in vc.items()
        ]

        columns.append(ColumnFacts(
            name=str(name),
            dtype=dtype,
            doc=docs.get(str(name), ""),
            null_rate=float(s.isna().mean()),
            distinct_count=int(s.nunique(dropna=True)),
            min_value=min_v,
            max_value=max_v,
            quantiles=quantiles,
            top_levels=top_levels,
        ))

    return CorpusFacts(
        corpus_id=corpus_id,
        row_count=len(authoring),
        columns=columns,
        table_doc=table_doc,
    )


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------

def save_facts(facts: CorpusFacts, path: str | Path) -> str:
    """Serialize CorpusFacts to JSON. Returns the content SHA-256."""
    payload = {
        "census_version": CENSUS_VERSION,
        "corpus_id": facts.corpus_id,
        "row_count": facts.row_count,
        "table_doc": facts.table_doc,
        "columns": [
            {
                "name": c.name, "dtype": c.dtype, "doc": c.doc,
                "null_rate": c.null_rate, "distinct_count": c.distinct_count,
                "min_value": c.min_value, "max_value": c.max_value,
                "quantiles": c.quantiles,
                "top_levels": [[v, n] for v, n in c.top_levels],
            }
            for c in facts.columns
        ],
    }
    text = json.dumps(payload, indent=2, sort_keys=True, default=str)
    Path(path).write_text(text)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_facts(path: str | Path) -> CorpusFacts:
    payload = json.loads(Path(path).read_text())
    return CorpusFacts(
        corpus_id=payload["corpus_id"],
        row_count=payload["row_count"],
        table_doc=payload.get("table_doc", ""),
        columns=[
            ColumnFacts(
                name=c["name"], dtype=c["dtype"], doc=c.get("doc", ""),
                null_rate=c.get("null_rate"),
                distinct_count=c.get("distinct_count"),
                min_value=c.get("min_value"), max_value=c.get("max_value"),
                quantiles=c.get("quantiles", {}),
                top_levels=[(v, n) for v, n in c.get("top_levels", [])],
            )
            for c in payload["columns"]
        ],
    )
