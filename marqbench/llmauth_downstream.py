"""
llmauth_downstream.py — Downstream impact and fairness for MARQ-Bench (Paper 2).

Answers RQ5: does gating training data with machine-authored rules change
downstream model discrimination or subgroup fairness, and at what authoring
cost?

DESIGN
------
Gating is a training-data curation decision, so the gate is applied to the
TRAINING partition only. Every configuration is then evaluated on the SAME
ungated test partition. Applying the gate to test data as well would change the
evaluation population per condition and make the comparison meaningless.

    evaluation split (80% from N0)
        |-- train 70%  <- gate applied here
        |-- test  30%  <- never gated, identical for every condition

LEAKAGE EXCLUSIONS ARE NOT OPTIONAL
-----------------------------------
Each corpus has features that encode the label. They are excluded by name here
rather than left to judgement, and the exclusions are reported in the paper:

  bank_marketing   `duration` — call length is known only after the call has
                   happened, and the UCI documentation states it should be
                   discarded for realistic predictive modelling.
  nyc_tlc_yellow   `total_amount` includes the tip; `payment_type` determines
                   whether a tip is recorded at all (cash tips are not).
  diabetes_130us   identifiers only.

INFEASIBLE GATES ARE A RESULT
-----------------------------
A rule set that empties the training partition, or removes an entire class,
makes model fitting impossible. That is recorded as `feasible=False` with a
reason, never silently skipped — roughly a sixth of machine-authored rule sets
fall into this category and excluding them would flatter the remainder.

Author: Ramesh Babu Kallam
License: MIT
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "DOWNSTREAM_VERSION", "TASKS", "TaskSpec", "DownstreamResult",
    "prepare_features", "train_test_split_eval", "fit_and_evaluate",
    "subgroup_report", "cost_per_retained_record",
]

DOWNSTREAM_VERSION = "1.0.0"
TEST_FRACTION = 0.30
SPLIT_SEED = 20260807
MIN_TRAIN_ROWS = 500
MIN_CLASS_ROWS = 25


@dataclass
class TaskSpec:
    """Downstream supervised task for one corpus."""
    corpus_id: str
    target_column: str
    positive: Any                       # value or callable(series) -> bool mask
    drop_columns: list[str] = field(default_factory=list)
    subgroups: list[str] = field(default_factory=list)
    max_rows: int | None = None         # subsample very large corpora
    note: str = ""


TASKS: dict[str, TaskSpec] = {
    "bank_marketing": TaskSpec(
        corpus_id="bank_marketing",
        target_column="y",
        positive="yes",
        drop_columns=["y", "duration"],
        subgroups=["age_band", "job", "marital"],
        note="`duration` dropped: known only after the call, documented by UCI "
             "as unsuitable for realistic prediction.",
    ),
    "diabetes_130us": TaskSpec(
        corpus_id="diabetes_130us",
        target_column="readmitted",
        positive="<30",
        drop_columns=["readmitted", "encounter_id", "patient_nbr"],
        subgroups=["race", "gender", "age"],
        note="Binarised to readmission within 30 days versus not.",
    ),
    "nyc_tlc_yellow": TaskSpec(
        corpus_id="nyc_tlc_yellow",
        target_column="tip_amount",
        positive=lambda s: pd.to_numeric(s, errors="coerce").fillna(0) > 0,
        drop_columns=["tip_amount", "total_amount", "payment_type"],
        subgroups=["payment_type", "PULocationID_band"],
        max_rows=250_000,
        note="Label is a RECORDED tip: cash tips are not captured. "
             "`total_amount` includes the tip and `payment_type` determines "
             "whether one is recorded, so both are excluded as leakage.",
    ),
}


@dataclass
class DownstreamResult:
    corpus_id: str
    condition: str
    model_id: str
    seed: int
    feasible: bool
    reason: str = ""
    train_rows: int = 0
    train_rows_ungated: int = 0
    retention: float = float("nan")
    positive_rate_train: float = float("nan")
    roc_auc: float = float("nan")
    pr_auc: float = float("nan")
    brier: float = float("nan")
    subgroup_auc: dict[str, dict[str, float]] = field(default_factory=dict)
    subgroup_auc_gap: dict[str, float] = field(default_factory=dict)
    subgroup_representation_shift: dict[str, float] = field(default_factory=dict)
    downstream_version: str = DOWNSTREAM_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# Feature preparation
# --------------------------------------------------------------------------

def _derive_helper_columns(df: pd.DataFrame, corpus_id: str) -> pd.DataFrame:
    """Add subgroup strata that are bands over an existing column."""
    out = df
    if corpus_id == "bank_marketing" and "age" in out.columns:
        out = out.assign(age_band=pd.cut(
            pd.to_numeric(out["age"], errors="coerce"),
            bins=[0, 30, 40, 50, 60, 200],
            labels=["<30", "30-39", "40-49", "50-59", "60+"]).astype(str))
    if corpus_id == "nyc_tlc_yellow" and "PULocationID" in out.columns:
        out = out.assign(PULocationID_band=pd.qcut(
            pd.to_numeric(out["PULocationID"], errors="coerce"),
            q=5, duplicates="drop").astype(str))
    return out


def prepare_features(
    evaluation: pd.DataFrame,
    task: TaskSpec,
    *,
    seed: int = SPLIT_SEED,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return (X_encoded, y, strata) for one corpus.

    Encoding is fitted on the FULL evaluation split before any gating, so that
    every condition sees an identical feature space. Encoding uses no label
    information, so this does not leak.
    """
    df = evaluation
    if task.max_rows is not None and len(df) > task.max_rows:
        df = df.sample(n=task.max_rows, random_state=seed).sort_index()
    df = _derive_helper_columns(df, task.corpus_id)

    if callable(task.positive):
        y = task.positive(df[task.target_column]).astype(int)
    else:
        y = (df[task.target_column] == task.positive).astype(int)

    strata = pd.DataFrame(index=df.index)
    for g in task.subgroups:
        if g in df.columns:
            strata[g] = df[g].astype(str)

    drop = set(task.drop_columns) | set(
        c for c in df.columns if c.endswith("_band"))
    feat = df.drop(columns=[c for c in drop if c in df.columns])

    encoded = {}
    for c in feat.columns:
        s = feat[c]
        if pd.api.types.is_numeric_dtype(s):
            encoded[c] = pd.to_numeric(s, errors="coerce")
        elif pd.api.types.is_datetime64_any_dtype(s):
            encoded[c] = s.astype("int64") // 10**9
        else:
            encoded[c] = pd.factorize(s.astype(str))[0].astype(float)
    X = pd.DataFrame(encoded, index=df.index)
    return X, y, strata


def train_test_split_eval(
    index: pd.Index, *, test_fraction: float = TEST_FRACTION,
    seed: int = SPLIT_SEED,
) -> tuple[pd.Index, pd.Index]:
    """Partition the evaluation split once. Identical across all conditions."""
    rng = np.random.default_rng(seed)
    pos = rng.permutation(len(index))
    n_test = int(round(test_fraction * len(index)))
    return index[np.sort(pos[n_test:])], index[np.sort(pos[:n_test])]


# --------------------------------------------------------------------------
# Fit and evaluate
# --------------------------------------------------------------------------

def fit_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    strata: pd.DataFrame,
    train_idx: pd.Index,
    test_idx: pd.Index,
    keep_mask: pd.Series | None,
    *,
    corpus_id: str,
    condition: str,
    model_id: str,
    seed: int,
    random_state: int = SPLIT_SEED,
) -> DownstreamResult:
    """Fit on the gated training rows, evaluate on the ungated test rows.

    `keep_mask` is the gate over the evaluation split (True = record survives).
    Pass None for the no-gate baseline.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

    res = DownstreamResult(corpus_id=corpus_id, condition=condition,
                           model_id=model_id, seed=seed, feasible=True)
    res.train_rows_ungated = len(train_idx)

    gated = train_idx
    if keep_mask is not None:
        gated = train_idx[keep_mask.reindex(train_idx).fillna(False).values]
    res.train_rows = len(gated)
    res.retention = len(gated) / max(len(train_idx), 1)

    if len(gated) < MIN_TRAIN_ROWS:
        res.feasible = False
        res.reason = f"gate left {len(gated)} training rows (< {MIN_TRAIN_ROWS})"
        return res

    y_tr = y.loc[gated]
    res.positive_rate_train = float(y_tr.mean())
    if y_tr.nunique() < 2 or min(y_tr.sum(), len(y_tr) - y_tr.sum()) < MIN_CLASS_ROWS:
        res.feasible = False
        res.reason = (f"gate left {int(y_tr.sum())} positive / "
                      f"{int(len(y_tr)-y_tr.sum())} negative training rows")
        return res

    clf = HistGradientBoostingClassifier(
        max_iter=150, learning_rate=0.1, random_state=random_state)
    clf.fit(X.loc[gated], y_tr)

    y_te = y.loc[test_idx]
    p = clf.predict_proba(X.loc[test_idx])[:, 1]
    res.roc_auc = float(roc_auc_score(y_te, p))
    res.pr_auc = float(average_precision_score(y_te, p))
    res.brier = float(brier_score_loss(y_te, p))

    # subgroup discrimination on the shared test set
    for g in strata.columns:
        vals = strata.loc[test_idx, g]
        per: dict[str, float] = {}
        for level, idx in vals.groupby(vals).groups.items():
            yy = y_te.loc[idx]
            if len(idx) < 100 or yy.nunique() < 2:
                continue
            per[str(level)] = float(roc_auc_score(yy, p[test_idx.get_indexer(idx)]))
        if len(per) >= 2:
            res.subgroup_auc[g] = per
            res.subgroup_auc_gap[g] = max(per.values()) - min(per.values())

        # how the gate reshaped the training population
        if keep_mask is not None:
            before = strata.loc[train_idx, g].value_counts(normalize=True)
            after = strata.loc[gated, g].value_counts(normalize=True)
            shift = (after.reindex(before.index).fillna(0) - before).abs().sum() / 2
            res.subgroup_representation_shift[g] = float(shift)

    return res


def subgroup_report(results: list[DownstreamResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        base = {"corpus": r.corpus_id, "condition": r.condition,
                "model": r.model_id, "seed": r.seed, "feasible": r.feasible}
        for g, gap in r.subgroup_auc_gap.items():
            rows.append({**base, "subgroup": g, "auc_gap": gap,
                         "representation_shift":
                             r.subgroup_representation_shift.get(g, float("nan"))})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Cost
# --------------------------------------------------------------------------

def cost_per_retained_record(
    cost_usd: float, retained_rows: int
) -> float:
    """USD per one million retained training records.

    Reported per million because per-record figures are unreadably small. A
    rule set that costs little to author but discards most of the corpus is
    expensive by this measure, which is the point: authoring cost alone
    understates the cost of a bad gate.
    """
    if retained_rows <= 0:
        return float("inf")
    return cost_usd / (retained_rows / 1e6)
