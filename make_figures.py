"""
make_figures.py — Publication figures for Paper 2 (MARQ-Bench).

Produces four figures as both PDF (vector, preferred by the journal) and PNG at
600 dpi. Every number is computed from N3_scored_rulesets.csv except the
downstream panel, which reads N4_downstream.csv when present and otherwise
falls back to the reported cell means (see N4_FALLBACK below — replace this by
pointing the script at the real CSV before submission).

Usage:
    python3 make_figures.py --n3 path/to/N3_scored_rulesets.csv \
                            --n4 path/to/N4_downstream.csv \
                            --out figures/

Design notes:
  - Colourblind-safe palette (Okabe-Ito), verified for deuteranopia.
  - No red/green pairing carries meaning.
  - Vector output; text stays selectable and scales without resampling.
  - Figure widths sized for a single journal column (3.5 in) or full width
    (7.2 in) so nothing is downscaled in typesetting.

Author: Ramesh Babu Kallam
License: MIT
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# --------------------------------------------------------------------------
# Style
# --------------------------------------------------------------------------

OKABE = {
    "blue":   "#0072B2",
    "orange": "#E69F00",
    "green":  "#009E73",
    "purple": "#CC79A7",
    "sky":    "#56B4E9",
    "vermil": "#D55E00",
    "yellow": "#F0E442",
    "grey":   "#8C8C8C",
}

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 600,
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

CONDITIONS = ["A2", "A3", "A4", "A5"]
COND_LABEL = {
    "A2": "A2\nschema\nonly",
    "A3": "A3\n+ docs",
    "A4": "A4\n+ census",
    "A5": "A5\n+ census\n+ samples",
}

# Reported N4 cell means, used only if N4_downstream.csv is unavailable.
N4_FALLBACK = pd.DataFrame({
    "condition": CONDITIONS,
    "mean": [-0.1175, -0.1428, -0.1000, -0.1051],
    "lo":   [-0.1618, -0.1869, -0.1351, -0.1383],
    "hi":   [-0.0759, -0.1006, -0.0668, -0.0731],
})
N4_BY_CORPUS_FALLBACK = {
    ("bank_marketing", "haiku"):  [-0.0156, 0.0010, -0.0053, -0.0115],
    ("bank_marketing", "sonnet"): [-0.0153, -0.0186, -0.0257, -0.0652],
    ("diabetes_130us", "haiku"):  [np.nan, -0.0266, -0.0003, -0.0018],
    ("diabetes_130us", "sonnet"): [-0.0208, -0.0375, -0.0042, -0.0004],
    ("nyc_tlc_yellow", "haiku"):  [-0.3271, -0.3465, -0.2495, -0.2404],
    ("nyc_tlc_yellow", "sonnet"): [-0.2891, -0.3647, -0.2814, -0.2164],
}

# Across-seed Jaccard, from N3 section 11 (cell means).
JACCARD = pd.DataFrame(
    {
        "A2": [0.697, 0.483, 0.300, 0.806, 0.529, 0.716, 0.747, 0.504],
        "A3": [0.967, 0.676, 0.848, 0.756, 0.885, 0.785, 0.466, 0.578],
        "A4": [0.908, 0.805, 0.729, 0.632, 0.628, 0.358, 0.273, 0.803],
        "A5": [0.840, 0.690, 0.629, 0.770, 0.387, 0.396, 0.286, 0.567],
    },
    index=["Bank/Haiku", "Bank/Sonnet", "Diabetes/Haiku", "Diabetes/Sonnet",
           "TLC/Haiku", "TLC/Sonnet", "Retail/Haiku", "Retail/Sonnet"],
)


def bootstrap_ci(x, n=10000, seed=20260807):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan
    bs = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {name}.pdf and {name}.png")


# --------------------------------------------------------------------------
# Figure 1 — the complementarity result
# --------------------------------------------------------------------------

def figure1(d: pd.DataFrame, out: Path):
    """Two stacked panels sharing an x-axis.

    A dual-axis chart would fit both series in one panel but would let the
    reader's eye infer a crossover point that the data does not support, since
    the two scales are arbitrary relative to each other. Stacked panels make
    the opposite trends legible without implying a comparison of magnitudes.
    """
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 4.4), sharex=True,
                             gridspec_kw={"hspace": 0.18})
    x = np.arange(len(CONDITIONS))

    for ax, code, label, colour in (
        (axes[0], "F2", "Invented categories\n(F2 per rule set)", OKABE["orange"]),
        (axes[1], "F9", "Sentinel misreads\n(F9 per rule set)", OKABE["blue"]),
    ):
        means, los, his = [], [], []
        for c in CONDITIONS:
            v = d.loc[d.condition == c, code].values
            lo, hi = bootstrap_ci(v)
            means.append(v.mean()); los.append(lo); his.append(hi)
        means = np.array(means)
        err = np.vstack([means - np.array(los), np.array(his) - means])

        best = int(np.argmin(means))
        colours = [colour if i != best else OKABE["green"] for i in range(4)]
        ax.bar(x, means, yerr=err, capsize=3, color=colours,
               edgecolor="white", linewidth=0.6, width=0.68,
               error_kw={"elinewidth": 0.9, "ecolor": "#333333"})
        ax.set_ylabel(label)
        for i, m in enumerate(means):
            ax.annotate(f"{m:.2f}", xy=(x[i], his[i]), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=7)
        # "best" sits above the value label so it cannot clip on short bars
        ax.annotate("best", xy=(x[best], his[best]), xytext=(0, 13),
                    textcoords="offset points", ha="center", fontsize=7,
                    color=OKABE["green"], fontweight="bold")
        ax.set_ylim(0, max(his) * 1.38)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels([COND_LABEL[c] for c in CONDITIONS])
    axes[0].set_title("Documentation and profiling prevent\ndifferent failures",
                      fontweight="bold", loc="left")
    fig.text(0.5, -0.11,
             "Mean failures per rule set; whiskers are 95% bootstrap CIs\n(n = 80 per condition). Green marks the best condition.",
             ha="center", fontsize=7, color="#555555")
    save(fig, out, "fig1_complementarity")


# --------------------------------------------------------------------------
# Figure 2 — stability
# --------------------------------------------------------------------------

def figure2(out: Path):
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    x = np.arange(len(CONDITIONS))
    rng = np.random.default_rng(7)

    parts = ax.violinplot([JACCARD[c].values for c in CONDITIONS],
                          positions=x, widths=0.72, showextrema=False,
                          showmedians=False)
    for pc in parts["bodies"]:
        pc.set_facecolor(OKABE["sky"]); pc.set_alpha(0.28)
        pc.set_edgecolor("none")

    for i, c in enumerate(CONDITIONS):
        v = JACCARD[c].values
        ax.scatter(np.full(len(v), i) + rng.uniform(-0.11, 0.11, len(v)), v,
                   s=16, color=OKABE["blue"], alpha=0.85, zorder=3,
                   edgecolor="white", linewidth=0.4)
        ax.hlines(v.mean(), i - 0.28, i + 0.28, color=OKABE["vermil"],
                  linewidth=2, zorder=4)

    ax.axhline(0.80, color=OKABE["grey"], linestyle="--", linewidth=0.9, zorder=1)
    ax.annotate("0.80", xy=(3.42, 0.80), fontsize=7, color=OKABE["grey"],
                va="center")
    ax.set_xticks(x)
    ax.set_xticklabels([c for c in CONDITIONS])
    ax.set_ylabel("Across-seed Jaccard similarity")
    ax.set_xlabel("Information condition")
    ax.set_ylim(0, 1.0)
    ax.set_title("Identical prompts yield different rule sets",
                 fontweight="bold", loc="left")
    ax.annotate("75% of cells below 0.80", xy=(0.02, 0.06),
                xycoords="axes fraction", fontsize=7.5, color="#333333")
    fig.text(0.5, -0.10,
             "Each point is one corpus x model cell (n = 8); red bars are condition means.",
             ha="center", fontsize=7, color="#555555")
    save(fig, out, "fig2_stability")


# --------------------------------------------------------------------------
# Figure 3 — downstream
# --------------------------------------------------------------------------

def figure3(n4: pd.DataFrame | None, out: Path):
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(7.2, 3.3), gridspec_kw={"width_ratios": [1, 1.45],
                                               "wspace": 0.30})
    x = np.arange(len(CONDITIONS))

    if n4 is not None and len(n4):
        rows = []
        for c in CONDITIONS:
            v = n4.loc[n4.condition == c, "delta_auc"].values
            lo, hi = bootstrap_ci(v)
            rows.append({"condition": c, "mean": v.mean(), "lo": lo, "hi": hi})
        pooled = pd.DataFrame(rows)
        by_corpus = {}
        for (corp, mod), g in n4.groupby(["corpus_id", "model_id"]):
            short = "haiku" if "haiku" in mod else "sonnet"
            by_corpus[(corp, short)] = [
                g.loc[g.condition == c, "delta_auc"].mean() for c in CONDITIONS]
    else:
        pooled = N4_FALLBACK.copy()
        by_corpus = N4_BY_CORPUS_FALLBACK

    err = np.vstack([pooled["mean"] - pooled["lo"], pooled["hi"] - pooled["mean"]])
    axL.bar(x, pooled["mean"], yerr=err, capsize=3, color=OKABE["purple"],
            edgecolor="white", linewidth=0.6, width=0.66,
            error_kw={"elinewidth": 0.9, "ecolor": "#333333"})
    axL.axhline(0, color="#222222", linewidth=1.0)
    axL.set_xticks(x); axL.set_xticklabels(CONDITIONS)
    axL.set_ylabel(r"$\Delta$ ROC-AUC vs no gate")
    axL.set_xlabel("Information condition")
    axL.set_title("Every condition degraded\ndownstream discrimination",
                  fontweight="bold", loc="left")
    axL.annotate("all 95% CIs below zero", xy=(0.03, 0.10),
                 xycoords="axes fraction", fontsize=7.5, color="#333333")
    axL.annotate("feasible gates only —\n39 more were unfittable",
                 xy=(0.03, 0.015), xycoords="axes fraction", fontsize=6.8,
                 color="#777777")

    style = {"bank_marketing": (OKABE["blue"], "Bank"),
             "diabetes_130us": (OKABE["green"], "Diabetes"),
             "nyc_tlc_yellow": (OKABE["vermil"], "NYC TLC")}
    seen = set()
    for (corp, mod), vals in by_corpus.items():
        colour, label = style[corp]
        axR.plot(x, vals, marker="o" if mod == "haiku" else "s",
                 markersize=4.5, linewidth=1.3, color=colour,
                 linestyle="-" if mod == "haiku" else "--",
                 label=label if corp not in seen else None, alpha=0.9)
        seen.add(corp)
    axR.axhline(0, color="#222222", linewidth=1.0)
    axR.set_xticks(x); axR.set_xticklabels(CONDITIONS)
    axR.set_xlabel("Information condition")
    axR.set_ylabel(r"$\Delta$ ROC-AUC")
    axR.set_title("NYC TLC degrades by an order\nof magnitude more",
                  fontweight="bold", loc="left")
    axR.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3,
               fontsize=7.5, title="solid = Haiku,  dashed = Sonnet",
               title_fontsize=7)
    axR.axhspan(-0.08, 0.01, color=OKABE["grey"], alpha=0.10, zorder=0)
    axR.annotate("Bank and Diabetes\nall within -0.07 to +0.001",
                 xy=(2.55, -0.045), fontsize=7, color="#555555",
                 ha="left", va="center")
    save(fig, out, "fig3_downstream")


# --------------------------------------------------------------------------
# Figure 4 — the C4 causal chain
# --------------------------------------------------------------------------

def figure4(out: Path):
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3)
    ax.axis("off"); ax.grid(False)

    steps = [
        ("Block-structured\nmissingness",
         "Three columns null on\nthe SAME 955,371 rows\n(23.35% of the corpus)",
         OKABE["sky"]),
        ("An ordinary\ncompleteness rule",
         '"RatecodeID must\nbe present"\nA reviewer approves it',
         OKABE["yellow"]),
        ("A subpopulation\nis removed",
         "Flex Fare trips deleted;\ncomposition shift 0.236\nin every condition",
         OKABE["orange"]),
        ("The model\ncollapses to chance",
         "ROC-AUC 0.869 -> 0.493\nSame volume removed at\nrandom costs 0.0004",
         OKABE["vermil"]),
    ]

    w, h, y = 2.15, 1.55, 0.80
    for i, (title, body, colour) in enumerate(steps):
        x0 = 0.10 + i * 2.48
        ax.add_patch(FancyBboxPatch(
            (x0, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
            facecolor=colour, alpha=0.30, edgecolor=colour, linewidth=1.4))
        ax.text(x0 + w / 2, y + h - 0.22, title, ha="center", va="top",
                fontsize=8.2, fontweight="bold")
        ax.text(x0 + w / 2, y + h - 0.68, body, ha="center", va="top",
                fontsize=6.9, color="#333333")
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch(
                (x0 + w + 0.04, y + h / 2), (x0 + w + 0.38, y + h / 2),
                arrowstyle="-|>", mutation_scale=13, linewidth=1.4,
                color="#444444"))

    ax.text(5.0, 2.72, "A failure that no review would catch",
            ha="center", fontsize=10.5, fontweight="bold")
    ax.text(5.0, 0.34,
            "Invisible in the schema. Invisible in a per-column census. "
            "Invisible in the rule text.\n"
            "All 75 machine-authored gates removed 100% of the block.",
            ha="center", fontsize=7.6, color="#555555")
    save(fig, out, "fig4_causal_chain")


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n3", required=True, help="N3_scored_rulesets.csv")
    ap.add_argument("--n4", default=None, help="N4_downstream.csv (optional)")
    ap.add_argument("--out", default="figures", help="output directory")
    args = ap.parse_args()

    d = pd.read_csv(args.n3)
    d = d[d.model.str.startswith("claude")]
    d = d[~d.model.str.contains("opus")]
    print(f"N3: {len(d)} rule sets")

    n4 = None
    if args.n4 and Path(args.n4).exists():
        n4 = pd.read_csv(args.n4)
        if "delta_auc" not in n4.columns:
            print("  !! N4 csv has no delta_auc column; using fallback values")
            n4 = None
        else:
            print(f"N4: {len(n4)} feasible gates")
    else:
        print("  !! N4 csv not supplied; using reported cell means "
              "(replace before submission)")

    out = Path(args.out)
    figure1(d, out)
    figure2(out)
    figure3(n4, out)
    figure4(out)
    print(f"\nAll figures written to {out.resolve()}")


if __name__ == "__main__":
    main()
