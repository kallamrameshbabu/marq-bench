# Results

The outputs the paper reports. Every statistic in the manuscript is computed
from files in this directory.

| File | What it is |
|---|---|
| `runs.jsonl` | **The rule corpus.** One line per generation: complete raw model response, prompt hash, model version, parameters as actually transmitted, tokens, cost, timestamps. 320 runs. |
| `N3_scored_rulesets.csv` | One row per rule set: retention (as written and excluding type mismatches), rule counts, F1–F10 counts, inert rate, independence ratio. |
| `N3_stability.csv` | Across-seed Jaccard similarity per corpus × model × condition. |
| `N4_downstream.csv` | Feasible gates with ROC-AUC, PR-AUC, Brier, delta vs no-gate, cost per retained record. |
| `N4_subgroups.csv` | Subgroup AUC gaps and representation shift. |
| `N4_all_gates.csv` | All gates including infeasible ones, with reasons. |
| `N5a_memorization.json` | Memorization probe results per corpus × model. |
| `*_provenance.json` | Per-notebook provenance: versions, seeds, file hashes, configuration. |

## Notes on interpretation

**`runs.jsonl` holds raw responses, not parsed rules.** Parsing happens in N3.
This is deliberate: when a parser bug surfaces, the raw corpus is what saves you
from regenerating everything. It is also what makes the analysis reproducible
after the models are withdrawn.

**Failed runs are present**, marked `ok: false` with a reason. They are part of
every denominator. Removing them would bias every rate statistic upward.

**`request_params` records `requested` and `applied` separately.** Neither model
exposes a seed, and one rejects `temperature`, so what was asked for and what was
sent differ. The log records what was sent.

**Infeasible gates are in `N4_all_gates.csv`, not `N4_downstream.csv`.** A gate
that empties the training partition cannot produce a model. Analysing only
`N4_downstream.csv` excludes the most destructive gates; see the paper's
survivorship analysis.
