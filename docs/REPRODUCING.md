# Reproducing the results

Three levels, depending on how much you want to re-run.

## 1. Verify the analysis — no API key, no cost, ~5 minutes

Everything reported in the paper is computed from files in `results/`.

```bash
pip install -r requirements.txt
python -m pytest tests/ -q                     # 137 tests
python make_figures.py --n3 results/N3_scored_rulesets.csv \
                       --n4 results/N4_downstream.csv \
                       --out figures/
```

The figures regenerate byte-comparably. Every statistic in the paper can be
recomputed from `N3_scored_rulesets.csv` and `N4_downstream.csv`.

## 2. Re-score the released rule corpus — no API key, ~1 hour

Obtain the corpora (`data/README.md`), then run N0, N1, N3, N4. These read the
released `results/runs.jsonl` rather than calling any model, so parsing, failure
coding, retention, stability, downstream, and fairness are all reproducible
without spend.

## 3. Regenerate rules — API key required, ~$11

Run N2. This calls the models and will **not** reproduce our outputs exactly:
neither model exposes a seed parameter, so generation is non-deterministic. That
non-reproducibility is itself one of the paper's findings (§4.5), not a defect
in this artefact.

Expect a different rule corpus with the same qualitative structure. If the
information-condition effects do not replicate in direction, we would want to
know.

## Environment we used

Google Colab, Python 3.12, pandas 2.x, scikit-learn 1.x. Model version strings
and per-call timestamps are recorded in `results/runs.jsonl`. Those models may
be withdrawn or updated; the raw responses are released so the analysis remains
reproducible regardless.

## If something disagrees

Open an issue with the disagreeing number and how you obtained it. Results in
`results/` are the ones the paper reports; if the code produces something else
from the same inputs, that is a bug and we want to hear about it.
