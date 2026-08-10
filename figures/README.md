# Figures

Vector PDF (for typesetting) and 600 dpi PNG. Regenerate with:

```bash
python make_figures.py --n3 results/N3_scored_rulesets.csv \
                       --n4 results/N4_downstream.csv --out figures/
```

| Figure | Shows |
|---|---|
| `fig1_complementarity` | Invented categories and sentinel misreads move in opposite directions across information conditions. The paper's central result. |
| `fig2_stability` | Across-seed Jaccard similarity; 75% of cells below 0.80. |
| `fig3_downstream` | ΔAUC vs no gate, pooled and per corpus. All CIs below zero. |
| `fig4_causal_chain` | The C4 mechanism, from block-structured missingness to a model at chance. |

Colours are Okabe-Ito, checked for deuteranopia. No red/green pairing carries
meaning.
