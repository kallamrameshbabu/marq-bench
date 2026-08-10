# MARQ-Bench

**Machine-Authored Rule Quality** — a benchmark and harness for evaluating data
quality validation rules authored by large language models.

This repository is the reproducibility artefact for:

> Ramesh Babu Kallam, *No Single Source Suffices: Documentation and Profiling Prevent
> Different Failures in Machine-Authored Data Quality Rules.* (under review)

[![DOI](https://zenodo.org/badge/1329363037.svg)](https://doi.org/10.5281/zenodo.21875068)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## What this is

Language models are increasingly asked to author the validation rules that gate
data entering a data product. Existing evaluations ask whether the generated
rules are *well-formed*. This benchmark asks what they *do*: how many valid
records they discard, whether they are stable across invocations, and what
happens to models trained on the surviving data.

The headline results from 7,609 rules generated across four corpora:

| Finding | |
|---|---|
| Documentation halves sentinel misreads | 1.125 → 0.525 per rule set |
| Profiling cuts invented categories by 77% | 4.188 → 0.963 per rule set |
| Neither source prevents both; supplying both is worse than either | — |
| Identical prompts give different rule sets | 75% of cells below 0.80 Jaccard |
| Every gate degraded downstream discrimination | ΔAUC −0.100 to −0.143 |
| One ordinary completeness rule reduced a model to chance | 0.869 → 0.493 AUC |
| Rules that reject nothing | 59.3% |

---

## Quick start

```bash
git clone https://github.com/kallamrameshbabu/marq-bench
cd marq-bench
pip install -r requirements.txt
python -m pytest tests/ -q          # 137 tests, no API key needed
```

Scoring an existing rule set requires no API access:

```python
import sys; sys.path.insert(0, "marqbench")
import llmauth_ir as IR

ruleset = IR.parse_llm_response(raw_model_output, corpus_id="bank_marketing",
                                condition="A2")
IR.code_ruleset(ruleset, schema, retentions)
for rule in ruleset.rules:
    print(rule.column, [c.value for c in rule.failure_codes])
```

> The modules use flat imports and are added to `sys.path`, matching how the
> notebooks load them in Colab. This is not yet a pip-installable package —
> converting it would require rewriting the imports and re-verifying the
> notebooks, and is left as future work rather than claimed here.

---

## Repository layout

```
marqbench/          the harness
  llmauth_census.py       corpus loading, splitting, profiling
  llmauth_docs.py         published data dictionaries (condition A3)
  llmauth_prompts.py      prompt construction, information conditions A2-A5
  llmauth_generate.py     provider-agnostic generation with crash-resume
  llmauth_ir.py           rule representation, failure coding, compilation
  llmauth_downstream.py   downstream models, subgroup fairness, cost
  llmauth_checkpoint.py   resumable step caching
notebooks/          N0-N5a, the full pipeline in execution order
tests/              137 tests
results/            scored outputs and the raw rule corpus
figures/            paper figures, vector and 600 dpi
data/               how to obtain the corpora (not redistributed)
docs/               protocol, methods, reproduction notes
make_figures.py     regenerates all figures from results/
```

---

## The pipeline

| Notebook | Does | Needs API key | Runtime |
|---|---|---|---|
| N0 | Load corpora, build authoring/evaluation splits | no | ~5 min |
| N1 | Build the census and the A2–A5 prompt payloads | no | ~10 min |
| N2 | Generate rules | **yes** | ~1–2 h |
| N3 | Parse, score, failure-code, stability, correlation | no | ~20 min |
| N4 | Downstream models, subgroup fairness, cost | no | ~30 min |
| N5a | Memorization probe | **yes** | ~45 min |

Only N2 and N5a call a model. Everything else runs on the released outputs, so
**all analysis in the paper can be reproduced without an API key or any spend.**

---

## Design notes worth knowing

**Loading is not incidental.** Each corpus has an explicit read configuration
because library defaults destroy the phenomena under study. pandas' default
missing-value list contains the string `"None"`, and reading the diabetes corpus
with defaults converts 96,420 valid values into nulls. Online Retail II ships as
a two-sheet workbook; reading only the first sheet yields 525,461 of 1,067,371
rows. `load_corpus` refuses to run on an unregistered corpus rather than guess,
and asserts expected row counts.

**Authoring and evaluation are separated.** Profiles come from a 20% authoring
split; every metric is computed on the disjoint 80%. Without this the census
would describe the same rows the resulting rules are scored against.

**Authoring and scoring code cannot see each other.** The sentinel registry and
protected-attribute lists live in `llmauth_ir` and are evaluation ground truth.
`llmauth_prompts` must not import them; a test asserts it.

**Nothing is silently dropped.** Unparseable responses become null rule sets
that stay in every denominator. Malformed rules are retained and coded. Failed
generations are recorded with a reason, never replaced by a fresh seed.

**Everything is resumable.** Free-tier notebooks disconnect. Each expensive step
checkpoints to disk; re-running skips completed work. To redo a step, delete its
file and re-run.

---

## Known issues in dependencies

`tabmemcheck` 0.1.6 leaks `config.max_tokens` when an exception occurs inside
`first_token_test`, silently truncating every subsequent generation to one token
and producing well-formatted zeros. See `docs/tabmemcheck_bug_report.md` for a
tested reproduction and patch. The N5a notebook resets the cap before and after
every test, so results here are unaffected.

---

## Citing

If you use this benchmark, please cite both the paper and the artefact — see
[CITATION.cff](CITATION.cff).

## Related work

This benchmark builds on the evaluation methodology of:

> Kallam, R. B. *Census-Informed Data Quality Governance for Lakehouse Data
> Products: Rule Authorship, Evaluation Bias, and Downstream Cost.*
> Artefact: https://doi.org/10.5281/zenodo.21813665 ·
> Code: https://github.com/kallamrameshbabu/census-informed-data-quality-governance

That work established that rule authorship dominates rule enforcement. This one
asks what happens when the author is a language model.

## License

MIT for the code. The corpora are not redistributed; see `data/README.md` for
their sources and licences.
