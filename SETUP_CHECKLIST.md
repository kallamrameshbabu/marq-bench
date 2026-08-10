# Setup checklist

All metadata is filled in from the Paper 1 manuscript. **One item remains, and
it cannot be filled until after the first Zenodo release.**

## Outstanding

| Where | Field | Why it waits |
|---|---|---|
| `README.md` line 11 | DOI badge | Zenodo mints the DOI when you create the GitHub release |

Sequence: push → create release → copy the **concept DOI** into `README.md` →
commit → tag `v1.0.1`.

## Filled in, verified against Paper 1

| Field | Value |
|---|---|
| Author | Ramesh Babu Kallam |
| ORCID | 0009-0008-5220-1775 |
| Email | kallamrameshbabu@gmail.com |
| Affiliation | Independent Researcher, Cloud Data Engineering, West Chester, Ohio 45069, United States |
| Repository | https://github.com/kallamrameshbabu/marq-bench |
| Linked prior artefact | 10.5281/zenodo.21813665 |

The affiliation string is **character-identical to Paper 1**. Keep it that way
across both papers, both artefacts, and your ORCID record — inconsistent
affiliation strings fragment author indexing in Scopus and Google Scholar, which
is the opposite of what a body of work needs.

`.zenodo.json` declares this artefact `isSupplementTo` the Paper 1 artefact, so
Zenodo and DataCite will show the two as related rather than as isolated
deposits.

## Creating the repository

```bash
# on github.com: New repository -> marq-bench -> Public -> no README
tar -xzf marq-bench-scaffold.tar.gz && cd marq-bench

# add results and cleaned notebooks BEFORE the first commit
cp /path/to/Drive/artifacts/*      results/
cp /path/to/Drive/runs/runs.jsonl  results/
python clean_notebooks.py --in /path/to/Drive/Notebooks --out notebooks/ --dry-run
python clean_notebooks.py --in /path/to/Drive/Notebooks --out notebooks/

# verify
git grep -nE "sk-[A-Za-z0-9]{20,}"     # must return nothing
python -m pytest tests/ -q

git init && git add -A
git commit -m "MARQ-Bench v1.0.0: benchmark, harness, and rule corpus"
git branch -M main
git remote add origin https://github.com/kallamrameshbabu/marq-bench.git
git push -u origin main
```

Then `docs/RELEASING.md` for Zenodo. **Enable the repo in Zenodo before creating
the release** — Zenodo only captures releases made after the switch is on.
