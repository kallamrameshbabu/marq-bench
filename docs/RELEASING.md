# Release checklist

## Before the first release

- [ ] Replace every `PLACEHOLDER` in `README.md`, `CITATION.cff`, `.zenodo.json`
- [ ] Add your ORCID (register free at orcid.org if you have not)
- [ ] Run `clean_notebooks.py` and **read each cleaned notebook end to end**
- [ ] Confirm no API keys anywhere: `git grep -nE "sk-[A-Za-z0-9]{20,}"`
- [ ] Confirm no corpora committed: `git status --short data/`
- [ ] `python -m pytest tests/ -q` passes
- [ ] `python make_figures.py ...` regenerates figures from `results/`

## GitHub → Zenodo

1. Create the GitHub repository and push.
2. Sign in to Zenodo with GitHub, go to **Settings → GitHub**, flip the repo on.
   *Do this before creating the release* — Zenodo only captures releases made
   after the switch is enabled.
3. On GitHub, **Releases → Create a new release**, tag `v1.0.0`.
4. Zenodo mints a DOI automatically and archives the tarball.
5. Copy the **concept DOI** (the version-independent one) into `README.md` and
   the paper. It always resolves to the latest version.

## Two DOIs or one?

One repository release gives you one DOI covering code and data together, which
is simplest.

Consider a **separate Zenodo dataset deposit** for the rule corpus
(`results/runs.jsonl` plus the scored CSVs) if you want it citable
independently. Datasets are cited by people who reuse the data without touching
the code, and a separately citable dataset is a distinct piece of evidence of
research contribution. Set its `upload_type` to `dataset` and cross-link the two
in both descriptions.

## In the paper

Data availability statement should name:

- the four corpora with DOIs and access dates (`data/README.md`)
- the artefact concept DOI
- the dataset DOI, if separate

Cite the artefact in the reference list as software, not only as a footnote.

## After acceptance

- [ ] Update `CITATION.cff` `preferred-citation` with journal, volume, pages, DOI
- [ ] Tag `v1.1.0` and release again — Zenodo versions it under the same concept DOI
- [ ] Post the accepted manuscript per publisher policy (IEEE permits the
      accepted version on arXiv permanently, with copyright notice and DOI link)
