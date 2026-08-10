# Corpora

**None of the four corpora are redistributed here.** All are publicly available;
obtain them from the sources below and place them in this directory.

File hashes are recorded so you can confirm you have the same bytes we did.

| ID | Corpus | Rows | Source |
|---|---|---:|---|
| C1 | Bank Marketing | 45,211 | https://doi.org/10.24432/C5K306 |
| C2 | Diabetes 130-US Hospitals | 101,766 | https://doi.org/10.24432/C5230J |
| C3 | Online Retail II | 1,067,371 | https://doi.org/10.24432/C5CG6D |
| C4 | NYC TLC Yellow Taxi, **May 2026** | 4,090,836 | https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page |

C1–C3 can be fetched programmatically:

```python
pip install ucimlrepo
from ucimlrepo import fetch_ucirepo
fetch_ucirepo(id=222)   # C1
fetch_ucirepo(id=296)   # C2
fetch_ucirepo(id=502)   # C3
```

## C4 provenance — read this

The contamination-control argument in the paper depends entirely on C4
post-dating the training cutoff of every model tested. Use the **May 2026**
yellow taxi file specifically; a different month invalidates that argument.

```
file    yellow_tripdata_2026-05.parquet
sha256  9aa5a1609e2bf07d9051b7d530de05b1019e12a560ecb2c59c137c8b3a8b6750
rows    4,090,836   columns 20
```

The TLC page notes the parquet schema may be standardised across years in
future, so the hash is what identifies the version we used.

## Loading

**Do not read these files with library defaults.** Use `llmauth_census.load_corpus`,
which carries an explicit read configuration per corpus and asserts expected row
counts. Two defaults are actively destructive here:

- pandas' default missing-value list contains the string `"None"`. A plain
  `read_csv` on C2 converts 96,420 occurrences of `max_glu_serum = "None"` — a
  valid value meaning the test was not administered — into nulls.
- C3 is a two-sheet workbook. `pd.read_excel(path)` reads only the first sheet
  and silently returns 525,461 of 1,067,371 rows.

```python
from marqbench import llmauth_census as C
df, provenance = C.load_corpus("diabetes_130us", "data/diabetic_data.csv")
```

## Licences

C1–C3 are distributed by the UCI Machine Learning Repository under CC BY 4.0 at
time of access. **Verify the licence on each dataset page before redistributing
anything derived from them** — we found a licence statement for one of these
corpora that did not match its documentation, and re-checked it directly.

C4 is New York City open data; consult the TLC page for terms of use.
