# Paper 2 — Design & Pre-Registration Protocol

**Working title:** *Machine-Authored Data Quality: Information, Not Intelligence, Determines Validation Rule Utility in Lakehouse Data Products*

**Author:** Ramesh Babu Kallam (sole author)
**Status:** Design phase — pre-registration draft
**Depends on:** Paper 1 (INFOSYS-D-26-00790), harness modules `ztlf_*`
**Target venues (verified no-APC route, Aug 2026):** The VLDB Journal → IEEE TKDE → Data & Knowledge Engineering. **All ACM venues are excluded — see §14.**

---

## 1. Thesis and positioning

Paper 1 established that **rule authorship dominates rule enforcement**: three of four validation tools produced identical detections given identical rules, while census-informed vs. census-uninformed authorship produced up to a 0.151 ROC-AUC gap and a retention swing from 0.2–12.3% to 20.5–68.3%.

Paper 2 asks the question that finding forces: **what happens when the author is a language model?**

The thesis — and the title claim — is that **the profile census, not the sophistication of the author, is the dominant factor in rule quality.** A frontier LLM given only a schema should author rules roughly as badly as a naive human heuristic; the same model given the census should approach or match census-informed authorship. If that holds, the contribution is a general principle about machine-authored governance artifacts, not a product review of any particular model.

### 1.1 Delta from Paper 1 (state this explicitly in the manuscript)

| | Paper 1 | Paper 2 |
|---|---|---|
| Manipulated variable | Rule *information* (census vs. no census), enforcement tool | Rule *author* (human-proxy heuristic vs. LLM), crossed with information |
| Novel construct | Census-informed authorship; injected-only evaluation bias | Machine authorship; **rule-set non-determinism**; failure taxonomy |
| Evaluation harness | Built | **Reused unchanged and cited by DOI** |
| Downstream finding | No gate improved discrimination (null) | Tests whether machine authorship changes that null |

Reuse of the harness is a *strength* (replication infrastructure), but it must be imported from a pinned Zenodo release and cited as software, never copy-pasted. Otherwise a reviewer or an integrity screen reads it as self-plagiarism.

### 1.2 Competitive landscape

The nearest prior work is Abughazala et al., *"Quality by Prompt: LLM-Powered Transformation of Data Quality Requirements Into Great Expectations"* (SEAA 2025, LNCS 16081), which fine-tunes an LLM and evaluates generated rules on fluency, accuracy, and domain alignment. That measures whether rules **look** right.

The open gap: **nobody measures whether machine-authored rules are right in *consequence*** — retention, precision against natural (not injected) defects, downstream model utility, subgroup fairness, and stability across invocations. That is precisely what your Paper 1 harness measures. Cite this paper prominently and position against it in the related work section; do not ignore it.

---

## 2. Research questions and pre-registered hypotheses

Register these on OSF **before** running Phase N2 (generation). Pre-registration is cheap, takes an afternoon, and is disproportionately persuasive to reviewers who have just been burned by a decade of unregistered LLM papers.

| ID | Research question |
|---|---|
| **RQ1** | How do LLM-authored validation rules compare to census-informed and census-uninformed baselines on detection quality against natural-defect ground truth? |
| **RQ2** | Does supplying the profile census in the prompt close the authorship gap — i.e., is information or author capability the dominant factor? |
| **RQ3** | How much does a rule set vary across repeated invocations at fixed prompt, and does that variance propagate to retention and downstream discrimination? |
| **RQ4** | What are the categories and frequencies of defective machine-authored rules? |
| **RQ5** | Do machine-authored gates change downstream model discrimination or subgroup fairness relative to no gate, and at what authoring cost? |

### Hypotheses

- **H1 (competence deficit).** Schema-only LLM rules will report high nominal compliance while retaining materially fewer records than census-informed rules — reproducing the census-uninformed pattern from Paper 1. *Directional; predicted retention gap > 20 percentage points on at least 2 of 3 corpora.*
- **H2 (information dominance — the core hypothesis).** Adding the census to the prompt (condition A4) will recover the majority of the retention gap between A2 and A1. *Pre-specified threshold: A4 recovers ≥ 50% of the A2→A1 gap.* Effect of **information condition will exceed effect of model identity** in the pooled model. This is the claim the title makes; it must be falsifiable and it must be possible for it to fail.
- **H3 (non-determinism).** At fixed prompt and non-zero temperature, numeric thresholds will vary across seeds with coefficient of variation > 0.10, and across-seed rule-set Jaccard similarity will be < 0.80, on a majority of corpus × model cells.
- **H4 (downstream null — pre-registered as possibly null).** No authorship condition will improve downstream ROC-AUC relative to no gate by more than 0.01. **This hypothesis is registered in the null direction on purpose.** Paper 1 found this; if Paper 2 reproduces it, that is a replication and it gets reported as a headline, not buried.
- **H5 (sentinel trap).** Rule authors without the census — human-proxy *and* machine — will disproportionately flag encoded-missing sentinel values as violations. *Pre-specified: sentinel-targeting rules (code F9) will appear in > 40% of A2 rule sets.*

---

## 3. The sentinel trap (verified against the actual corpora)

This is the paper's most concrete and most quotable mechanism, and I verified every number below directly against the CSVs rather than relying on dataset documentation.

### Bank Marketing (n = 45,211)

| Column | Sentinel / trap | Prevalence | Naive rule that destroys the corpus |
|---|---|---|---|
| `pdays` | `-1` means "not previously contacted" | **81.7%** (36,954 rows) | `pdays >= 0` → **retains 18.3%** |
| `poutcome` | `"unknown"` is a valid level | **81.7%** (36,959) | `poutcome IN ('success','failure','other')` → retains 18.3% |
| `contact` | `"unknown"` is a valid level | 28.8% (13,020) | `contact IN ('cellular','telephone')` → retains 71.2% |
| `balance` | negative = legitimate overdraft | 8.3% | `balance >= 0` → retains 91.7% |
| `education` | `"unknown"` is a valid level | 4.1% (1,857) | domain-restriction rule → retains 95.9% |

### Diabetes 130-US Hospitals (n = 101,766)

| Column | Sentinel / trap | Prevalence | Note |
|---|---|---|---|
| `max_glu_serum` | literal string **`"None"`** = test not administered | **94.7%** (96,420) | **Best trap in the study.** `"None"` *looks like* a null literal but is a semantically valid value. A null-token rule annihilates the column. |
| `A1Cresult` | literal string `"None"` = test not administered | **83.3%** (84,748) | Same mechanism |
| `weight` | `"?"` = missing | **96.9%** (98,569) | Completeness rule drops nearly everything |
| `medical_specialty` | `"?"` = missing | 49.1% (49,949) | |
| `payer_code` | `"?"` = missing | 39.6% (40,256) | |
| `race` | `"?"` = missing | 2.2% (2,273) | **Fairness-hazardous:** gating on this systematically removes records with unrecorded race |
| `age` | string buckets `"[70-80)"`, not numeric | 100% | Type-mismatch trap: a numeric range rule on `age` is non-executable |
| `gender` | `"Unknown/Invalid"`, **n = 3** | 0.003% | Rare-category trap: census at low sample rates may miss it |

### NYC TLC Yellow Taxi, 2026-05 (n = 4,090,836, 20 columns)

Verified against the downloaded file, SHA-256 `9aa5a1609e2bf07d9051b7d530de05b1019e12a560ecb2c59c137c8b3a8b6750`.

| Column | Trap | Prevalence | Note |
|---|---|---|---|
| `RatecodeID` | **Three-way**: `NaN` (true null), `99` (documented "Null/unknown"), `1`–`6` (valid codes) | NaN **23.4%** (955,371); `99` **3.4%** (140,897) | **The strongest trap in the study** — see below |
| `payment_type` | `0` = Flex Fare, a valid but obscure level | **23.4%** (955,371) | Count is *identical* to `RatecodeID` NaN — structurally coupled |
| `passenger_count` | `0` occurs (driver-reported) | 0.31% | Low-prevalence variant of the same mechanism |
| `Airport_fee` | Capitalized while every sibling column is lowercase | n/a | **Naming-inconsistency trap**: an author writing `airport_fee` earns F1 |
| `cbd_congestion_fee` | Added for 2025 data onward | n/a | Schema-evolution trap, absent from C1–C3 |
| `payment_type` `5`/`6` | Documented sentinels (Unknown, Voided) | **absent this month** | Registry retains them; F9 simply never fires on them here. Report the absence. |

**`RatecodeID` is a three-way discrimination problem and no other corpus has one.** An author must distinguish a genuine null (23.4%), an encoded "unknown" that is *not* null (3.4%), and six valid codes — in a single column. C1's `pdays` and C2's `max_glu_serum` are two-way. A rule such as `RatecodeID IN (1,2,3,4,5,6)` discards **26.80%** of the month (1,096,268 rows); `RatecodeID IS NOT NULL` discards 23.35%; an author writing both discards 26.80% while believing they addressed two separate quality problems. This deserves its own subsection in the results.

### Block-structured missingness in C4 — verified, and it breaks an assumption

Three columns are null or zero-coded on **exactly the same 955,371 rows** (23.3535%, matching to the row):

| Column | Value | Count |
|---|---|---|
| `RatecodeID` | `NaN` | 955,371 |
| `payment_type` | `0` (Flex Fare) | 955,371 |
| `passenger_count` | `NaN` | 955,371 |

Flex Fare trips arrive through a pipeline that populates none of these three fields. The missingness is **block-structured**, not column-independent.

**Consequence for the study.** Retention accounting implicitly assumes rules remove roughly independent record sets, so that stacking rules stacks losses. Here, three completeness rules on three different columns — which any author would regard as addressing three distinct problems — remove the *identical* 955,371 rows. Set retention is unchanged whether one, two, or all three are enforced.

This yields a measurable phenomenon nobody in the rule-authorship literature has reported: **rule effects are correlated, and the correlation structure is invisible from the schema.** Whether the census condition (A4/A5) lets an author detect the coupling — and whether any author writes the `cross_column` rule that actually expresses it — is a strong secondary result. Add the **rule-effect correlation matrix** described in §8 to the analysis plan.

### Sub-threshold over-rejection in C4 (the F4 blind spot, with real numbers)

| Naive rule | Removes | Why it is wrong |
|---|---|---|
| `fare_amount >= 0` | 14,231 rows (0.348%) | Refunds and voided trips are valid records |
| `total_amount >= 0` | 14,877 rows (0.364%) | Same |
| `trip_distance > 0` | 113,031 rows (2.763%) | Cancellations, GPS failures, genuine short trips |
| `passenger_count > 0` | 12,533 rows (0.306%) | Driver under-reporting, not invalid data |

None of these trips F4 (each retains > 97%), none is a registered sentinel so none trips F9, and none contradicts the census so none trips F3. **All four receive zero failure codes** — exactly the `balance >= 0` blind spot found on C1. This is the empirical justification for the continuous excess-rejection measure in §8; C4 supplies four clean instances of it.

**Why this matters for the paper.** These are not synthetic traps you injected — they are naturally occurring encoding conventions in widely used public corpora. Rules that fail on them fail for a *semantic* reason (misreading an encoding convention), not a statistical one. That makes the failure mode generalizable and gives the paper a mechanism, not just a benchmark table. Lead the discussion section with `max_glu_serum = "None"`.

---

### 3.3 Dataset sources, DOIs, and licenses

| ID | Source page | DOI | License |
|---|---|---|---|
| C1 | `https://archive.ics.uci.edu/dataset/222/bank+marketing` | 10.24432/C5K306 | CC BY 4.0 |
| C2 | `https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008` | 10.24432/C5230J | CC BY 4.0 |
| C3 | `https://archive.ics.uci.edu/dataset/502/online+retail+ii` | 10.24432/C5CG6D | CC BY 4.0 — **re-verify, see below** |
| C4 | `https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page` | n/a — NYC open data | Confirm terms of use on the page |

Programmatic access for C1–C3: `pip install ucimlrepo`, then `fetch_ucirepo(id=222 / 296 / 502)`.

**C3 licensing — check this one yourself.** The UCI page currently states CC BY 4.0, but you previously caught an incorrect CC BY 4.0 claim about this specific dataset. Either UCI has since corrected the page, or the earlier problem lay elsewhere — a mirror, a derived copy, or the underlying source's terms. Open the page, confirm the license text, screenshot it with the access date, and store that in the artifact repository. Do not carry the claim forward from Paper 1 on trust.

**C4 data dictionary:** `https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf`

**C4 sentinel structure is confirmed, and it is excellent.** The published yellow-taxi data dictionary documents encoded sentinels outright:

- `RatecodeID`: **`99` = Null/unknown** — a numeric sentinel documented as such, structurally identical to Bank's `pdays = -1`
- `payment_type`: `5` = Unknown, `6` = Voided trip, `0` = Flex Fare
- `VendorID`: coded provider identifiers, with codes added over time
- `passenger_count`: driver-reported, so `0` occurs
- `cbd_congestion_fee`: added for 2025 data onward — a **schema-evolution trap** absent from the other three corpora

C4 is therefore a genuine replication of the sentinel mechanism in a fourth domain, on data that could not have been memorized. **C4 is confirmed; the `folktables` fallback is no longer needed.**

**C4 month: SELECTED AND LOCKED.** `yellow_tripdata_2026-05.parquet`, 4,090,836 rows × 20 columns, SHA-256 `9aa5a1609e2bf07d9051b7d530de05b1019e12a560ecb2c59c137c8b3a8b6750`. May 2026 post-dates the training cutoff of every model in the panel. Sample to ~1M rows with a pinned seed to match C3's scale and stay within Colab's memory budget; archive the sampled row-index manifest.

Note the TLC page's warning that the parquet schema may be standardized across years in future. The hash above is what protects the reproducibility claim if the hosted file is silently replaced — verify it before every rerun.

**C4 downstream target:** `tip_amount > 0`, observed at **62.3%** — a well-balanced binary label, better balanced than C1 (`y` at 11.7%) or C2.

**Target leakage — mandatory feature exclusion.** The dictionary states tip amounts are populated automatically for card payments and that cash tips are not recorded. Cash trips (`payment_type = 2`) are 9.1% of the month and will show `tip_amount = 0` almost by construction. **`payment_type` must be excluded from the downstream feature set on C4**, and the exclusion must be stated in the manuscript. Leaving it in would let the model partly read the label off a payment code, and a reviewer familiar with this data will look for exactly that.

Related consequence: this makes `payment_type` a **fairness-relevant** variable for C4 rather than a nuisance column. A rule that gates on payment type differentially removes cash-paying riders, who are not randomly distributed across pickup zones. Treat `payment_type` as a protected-adjacent attribute in the F10 analysis alongside pickup zone.

Frame the label as **"recorded tip," not "tipped,"** throughout.

## 4. Experimental design

### 4.1 Factors

**Corpus (4).** Three carried forward from Paper 1 for baseline comparability, plus one new corpus for contamination control.

| # | Corpus | Rows | Role | Downstream label |
|---|---|---|---|---|
| C1 | UCI Bank Marketing | 45,211 | Numeric sentinel structure (`pdays = -1` @ 81.7%) | `y` |
| C2 | UCI Diabetes 130-US | 101,766 | Categorical sentinels (`"None"`, `"?"`); protected attributes for F10 | `readmitted` |
| C3 | UCI Online Retail II | ~1.07M | Scale, cost, and retention at volume | none |
| C4 | **NYC TLC trip records, recent month** | ~1–3M (sample to ~1M) | **Contamination control + external validity** | derived (e.g. tip > 0) |

C1–C3 are **not optional**: conditions A0 and A1 are Paper 1's baselines, and those baselines only mean anything on the corpora where they were established. Comparability requires reuse.

C4 exists to answer the two objections reuse invites. Choose a month released **after the training cutoffs of every model tested**. The schema is public and memorizable; the *values and their distributions* cannot be. That makes C4 a clean test of whether the observed authorship deficit survives when memorization is impossible — which is the strongest single robustness result the paper can carry.

**C4 is confirmed** — its published data dictionary documents encoded sentinels directly (`RatecodeID = 99` means Null/unknown). Details and the remaining verification steps are in §3.3. The `folktables` fallback is retired.

Downstream modelling on C1, C2, and C4. C3 carries no natural supervised label and serves as the scale corpus for detection, retention, and cost only.

**Author condition (6).** The information ladder is the design's spine.

| ID | Author | Prompt payload |
|---|---|---|
| **A0** | Heuristic (Paper 1) | Census-uninformed rules — deterministic baseline |
| **A1** | Heuristic (Paper 1) | Census-informed rules — deterministic ceiling |
| **A2** | LLM | Schema only (column names + inferred dtypes) |
| **A3** | LLM | Schema + data dictionary (UCI prose descriptions) |
| **A4** | LLM | Schema + census (Phase-1 profile statistics) |
| **A5** | LLM | Schema + census + 20 sampled rows |

A3 is not decoration: it separates *semantic* information (documentation, which a practitioner would have) from *statistical* information (the census). If A3 recovers most of the gap, the story changes from "you need profiling" to "you need documentation" — which is a different and also publishable finding. Design for either outcome.

**Model (4).** Two open-weight, locally run in Colab at 4-bit quantization (e.g. Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct); two commercial API models (one frontier, one small/cheap). Rationale: the open-weight pair makes the study fully reproducible without credentials — the same reproducibility argument that justified leaving Azure Databricks — and the open-vs-commercial contrast is a free secondary finding.

**Seeds (10)** per LLM cell at temperature 0.7, plus a **temperature sub-study** at T = 0.0 with 10 seeds on one corpus, to separate "non-determinism from sampling" from "non-determinism from infrastructure." Note that T = 0 is not truly deterministic in batched inference — that is itself worth a sentence in the paper.

**Generation runs:** 4 corpora × 4 LLM conditions × 4 models × 10 seeds = **640** generations, plus 8 deterministic baseline rule sets and the temperature sub-study.

Half run locally on Colab (open-weight, free). The 320 commercial calls at roughly 4k input / 1.5k output tokens each come to about **$6–10 total**; budget **$25** to absorb the pilot, retries, and the no-warning ablation. That is two orders of magnitude below a single APC and is the only money this paper needs.

### 4.2 The authoring / evaluation split — do not skip this

**Rules are authored from a profile computed on a 20% authoring split; all evaluation happens on the disjoint 80% holdout.**

Without this, the LLM's census describes the same rows the rules are then scored on, and a reviewer will correctly call it leakage. Fix the split with a pinned seed, store the row-id manifest as an artifact, and state it in the abstract. This is the single most likely methodological objection and it costs you nothing to pre-empt.

### 4.3 Anti-tautology safeguards

Given the DATAK history, make these explicit in a numbered subsection of the Methods. Reviewers who see a paper openly guarding against tautology trust the rest of it more.

1. **The LLM never sees the corruption engine, the injection specification, or any Phase-2 code.** Prompt payloads are restricted to schema, dictionary, census, and sampled rows. Log the exact payload for every run.
2. **Ground truth is the natural-defect census plus independently specified injections.** Report detection metrics against *both* the union and the injected-only ground truth, so Paper 1's evaluation-bias finding is carried forward rather than re-committed.
3. **Enforcement is held constant.** All rule sets, whatever their author, compile to one intermediate representation and execute on one engine. Paper 1 already licensed this by showing three of four tools were identical — cite that result as the justification.
4. **Rule authoring and defect injection are separately versioned modules with no shared imports.** Enforce with an import-graph assertion in CI so it is checkable, not merely claimed.

---

## 5. Rule intermediate representation (IR)

Force every model to emit structured JSON against a fixed schema. This makes compilation deterministic, comparison model-agnostic, and lets you compile one rule set to Great Expectations, Deequ, and Delta CHECK constraints from a single source. **The IR is itself a citable contribution — package it.**

```json
{
  "rules": [
    {
      "rule_id": "string",
      "column": "string",
      "dimension": "completeness|validity|consistency|uniqueness|accuracy|timeliness",
      "predicate_type": "not_null|in_set|range|regex|unique|cross_column|type",
      "parameters": { "min": 0, "max": 100, "allowed": ["a","b"], "pattern": "..." },
      "severity": "reject|quarantine|warn",
      "rationale": "one sentence, free text"
    }
  ]
}
```

The `rationale` field is not cosmetic — it is your qualitative data for the failure taxonomy and it lets you show *why* a model flagged `max_glu_serum = "None"`. Quote a few verbatim rationales in the paper; they are memorable and get screenshotted.

**Compilation discipline:** any rule that fails schema validation is recorded as failure code F7 and counted, never silently dropped. Silent dropping would inflate the quality of the surviving rules and is exactly the kind of bug you caught twice in Paper 1.

---

## 6. Prompt specification

Fix one system prompt across all conditions. Vary only the payload block. Store all prompts as files under version control; never inline them in a notebook cell where they can drift.

**System prompt (constant):**

> You are a data quality engineer. Given a description of a table, author validation rules that identify genuinely defective records. Emit only JSON conforming to the provided schema, with no prose outside the JSON. Do not author rules for columns not listed. Prefer rules that would reject records a domain expert would consider erroneous, and avoid rules that would reject records that are merely unusual or that use a documented encoding convention.

The final clause deserves a note: it makes the sentinel trap a **fair test**. The model is explicitly warned about encoding conventions. If it walks into the trap anyway without the census, that is a much stronger finding than if you had said nothing. Run a small ablation without that clause on one corpus to quantify how much the warning helps — likely answer: very little, which is a great result.

**Payload blocks by condition:**

- **A2:** column names + dtypes only.
- **A3:** A2 + the UCI prose description per column.
- **A4:** A2 + census: null rate, distinct count, min/max/quantiles for numerics, top-k categories with frequencies, detected sentinel candidates *reported as observed values without interpretation* (e.g. "`pdays`: min −1, 81.7% of values = −1"). Do not tell the model that −1 is a sentinel — that would hand it the answer and make the finding tautological.
- **A5:** A4 + 20 rows sampled with a pinned seed from the authoring split.

---

## 7. Pre-specified failure taxonomy

Fix these codes **before** looking at any output; post-hoc taxonomies are unfalsifiable. Code every generated rule. Double-code a 15% random sample after a two-week gap and report intra-rater agreement (Cohen's κ) — as a sole author you cannot do inter-rater, so intra-rater with a washout is the honest substitute, and saying so openly is better than pretending otherwise.

| Code | Failure mode |
|---|---|
| **F1** | Hallucinated column (references a column not in schema) |
| **F2** | Hallucinated category (allowed-set omits or invents observed levels) |
| **F3** | Threshold contradicts census (bound outside observed range) |
| **F4** | Over-tight (rule alone retains < 50% of records) |
| **F5** | Vacuous (rule is satisfied by 100% of records) |
| **F6** | Type mismatch (e.g. numeric range on `age` string buckets) |
| **F7** | Non-executable (fails IR schema validation or compilation) |
| **F8** | Redundant (semantically duplicates another rule in the same set) |
| **F9** | **Sentinel misread** (flags a documented encoded-missing value as a defect) |
| **F10** | **Fairness-hazardous** (gates on or via a protected attribute — `race`, `gender`, `age`) |

F10 is worth its own paragraph in the discussion. A machine-authored rule that quarantines records with `race = "?"` silently removes 2,273 patients and shifts the subgroup composition of the training set. That is a governance failure with a compliance dimension, and it connects your work to the fairness literature — which is a much larger citing community than data quality.

---

## 8. Metrics

**Rule-level.** Syntactic validity rate; schema-groundedness (fraction referencing real columns); rules per set; dimension distribution; failure-code frequencies.

**Detection.** Precision, recall, F1 against (a) natural-defect census ∪ injections and (b) injected-only. Report the **ratio** between the two — this extends Paper 1's 28× finding to machine-authored rules and is a direct continuity hook.

**Retention.** Percent of holdout records surviving the gate, per rule set and per rule.

**Continuous over-rejection** (reported alongside F4, which is a coarse categorical threshold): **sentinel rejection rate**, the proportion of evaluation records rejected solely because a value is a registered sentinel; and **excess rejection**, retention under A1 minus retention under the scored condition. These catch rules that discard a harmful but sub-threshold fraction of valid records — `balance >= 0` on C1 (8.3% of legitimate overdrafts) and `fare_amount >= 0` on C4 (14,231 refund and voided-trip records). Both receive zero failure codes under the categorical taxonomy; see §3.

**Rule-effect correlation** (added after C4 verification, before registration). For each rule set, compute pairwise Jaccard similarity of the *record sets each rule rejects* on the evaluation split. Report mean off-diagonal similarity, and the ratio of set-level retention to the product of per-rule retentions — near 1 means near-independent effects, far above 1 means heavy overlap.

C4's Flex Fare block is the motivating case: three completeness rules on three different columns remove the identical 955,371 rows, so set retention is unchanged whether one or all three are enforced. Independence-assuming retention accounting predicts a compounding loss that never occurs. Report per corpus and per condition, and test whether census-informed authors produce less redundant rule sets than schema-only authors.

**Downstream.** ROC-AUC, PR-AUC, Brier score, plus subgroup metrics: Bank (age band, `job`), Diabetes (`race`, `gender`). Compare against the no-gate baseline.

**Stability (RQ3, your differentiator).** Across-seed Jaccard similarity of rule sets; coefficient of variation of numeric thresholds; standard deviation of retention and of downstream ROC-AUC across seeds. **Report the SD of downstream AUC across seeds alongside the between-condition effect.** If seed variance is comparable to condition effects, that is a genuinely alarming and highly citable result about the auditability of machine-authored governance.

**Cost.** Input/output tokens, USD, wall-clock, and cost per *retained* record — a metric that punishes over-tight rules and that practitioners will actually quote.

---

## 9. Statistical analysis plan

- **Primary model:** mixed-effects regression, `outcome ~ information_condition + model_family + (1 | corpus)`, with seed as a nested random effect for LLM cells. With only 3 corpora, also report per-corpus results in full; do not let pooling hide a corpus-specific effect.
- **H2 test:** compare variance explained by information condition vs. model identity (partial R² or equivalent). This is the headline number in the abstract.
- **Multiplicity:** Holm–Bonferroni across the pre-specified family of primary comparisons. Everything else is labelled exploratory. Label it honestly — reviewers respect a clearly marked exploratory section far more than they respect five hypotheses that all happened to confirm.
- **Effect sizes:** Cliff's delta for non-normal outcomes; 95% bootstrap CIs (10,000 resamples) on every reported difference. No bare p-values.
- **Power:** n = 10 seeds is a pragmatic choice, not a powered one. State that plainly, report observed variance, and note what it implies for future replications.

---

## 10. Notebook and module architecture

Drive folder: **`Paper2_RuleAuthorship`**

| Notebook | Purpose |
|---|---|
| `P2_N0_environment.ipynb` | Pin PySpark 3.5.3 / delta-spark 3.2.1 / OpenJDK 11; install `ztlf_*` from pinned Zenodo release; build and freeze the authoring/evaluation split manifest |
| `P2_N1_census_and_payloads.ipynb` | Profile the authoring split (`ztlf_profiling`); assemble the four prompt payload variants; snapshot every payload |
| `P2_N2_generation_sweep.ipynb` | 480 generations; capture raw model output verbatim before any parsing |
| `P2_N3_compile_and_code.ipynb` | IR validation, compilation, failure-code assignment, detection scoring |
| `P2_N4_retention_downstream.ipynb` | Retention, downstream models, subgroup fairness, stability, cost |
| `P2_N5_figures_tables.ipynb` | All figures and tables, regenerable from stored intermediates |

**New modules (write from scratch):** `llmauth_prompts.py`, `llmauth_generate.py`, `llmauth_ir.py`, `llmauth_compile.py`, `llmauth_taxonomy.py`, `llmauth_stability.py`.

**Reused unchanged from Paper 1:** `ztlf_profiling`, `ztlf_specs`, `ztlf_corruption`, `ztlf_baselines`, `ztlf_downstream`. Import from the pinned release. If you find a bug in a `ztlf_` module during Paper 2, fix it in a **new patch release**, cite that version, and disclose the change — do not silently edit the code Paper 1's results depend on.

**Capture raw output before parsing.** The single most common irrecoverable mistake in LLM studies is storing only the parsed rules. When your parser turns out to have a bug in month three, the raw completions are the only thing that saves you from regenerating everything.

---

## 11. Reproducibility and version pinning

Commercial models change under fixed names, which will silently break replication and is a real threat to a paper whose subject is stability.

- Record the **exact model version string** and the **UTC timestamp** for every commercial API call.
- Pin open-weight models by **Hugging Face commit hash**, not tag.
- Log full request parameters: temperature, top-p, max tokens, seed, quantization config.
- Publish the complete raw completion corpus. This is the artifact that makes the paper replicable even after the commercial models are retired — and it is a citable dataset in its own right.

Turn this constraint into a contribution: a short subsection on **"the reproducibility half-life of machine-authored governance artifacts"** will be quoted by people writing about LLM evaluation generally, well outside data quality.

---

## 12. Threats to validity (draft the section now, not at the end)

- **Construct.** Prompt phrasing is a researcher degree of freedom. Mitigate with the fixed system prompt, the pre-registered payload ladder, and the no-warning ablation.
- **Internal.** Authoring/evaluation split prevents census leakage; the import-graph assertion prevents corruption/authoring contamination.
- **External.** Three public tabular corpora in two domains. Do not claim generality to streaming, nested, or unstructured data. Say so before a reviewer does.
- **Contamination.** These are famous UCI datasets; the models have almost certainly seen them in pretraining. **Address this head-on** — it is the second-most-likely reviewer objection. Probe it: ask each model to reproduce the schema and value distributions without being shown them, and report memorization rates. If models can recite the corpora, that *biases toward better LLM performance*, which makes any observed deficit a conservative lower bound. Frame it that way and the objection becomes a strength.
- **Conclusion.** n = 10 seeds; effect sizes and CIs reported throughout.

---

## 13. Artifact and citation strategy

You asked for citations. Citations follow *reusable artifacts* far more reliably than they follow good prose. Concretely:

1. **Name the benchmark: MARQ-Bench.** MARQ = *Machine-Authored Rule Quality*. Use **MARQ** for the construct in prose ("MARQ scores," "the MARQ dimensions") and **MARQ-Bench** for the artifact — repository, package, and leaderboard table. There is no collision in the CS literature, but bare "MARQ" competes with a consumer electronics brand in general web search; the `-Bench` suffix eliminates that, follows the dominant convention (SWE-bench, LiveBench, CSR-Bench), and immediately signals the artifact type to anyone scanning a reference list.
2. **Three separately citable artifacts, three DOIs:** (a) the harness + benchmark code, (b) the raw rule corpus — thousands of machine-authored rules with failure codes, which is a dataset other people can analyze without rerunning anything, (c) the pinned `ztlf_*` release Paper 1 also cites.
3. **Ship a pip-installable package.** `pip install marq` with a five-line quickstart is worth more citations than a perfect discussion section. Most people who cite a benchmark never read the paper closely; they cite it because they ran it.
4. **arXiv preprint at submission time.** Permitted by Elsevier and by ACM. Preprints accumulate citations during the 6–12 month review latency, which matters given your filing timeline.
5. **A leaderboard-shaped results table.** Format the main results so a future author can add one row for their model. That makes your paper the natural citation for anyone doing the same thing later.

One honest note on timing: citations accrue over 18–36 months. For an EB-1A filing sooner than that, this paper's value sits mostly in the *authorship* criterion and — through the artifacts — in building a documented case for *original contributions of major significance*. If your filing window is tight, the artifact and adoption evidence (downloads, GitHub stars, forks, independent usage) will be doing more work than the citation count. Plan to collect that evidence systematically from day one: it is much harder to reconstruct later.

---

## 14. Venue logistics

### 14.1 Correction: ACM is no longer viable for you

An earlier draft of this protocol recommended ACM JDIQ. **That recommendation was wrong and is withdrawn.**

As of 1 January 2026, ACM became a fully open-access publisher — every ACM journal, magazine, and proceedings. Publication is free only if the corresponding author is affiliated with an institution participating in ACM Open, or if the author's institution is in a country the World Bank classifies as low-income. Otherwise an APC applies.

You meet neither condition: you are an unaffiliated independent researcher in the United States. Worse, ACM's discretionary waiver policy addresses your exact situation and closes the door on it — it states that inability to pay because one is a graduate student or an independent consultant without an institutional affiliation is not itself a demonstration of financial hardship, and that decisions turn on extraordinary circumstances such as war, political instability, or natural disaster.

**Consequence: JDIQ, TODS, and TKDD are all off the table**, along with every ACM conference. This is a structural constraint on your entire publication program, not a Paper 2 detail — plan the whole series around it.

### 14.2 Verified alternatives

| Venue | Publisher | No-APC route | Notes |
|---|---|---|---|
| **The VLDB Journal** | Springer | **Verified.** Hybrid; the subscription publishing model carries no APC and is offered after acceptance | Gold OA would be $2,990 — decline it. Primary target. |
| IEEE TKDE | IEEE | Hybrid, traditional route free | Confirm current terms at submission |
| Data & Knowledge Engineering | Elsevier | Hybrid, subscription route free | Prior withdrawal there was pre-review and is not disqualifying |
| Information Systems | Elsevier | Hybrid, subscription route free | Hold until Paper 1 clears |

**Primary target: The VLDB Journal.** Better prestige than JDIQ, verified free route, and the LLM-for-data-management theme is squarely in scope.

**Re-verify before submission.** The ACM change landed in January 2026 and Springer's consortium quotas shift mid-year. Publisher OA terms are now volatile enough that "it was free last year" is not evidence. Check the journal's own fees page the week you submit, and screenshot it.

**Green OA fallback.** Choosing the subscription route means the paper sits behind a paywall, which suppresses exactly the citations you want. Counter it: post the arXiv preprint at submission, and deposit the accepted manuscript per the publisher's self-archiving policy. The preprint plus the MARQ-Bench artifact will carry most of the discoverability regardless of the paywall.

**Compliance items to prepare regardless of venue:**
- **Generative AI disclosure.** LLMs here are the *object of study* — that is methodology and belongs in Methods. Any use of AI in *manuscript preparation* is separate and requires the declaration statement before the reference list. Conflating the two looks careless.
- **Data statement + repository DOIs.**
- **Highlights:** 3–5 bullets, ≤ 85 characters each, for Elsevier venues.
- Confirm each commercial model's **terms of service permit publishing benchmark results** before running the sweep.

---

## 15. Indicative timeline

| Weeks | Milestone |
|---|---|
| 1–2 | OSF pre-registration; N0 environment + split manifest; prompt files frozen |
| 3–4 | N1 census + payloads; IR schema + compiler; pilot on Bank only (1 model, 3 seeds) |
| 5 | **Pilot review checkpoint** — sanity-check before committing to the full sweep |
| 6–8 | N2 full generation sweep; raw corpus archived |
| 9–11 | N3 compilation, failure coding, detection scoring; intra-rater washout begins |
| 12–14 | N4 retention, downstream, fairness, stability, cost |
| 15–16 | N5 figures; artifact packaging; DOIs minted |
| 17–20 | Manuscript draft, threats section, internal review |
| 21 | arXiv preprint + JDIQ submission |

The week-5 pilot checkpoint is deliberate. The DATAK submission failed because a design flaw was found after everything was built. A pilot on one corpus with one model surfaces IR problems, prompt problems, and scoring problems while they still cost days instead of months.

---

## 16. Locked decisions

| # | Decision | Resolution |
|---|---|---|
| 1 | Benchmark name | **MARQ-Bench** (construct: MARQ) |
| 2 | Model panel | 2 open-weight 4-bit local + 2 commercial, one frontier and one small-tier, **from different vendors** |
| 3 | Condition A3 (data dictionary) | **Included** |
| 4 | Temperature sub-study corpus | **C2 Diabetes** |
| 5 | Corpora | C1–C3 carried forward, **C4 added** for contamination control |

**On the model panel (#2).** Select the two commercial models on three criteria, in this order: (a) terms of service explicitly permit publishing benchmark results — verify in writing before the sweep, not after; (b) a stable, dated version string you can pin; (c) one frontier-tier and one small-tier model **from different vendors**, so a vendor-specific quirk cannot masquerade as a finding. The specific models matter far less than the tier spread, because the paper's claim is about information conditions, not about which vendor wins. Name them in the manuscript with version strings and access dates, and frame the model panel as a sample from a population of authors — never as a leaderboard. A paper that says "Model X beat Model Y" is obsolete in six months; a paper that says "the census beat every model" is not.

**Why A3 stays (#3).** Without it, the only comparison is statistics-vs-nothing, and a reviewer will fairly ask whether ordinary documentation would have sufficed — no profiling infrastructure required. A3 answers that. If A3 alone closes most of the gap, the paper's practical recommendation changes from "build profiling" to "write data dictionaries," which is cheaper, more actionable, and just as publishable. Design for either result.

**Why Diabetes for the temperature study (#4).** It has the densest and most varied sentinel structure, so it gives seed-to-seed variation the most surface to express itself on. If thresholds are going to wobble anywhere, they will wobble on `max_glu_serum`.

## 17. Remaining verification tasks

1. **C3 license.** Open the UCI page, confirm the license text, screenshot with access date. You caught an error here once before.
2. **C4 month + terms.** Pick the most recent month post-dating all model cutoffs; record URL, file hash, download date; confirm the NYC terms of use permit redistributing derived artifacts.
3. **Commercial model ToS** permit published benchmarking.
4. **Venue terms at submission.** Re-check The VLDB Journal's fees page the week you submit and screenshot it. Publisher OA terms are volatile — ACM's January 2026 change is the proof.
5. **Namespace.** Confirm `marq-bench` is free on PyPI and GitHub before minting DOIs.
