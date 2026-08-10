# Upstream bug report — `tabmemcheck`

**Repository:** https://github.com/interpretml/LLM-Tabular-Memorization-Checker
**Version affected:** 0.1.6 (and any version containing the code below)
**Severity:** silent data corruption — produces plausible, well-formatted, wrong results

Submit as a GitHub issue, ideally with the patch below as a pull request. Text
is ready to paste; replace the placeholder in the reproduction section with your
own adapter if you prefer.

---

## Title

`first_token_test` leaks `config.max_tokens` on exception, silently truncating all subsequent generations to one token

## Body

### Summary

`first_token_test` temporarily lowers the global `tabmem.config.max_tokens` to
`num_digits` (typically 1) as a speed optimisation, then restores it. The
restore is a plain statement, not a `finally` block. If any exception is raised
between the two — for example an API error, a rate limit, or a network failure
during the row-completion phase — the restore never executes and the global cap
remains at 1 for the rest of the interpreter session.

Every subsequent request, in every test and on every dataset, is then truncated
to a single token. Responses come back empty, and the tests report
`0/N exact matches` without any error.

The failure is silent. Output is well-formatted and plausible. In our case it
inverted the conclusion: a model that reproduces 14 consecutive rows of UCI Bank
Marketing character-for-character was reported as showing no memorization at all.

### Location

`tabmemcheck/functions.py`, in `first_token_test`:

```python
#  set max_tokens to the number of digits (speedup)
prev_max_tokes = tabmem.config.max_tokens
tabmem.config.max_tokens = num_digits

# perform a row completion task
if llm.chat_mode:
    _, test_suffixes, responses = row_chat_completion(...)
else:
    _, test_suffixes, responses = row_completion(...)

# reset max_tokens
tabmem.config.max_tokens = prev_max_tokes      # <-- skipped on exception
```

### Reproduction

The exception must occur **after** the cap is lowered, i.e. during the row
completion phase. Two earlier checks (`build_first_token` and a statistical
randomness pre-check) can return or raise before that point, in which case no
leak occurs — so the CSV must be constructed to pass both. All-numeric columns
with a random leading value satisfy them.

```python
import numpy as np, pandas as pd, tempfile, os
import tabmemcheck as tabmem

# A CSV that passes build_first_token (>=3 distinct leading tokens, none >50%)
# and the randomness pre-check (rows are independent).
rng = np.random.default_rng(0); n = 800
df = pd.DataFrame({
    "id": rng.integers(100, 999, n),       # random 3-digit leading column
    "a":  rng.normal(0, 1, n).round(3),
    "b":  rng.integers(0, 5, n),
    "c":  rng.normal(10, 3, n).round(2),
})
path = os.path.join(tempfile.mkdtemp(), "synthetic.csv")
df.to_csv(path, index=False)

class FailingLLM(tabmem.LLM_Interface):
    chat_mode = True
    def chat_completion(self, messages, temperature, max_tokens):
        raise RuntimeError("simulated API failure")

tabmem.config.max_tokens = 1000
print("before:", tabmem.config.max_tokens)     # 1000
try:
    tabmem.first_token_test(path, FailingLLM())
except Exception as e:
    print("raised:", type(e).__name__)         # RuntimeError
print("after :", tabmem.config.max_tokens)     # 1   <-- leaked
```

Verified against 0.1.6: prints `before: 1000` / `after : 1`.

Any test run after this point returns empty responses and scores zero, with no
error raised.

**Note on intermittency.** The randomness pre-check
(`statistical_feature_prediction_test`) draws its train/test split with
`np.random.choice` and no fixed seed, so on a given dataset it may abort the
test on one run and proceed on the next. We observed exactly this: the same
corpus leaked in one session and returned early in another. That makes the bug
intermittent and correspondingly harder to attribute.

### Impact

Anyone whose run hits a transient API error during `first_token_test` — and then
continues, as `run_all_tests` and any multi-dataset loop will — gets zeros for
everything that follows. Because the tests report scores rather than raising,
the corruption is invisible unless raw responses are inspected.

We found it only by noticing that 50 API calls had produced 50 output tokens in
total.

### Suggested fix

Wrap the restore in `try/finally` so it runs on every path.

```diff
     #  set max_tokens to the number of digits (speedup)
-    prev_max_tokes = tabmem.config.max_tokens
-    tabmem.config.max_tokens = num_digits
-
-    # perform a row completion task
-    if llm.chat_mode:
-        _, test_suffixes, responses = row_chat_completion(
-            llm,
-            csv_file,
-            system_prompt,
-            num_prefix_rows,
-            num_queries,
-            few_shot,
-            out_file,
-            rng=rng,
-        )
-    else:
-        _, test_suffixes, responses = row_completion(
-            llm, csv_file, num_prefix_rows, num_queries, out_file
-        )
-
-    # reset max_tokens
-    tabmem.config.max_tokens = prev_max_tokes
+    prev_max_tokens = tabmem.config.max_tokens
+    tabmem.config.max_tokens = num_digits
+
+    # perform a row completion task
+    try:
+        if llm.chat_mode:
+            _, test_suffixes, responses = row_chat_completion(
+                llm,
+                csv_file,
+                system_prompt,
+                num_prefix_rows,
+                num_queries,
+                few_shot,
+                out_file,
+                rng=rng,
+            )
+        else:
+            _, test_suffixes, responses = row_completion(
+                llm, csv_file, num_prefix_rows, num_queries, out_file
+            )
+    finally:
+        # restore on every path: an exception here would otherwise leave the
+        # global cap at one token for the rest of the session
+        tabmem.config.max_tokens = prev_max_tokens
```

(The patch also corrects the spelling of `prev_max_tokes`.)

### Secondary issue found while writing the reproduction

`statistical_feature_prediction_test` calls `df.fillna(df.mean())` on a frame
that may contain string columns, which raises `TypeError` on pandas 2.x. Worth a
separate issue: it makes `first_token_test` fail on any dataset with an
unconverted object column. (The categorical conversion above it only covers
columns present in `feature_names`.)

### Optional hardening

A context manager would prevent this class of bug wherever global config is
adjusted temporarily:

```python
from contextlib import contextmanager

@contextmanager
def temporary_config(**overrides):
    previous = {k: tabmem.config[k] for k in overrides}
    tabmem.config.update(overrides)
    try:
        yield
    finally:
        tabmem.config.update(previous)
```

Used as:

```python
with temporary_config(max_tokens=num_digits):
    ...
```

A warning when a test returns entirely empty responses would also surface this
and similar failures at the point they occur rather than in the reported score.

### Environment

- `tabmemcheck` 0.1.6
- Python 3.12, Google Colab
- Custom `LLM_Interface` adapter for the Anthropic Messages API

---

## Why this is worth filing

Two reasons beyond good citizenship.

It is a real contribution to a tool used for evaluating LLM memorization, and a
merged fix is documentary evidence of contribution to the field — relevant if
you are assembling a record of impact.

And the paper cites this library for its §4.3 results. A reviewer who tries to
reproduce them against the unpatched version could hit the same corruption. A
filed issue you can reference in the artefact README protects the reproducibility
claim.

Link the issue from the artefact README and from the methodological note in §4.3.
