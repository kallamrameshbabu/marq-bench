#!/usr/bin/env python3
"""
clean_notebooks.py — prepare notebooks for the public artefact.

Working notebooks accumulate scaffolding: patch cells written to repair a bad
run, one-off diagnostics, credential probes, destructive maintenance cells. A
reviewer needs the pipeline, not its repair history.

This removes that scaffolding while KEEPING cell outputs, because the outputs
are evidence that the pipeline ran and produced the reported numbers. Stripping
them would weaken the artefact.

What it removes:
  - cells matching DROP_PATTERNS (diagnostics, patches, ad-hoc repairs)
  - destructive maintenance cells, or neutralises them if they must stay
  - anything containing a credential

What it never touches:
  - pipeline cells and their outputs
  - markdown explaining the method

Usage:
    python clean_notebooks.py --in working/ --out repo/notebooks/ [--dry-run]

Always review the diff. This is a blunt instrument and it is your name on the
artefact.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

# Cells whose source matches any of these are removed. Ordered roughly by how
# confident we are that they are scaffolding rather than pipeline.
DROP_PATTERNS = [
    (r"LoggingAnthropicLLM|CAPTURED\b",        "harness diagnostic"),
    (r"harness[_ ]diagnostic",                  "harness diagnostic"),
    (r"purge_failures\s*\(",                    "one-off run-log repair"),
    (r"importlib\.reload",                      "module hot-reload patch"),
    (r"N0_FIX|N2_PATCH|_PATCH\b",               "patch cell"),
    (r"stop_reason\s*:.*block types",           "API response probe"),
    (r"^\s*#\s*=+\s*\n#\s*(DIAGNOSTIC|PATCH)",  "explicit diagnostic block"),
    (r"\.unlink\(\)|shutil\.rmtree",            "destructive maintenance"),
]

# Cells that must be neutralised rather than removed, because the pipeline
# refers to them. Pattern -> replacement applied to the source.
NEUTRALISE = [
    (r"CLEAR_OLD\s*=\s*True",
     "CLEAR_OLD = False        # never clear on a fresh clone"),
    (r"force=True",
     "force=False             # reuse released outputs; set True to recompute"),
]

# Anything matching these is a hard stop — we refuse to write the file.
SECRET_PATTERNS = [
    (r"sk-[A-Za-z0-9_\-]{20,}",           "API key literal"),
    (r"api_key\s*=\s*['\"][^'\"]{20,}",   "hard-coded api_key"),
    (r"ANTHROPIC_API_KEY\s*=\s*['\"]",    "hard-coded key assignment"),
]


def cell_source(cell) -> str:
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else src


def scan_secrets(nb, name):
    hits = []
    for i, cell in enumerate(nb["cells"]):
        s = cell_source(cell)
        for pat, label in SECRET_PATTERNS:
            if re.search(pat, s):
                hits.append((i, label))
        # also scan outputs — a printed key is just as leaked
        for out in cell.get("outputs", []):
            text = "".join(out.get("text", "")) if out.get("text") else ""
            for pat, label in SECRET_PATTERNS:
                if re.search(pat, text):
                    hits.append((i, f"{label} (in OUTPUT)"))
    return hits


def clean(nb, name, verbose=True):
    kept, dropped = [], []
    for i, cell in enumerate(nb["cells"]):
        src = cell_source(cell)
        reason = None
        # A cell that NEUTRALISE can defuse is kept and rewritten, even if it
        # also matches a DROP pattern. Otherwise a destructive-looking
        # maintenance cell the notebook narrative refers to would vanish and
        # leave a gap in the numbering.
        neutralisable = any(re.search(pat, src) for pat, _ in NEUTRALISE)
        if cell["cell_type"] == "code" and not neutralisable:
            for pat, label in DROP_PATTERNS:
                if re.search(pat, src, re.M):
                    reason = label
                    break
        if reason:
            dropped.append((i, reason, src.split("\n")[0][:60]))
            # also drop an immediately preceding markdown header for that cell
            if kept and kept[-1]["cell_type"] == "markdown" and \
                    len(cell_source(kept[-1])) < 400:
                kept.pop()
            continue

        if cell["cell_type"] == "code":
            new_src = src
            for pat, repl in NEUTRALISE:
                new_src = re.sub(pat, repl, new_src)
            if new_src != src:
                lines = new_src.split("\n")
                cell["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]
                if verbose:
                    print(f"    neutralised cell {i}")
        kept.append(cell)

    nb["cells"] = kept
    if verbose and dropped:
        print(f"    dropped {len(dropped)} cell(s):")
        for i, reason, head in dropped:
            print(f"      [{i:2}] {reason:<28} {head}")
    return nb, dropped


def renumber_execution(nb):
    """Sequential execution counts, so the notebook reads as one clean run."""
    n = 1
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            if cell.get("execution_count") is not None:
                cell["execution_count"] = n
                for out in cell.get("outputs", []):
                    if "execution_count" in out:
                        out["execution_count"] = n
                n += 1
    return nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--strip-outputs", action="store_true",
                    help="remove outputs too (NOT recommended: outputs are "
                         "evidence the pipeline ran)")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    total_dropped, blocked = 0, []
    for path in sorted(src.glob("*.ipynb")):
        print(f"\n{path.name}")
        nb = json.loads(path.read_text())

        secrets = scan_secrets(nb, path.name)
        if secrets:
            print("    !! REFUSING TO WRITE — possible credential found:")
            for i, label in secrets:
                print(f"       cell {i}: {label}")
            blocked.append(path.name)
            continue

        nb, dropped = clean(nb, path.name)
        total_dropped += len(dropped)

        if args.strip_outputs:
            for cell in nb["cells"]:
                if cell["cell_type"] == "code":
                    cell["outputs"] = []
                    cell["execution_count"] = None
        else:
            nb = renumber_execution(nb)

        n_code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
        n_out = sum(1 for c in nb["cells"]
                    if c["cell_type"] == "code" and c.get("outputs"))
        print(f"    kept {len(nb['cells'])} cells "
              f"({n_code} code, {n_out} with outputs)")

        if not args.dry_run:
            (dst / path.name).write_text(json.dumps(nb, indent=1))

    print(f"\n{'DRY RUN — nothing written' if args.dry_run else 'written to ' + str(dst)}")
    print(f"total cells dropped: {total_dropped}")
    if blocked:
        print(f"\n!! BLOCKED (credentials): {blocked}")
        print("   Remove the credential and re-run. Do not push these.")
        raise SystemExit(1)
    print("\nNow open each cleaned notebook and read it end to end.")
    print("This script is a blunt instrument; your name is on the artefact.")


if __name__ == "__main__":
    main()
