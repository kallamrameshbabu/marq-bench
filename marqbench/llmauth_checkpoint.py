"""
llmauth_checkpoint.py — Resumable step caching for MARQ-Bench notebooks.

Free Colab disconnects. Every expensive step therefore writes its output to
Drive and is skipped on re-run if that output already exists.

The contract, in one line: **to redo a step, delete its file and re-run the
cell.** Nothing else. No flags to remember, no hidden state.

    ckpt = Checkpoint("/content/drive/MyDrive/Paper2_RuleAuthorship/checkpoints")

    facts, cached = ckpt.step(
        "census_bank_marketing", "json",
        lambda: build_the_census(),          # only runs if not already saved
    )

Every save is paired with a sidecar `.meta.json` recording when it was written,
by which module version, and the SHA-256 of the content — so a stale artifact
from an older code version is identifiable rather than silently reused.

ANTI-TAUTOLOGY BOUNDARY
-----------------------
Infrastructure only. Imports nothing from the project.

Author: Ramesh Babu Kallam
License: MIT
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

__all__ = ["Checkpoint", "CheckpointInfo", "CHECKPOINT_VERSION"]

CHECKPOINT_VERSION = "1.0.0"
_KINDS = ("json", "jsonl", "parquet", "csv", "text", "pickle")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class CheckpointInfo:
    name: str
    kind: str
    path: str
    exists: bool
    size_bytes: int | None = None
    written_at_utc: str | None = None
    code_version: str | None = None
    content_sha256: str | None = None


class Checkpoint:
    """File-backed step cache rooted at a Drive directory."""

    _EXT = {"json": ".json", "jsonl": ".jsonl", "parquet": ".parquet",
            "csv": ".csv", "text": ".txt", "pickle": ".pkl"}

    def __init__(self, root: str | Path, *, verbose: bool = True):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose

    # ---------------------------------------------------------------- paths

    def path_for(self, name: str, kind: str) -> Path:
        if kind not in _KINDS:
            raise ValueError(f"unknown kind {kind!r}; expected one of {_KINDS}")
        if "/" in name or "\\" in name:
            raise ValueError(f"checkpoint name must not contain path separators: {name!r}")
        return self.root / f"{name}{self._EXT[kind]}"

    def _meta_path(self, name: str, kind: str) -> Path:
        return self.path_for(name, kind).with_suffix(
            self._EXT[kind] + ".meta.json"
        )

    def exists(self, name: str, kind: str) -> bool:
        return self.path_for(name, kind).exists()

    # ------------------------------------------------------------- main API

    def step(
        self,
        name: str,
        kind: str,
        build: Callable[[], Any],
        *,
        force: bool = False,
        code_version: str = "",
        save: Callable[[Any, Path], None] | None = None,
        load: Callable[[Path], Any] | None = None,
    ) -> tuple[Any, bool]:
        """Return (value, was_cached).

        Runs `build()` only when the artifact is absent or `force=True`.
        Custom `save`/`load` override the built-in handlers for that kind.
        """
        path = self.path_for(name, kind)

        if path.exists() and not force:
            value = (load or self._default_load(kind))(path)
            if self.verbose:
                meta = self._read_meta(name, kind)
                when = (meta or {}).get("written_at_utc", "unknown time")
                print(f"  [cached] {name}{self._EXT[kind]}  (written {when})")
            return value, True

        if self.verbose:
            print(f"  [build ] {name}{self._EXT[kind]} ...")
        value = build()
        (save or self._default_save(kind))(value, path)
        self._write_meta(name, kind, code_version)
        if self.verbose:
            print(f"  [saved ] {name}{self._EXT[kind]}  "
                  f"({path.stat().st_size:,} bytes)")
        return value, False

    def invalidate(self, name: str, kind: str) -> bool:
        """Delete an artifact and its metadata. Equivalent to deleting by hand."""
        removed = False
        for p in (self.path_for(name, kind), self._meta_path(name, kind)):
            if p.exists():
                p.unlink()
                removed = True
        if self.verbose and removed:
            print(f"  [deleted] {name}{self._EXT[kind]} — next run will rebuild")
        return removed

    def clear_all(self, *, confirm: str = "") -> int:
        """Delete every checkpoint. Requires confirm='yes, delete everything'."""
        if confirm != "yes, delete everything":
            raise RuntimeError(
                "refusing to clear checkpoints without explicit confirmation; "
                "pass confirm='yes, delete everything'"
            )
        n = sum(1 for _ in self.root.iterdir())
        shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        return n

    # ---------------------------------------------------------------- status

    def info(self, name: str, kind: str) -> CheckpointInfo:
        path = self.path_for(name, kind)
        meta = self._read_meta(name, kind) or {}
        return CheckpointInfo(
            name=name, kind=kind, path=str(path), exists=path.exists(),
            size_bytes=path.stat().st_size if path.exists() else None,
            written_at_utc=meta.get("written_at_utc"),
            code_version=meta.get("code_version"),
            content_sha256=meta.get("content_sha256"),
        )

    def status(self, expected: Iterable[tuple[str, str]] | None = None) -> None:
        """Print what is and is not checkpointed. Safe to call in any cell."""
        print(f"Checkpoints in {self.root}")
        if expected:
            for name, kind in expected:
                i = self.info(name, kind)
                mark = "OK  " if i.exists else "--  "
                size = f"{i.size_bytes:,} B" if i.size_bytes else "missing"
                print(f"  {mark}{name}{self._EXT[kind]:<10} {size}")
            return
        files = sorted(p for p in self.root.iterdir()
                       if p.is_file() and not p.name.endswith(".meta.json"))
        if not files:
            print("  (none yet)")
        for p in files:
            print(f"  OK  {p.name:<40} {p.stat().st_size:,} B")

    # -------------------------------------------------------------- internals

    def _write_meta(self, name: str, kind: str, code_version: str) -> None:
        path = self.path_for(name, kind)
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        self._meta_path(name, kind).write_text(json.dumps({
            "name": name, "kind": kind,
            "written_at_utc": _utc_now(),
            "code_version": code_version,
            "content_sha256": h.hexdigest(),
            "checkpoint_version": CHECKPOINT_VERSION,
        }, indent=2))

    def _read_meta(self, name: str, kind: str) -> dict[str, Any] | None:
        p = self._meta_path(name, kind)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _default_save(kind: str) -> Callable[[Any, Path], None]:
        def save_json(v, p):
            tmp = p.with_suffix(p.suffix + ".tmp")
            tmp.write_text(json.dumps(v, indent=2, sort_keys=True, default=str))
            os.replace(tmp, p)          # atomic: a killed session cannot
                                        # leave a half-written artifact that
                                        # would then be treated as complete

        def save_jsonl(v, p):
            tmp = p.with_suffix(p.suffix + ".tmp")
            with open(tmp, "w", encoding="utf-8") as fh:
                for row in v:
                    fh.write(json.dumps(row, default=str) + "\n")
            os.replace(tmp, p)

        def save_parquet(v, p):
            tmp = p.with_suffix(p.suffix + ".tmp")
            v.to_parquet(tmp, index=True)
            os.replace(tmp, p)

        def save_csv(v, p):
            tmp = p.with_suffix(p.suffix + ".tmp")
            v.to_csv(tmp, index=True)
            os.replace(tmp, p)

        def save_text(v, p):
            tmp = p.with_suffix(p.suffix + ".tmp")
            tmp.write_text(str(v))
            os.replace(tmp, p)

        def save_pickle(v, p):
            import pickle
            tmp = p.with_suffix(p.suffix + ".tmp")
            with open(tmp, "wb") as fh:
                pickle.dump(v, fh)
            os.replace(tmp, p)

        return {"json": save_json, "jsonl": save_jsonl, "parquet": save_parquet,
                "csv": save_csv, "text": save_text, "pickle": save_pickle}[kind]

    @staticmethod
    def _default_load(kind: str) -> Callable[[Path], Any]:
        def load_json(p):
            return json.loads(p.read_text())

        def load_jsonl(p):
            return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]

        def load_parquet(p):
            import pandas as pd
            return pd.read_parquet(p)

        def load_csv(p):
            import pandas as pd
            return pd.read_csv(p, index_col=0)

        def load_text(p):
            return p.read_text()

        def load_pickle(p):
            import pickle
            with open(p, "rb") as fh:
                return pickle.load(fh)

        return {"json": load_json, "jsonl": load_jsonl, "parquet": load_parquet,
                "csv": load_csv, "text": load_text, "pickle": load_pickle}[kind]
