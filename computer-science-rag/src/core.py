"""Shared primitives for the educational RAG system.

This module is deliberately side-effect free.  In particular, importing it
never loads ``.env`` and can therefore never activate paid APIs during tests,
benchmark validation, or source inspection.  Executable entry points must call
``load_runtime_environment`` explicitly.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "Data"
PROCESSED_ROOT = ROOT / "data_processed"
BUILD_SCHEMA_VERSION = "2.0"
TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_+/#.-]*")

AUTHORITY = {
    "MARK_SCHEME": 6,
    "SYLLABUS": 5,
    "TEXTBOOK": 4,
    "EXAMINER_REPORT": 4,
    "QUESTION_PAPER": 3,
}


def load_runtime_environment(path: Path | None = None, *, override: bool = False) -> bool:
    """Load runtime credentials only from an explicit executable entry point.

    Returning a boolean keeps callers testable and avoids logging any secret
    value.  Library imports intentionally never call this function.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return False
    return bool(load_dotenv(path or ROOT / ".env", override=override))


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(payload: Any, length: int = 16) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:length]


def source_fingerprint(data_root: Path = DATA_ROOT) -> tuple[str, list[dict[str, Any]]]:
    """Return a content fingerprint and auditable inventory for all source PDFs."""
    inventory = [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(data_root.rglob("*.pdf"))
    ]
    return stable_hash({"schema": BUILD_SCHEMA_VERSION, "files": inventory}, 20), inventory


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def tokens(text: str) -> list[str]:
    return [value.lower() for value in TOKEN_RE.findall(text or "")]


def token_count(text: str) -> int:
    # A deterministic lexical count is used for chunk boundaries.  Provider
    # usage returned by the API remains authoritative for billing.
    return len(tokens(text))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def pending_build_path() -> Path:
    pointer = PROCESSED_ROOT / "pending.json"
    if not pointer.exists():
        raise RuntimeError("No pending corpus build exists. Run scripts/build_corpus.py first.")
    payload = json.loads(pointer.read_text(encoding="utf-8"))
    path = (PROCESSED_ROOT / "builds" / payload["build_id"]).resolve()
    if path.parent != (PROCESSED_ROOT / "builds").resolve() or not path.exists():
        raise RuntimeError("Pending build pointer is invalid or incomplete.")
    return path


def current_build_path(*, require_index: bool = True) -> Path:
    """Resolve the atomically promoted build and validate its readiness."""
    pointer = PROCESSED_ROOT / "current.json"
    if not pointer.exists():
        raise RuntimeError("No active RAG build exists. Build and promote the indexes first.")
    payload = json.loads(pointer.read_text(encoding="utf-8"))
    path = (PROCESSED_ROOT / "builds" / payload["build_id"]).resolve()
    if path.parent != (PROCESSED_ROOT / "builds").resolve() or not path.exists():
        raise RuntimeError("Active build pointer does not resolve to a valid build directory.")
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    if manifest.get("schema_version") != BUILD_SCHEMA_VERSION:
        raise RuntimeError("Active build schema is incompatible; rebuild the corpus and indexes.")
    if require_index and manifest.get("status") != "READY":
        raise RuntimeError("Active build is not READY; a complete index rebuild is required.")
    return path
