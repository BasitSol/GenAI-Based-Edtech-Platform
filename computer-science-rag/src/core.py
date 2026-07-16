from __future__ import annotations

import csv, hashlib, json, re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    # The system remains usable in retrieval-only mode without optional generation.
    pass
TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_+/#.-]*")

def load_yaml(path: Path) -> dict[str, Any]:
    import yaml
    with path.open(encoding="utf-8") as handle: return yaml.safe_load(handle) or {}

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()

def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")

def tokens(text: str) -> list[str]: return [x.lower() for x in TOKEN_RE.findall(text)]
def token_count(text: str) -> int: return len(tokens(text))
def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows: handle.write(json.dumps(row, ensure_ascii=False) + "\n")
def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists(): return []
    with path.open(encoding="utf-8") as handle: return [json.loads(line) for line in handle if line.strip()]
def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)

AUTHORITY = {"SYLLABUS": 5, "MARK_SCHEME": 5, "TEXTBOOK": 4, "QUESTION_PAPER": 4, "EXAMINER_REPORT": 4, "MARKING_PATTERN": 2}
