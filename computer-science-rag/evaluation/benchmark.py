"""Strict benchmark schema and corruption checks.

Only human-reviewed records and deterministic question/mark-scheme pairs are
measurable.  Drafts remain visible in reports but cannot inflate metrics.
"""
from __future__ import annotations

import json
import re
from pathlib import Path


MEASURABLE_STATUSES = {"REVIEWED", "AUTO_VALIDATED_EXACT", "AUTO_VALIDATED_CURRICULUM"}
VALID_STATUSES = MEASURABLE_STATUSES | {"DRAFT", "REJECTED"}
REQUIRED = {
    "id", "question", "category", "difficulty", "expected_intent",
    "gold_answer", "ground_truth_references", "expected_pages",
    "expected_topics", "expected_citations", "review_status",
}
MOJIBAKE = re.compile(r"(?:Ã.|â€|â€™|ï¬|�)")


def _validate(row: dict, path: Path, line: int) -> None:
    missing = REQUIRED - set(row)
    if missing:
        raise ValueError(f"{path}:{line} missing required fields {sorted(missing)}")
    if row["review_status"] not in VALID_STATUSES:
        raise ValueError(f"{path}:{line} has invalid review_status {row['review_status']!r}")
    if not str(row["question"]).strip() or not str(row["gold_answer"]).strip():
        raise ValueError(f"{path}:{line} has an empty question or gold answer")
    if MOJIBAKE.search(row["question"] + row["gold_answer"]):
        raise ValueError(f"{path}:{line} contains corrupted/undecoded text")
    if row["review_status"] in MEASURABLE_STATUSES:
        if not row["ground_truth_references"] or not row["expected_citations"]:
            raise ValueError(f"{path}:{line} measurable record has no ground-truth references")
        for source in row["ground_truth_references"]:
            if not source.get("document_id") or not source.get("chunk_id") or source.get("page") is None:
                raise ValueError(f"{path}:{line} has an incomplete ground-truth identity")
    if row["review_status"] == "AUTO_VALIDATED_EXACT":
        if row.get("generation_provenance") != "exact_question_mark_scheme_pair":
            raise ValueError(f"{path}:{line} auto-validated row lacks exact-pair provenance")
        if not row.get("question_chunk_id") or not row.get("answer_chunk_id"):
            raise ValueError(f"{path}:{line} auto-validated row lacks paired chunk identities")
    if row["review_status"] == "AUTO_VALIDATED_CURRICULUM":
        if row.get("generation_provenance") not in {"curated_question_source_validated", "source_derived_question"}:
            raise ValueError(f"{path}:{line} curriculum row lacks source-validation provenance")
        if not row.get("source_validation_terms"):
            raise ValueError(f"{path}:{line} curriculum row lacks validation terms")


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Benchmark does not exist: {path}. Generate it after building the fresh index.")
    rows, identifiers = [], set()
    for line, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        row = json.loads(raw)
        _validate(row, path, line)
        if row["id"] in identifiers:
            raise ValueError(f"{path}:{line} duplicate benchmark id {row['id']!r}")
        identifiers.add(row["id"])
        rows.append(row)
    return rows


def measurable_records(path: Path) -> list[dict]:
    return [row for row in load_records(path) if row["review_status"] in MEASURABLE_STATUSES]


# Compatibility name retained for downstream callers, with stricter meaning.
approved_records = measurable_records


def benchmark_status(path: Path) -> dict:
    records = load_records(path)
    counts = {status: 0 for status in sorted(VALID_STATUSES)}
    for row in records:
        counts[row["review_status"]] += 1
    measurable = sum(counts[status] for status in MEASURABLE_STATUSES)
    categories: dict[str, int] = {}
    for row in records:
        categories[row["category"]] = categories.get(row["category"], 0) + 1
    return {
        "total": len(records), "by_review_status": counts,
        "measurable_count": measurable, "minimum_required": 100,
        "minimum_satisfied": measurable >= 100, "category_distribution": categories,
    }


def audit_records(records: list[dict]) -> dict:
    categories, levels = {}, {}
    for row in records:
        categories[row["category"]] = categories.get(row["category"], 0) + 1
        levels[row.get("level", "UNKNOWN")] = levels.get(row.get("level", "UNKNOWN"), 0) + 1
    return {"total": len(records), "levels": levels, "categories": categories,
            "measurable_count": sum(row["review_status"] in MEASURABLE_STATUSES for row in records)}
