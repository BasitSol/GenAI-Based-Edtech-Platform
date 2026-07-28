"""Human-agreement evaluation for the Phase 2 typed-submission grading MVP."""
from __future__ import annotations

import csv
from pathlib import Path

from backend.module2_generation.grading.agreement import score_agreement


REQUIRED_FIELDS = {"ai_score", "human_score"}


def evaluate_grading_agreement(path: Path) -> dict:
    """Evaluate human-reviewed grades without calling a model or reading PII."""
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not REQUIRED_FIELDS.issubset(set(rows[0])):
        raise ValueError("CSV must contain ai_score and human_score columns.")
    parsed = []
    invalid_rows = []
    for index, row in enumerate(rows, 2):
        try:
            parsed.append({"ai_score": float(row["ai_score"]), "human_score": float(row["human_score"])})
        except (TypeError, ValueError):
            invalid_rows.append(index)
    return {"input_path": str(path), "total_rows": len(rows), "valid_rows": len(parsed),
            "invalid_rows": invalid_rows, "agreement": score_agreement(parsed),
            "interpretation": "Use at least 15-20 independently teacher-reviewed typed submissions before making a grading-quality claim."}
