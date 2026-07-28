"""Validate that every selectable syllabus topic can build a mock-test plan.

This offline production-readiness check does not call an LLM or persist an
assessment. It verifies active-build textbook coverage, bounded context
assembly, and all eight deterministic source assignments.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.module2_generation.assessment_engine import (
    _plan,
    _question_source_assignments,
    _source_catalog,
    retrieve_assessment_evidence,
)
from backend.module2_generation.assessments.book_catalog import BOOK_CATALOG
from backend.module2_generation.assessments.syllabus_catalog import resolve_topics


def validate_catalog() -> dict:
    failures: list[dict] = []
    summaries: dict[str, dict] = {}
    for level, chapters in BOOK_CATALOG.items():
        checked = 0
        maximum_context = 0
        without_topical_paper = 0
        for chapter in chapters:
            for topic in chapter["topics"]:
                scope = resolve_topics(level, [chapter["id"]], [topic["id"]])
                evidence = retrieve_assessment_evidence(
                    topic["name"],
                    "mock_test",
                    level,
                    scope["topics"],
                )
                plan = _plan(
                    topic["name"],
                    "medium",
                    "mock_test",
                    8,
                    "mixed",
                    scope["topics"],
                    25,
                    scope["allows_code"],
                    scope.get("code_kind"),
                )
                catalog, _ = _source_catalog(evidence["chunks"])
                assignments = _question_source_assignments(plan, catalog)
                checked += 1
                maximum_context = max(maximum_context, len(evidence["chunks"]))
                without_topical_paper += bool(evidence["missing_past_paper_topic_ids"])
                if evidence["missing_selected_topic_ids"] or len(assignments) != 8:
                    failures.append({
                        "level": level,
                        "chapter": chapter["id"],
                        "topic": topic["id"],
                        "missing_textbook": evidence["missing_selected_topic_ids"],
                        "style_chunks": evidence["past_paper_style_chunk_count"],
                        "assignments": len(assignments),
                    })
        summaries[level] = {
            "topics_checked": checked,
            "maximum_context_chunks": maximum_context,
            "topics_without_topical_paper": without_topical_paper,
        }
    return {"summaries": summaries, "failures": failures}


if __name__ == "__main__":
    report = validate_catalog()
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(1 if report["failures"] else 0)
