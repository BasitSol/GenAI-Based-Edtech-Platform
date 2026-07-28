"""Public boundary for exact 25-mark syllabus-scoped mock tests.

Mock tests intentionally have a stricter contract than ordinary quizzes:
selected topics are mandatory, the total is fixed at 25 marks, and the shared
engine receives the resolved syllabus scope rather than free-form UI labels.
"""
from __future__ import annotations

from typing import Any

from backend.module2_generation.assessment_engine import generate_assessment


def generate_mock_test(*, topic_names: list[str], difficulty: str, level: str,
                       selected_topics: list[dict[str, Any]], allows_code: bool,
                       code_kind: str | None = None) -> dict[str, Any]:
    """Generate an eight-question, exact 25-mark teacher-review draft."""
    if not topic_names or not selected_topics:
        raise ValueError("At least one resolved syllabus topic is required.")
    return generate_assessment(
        topic="; ".join(topic_names),
        difficulty=difficulty,
        assessment_type="mock_test",
        question_count=8,
        question_format="mixed",
        level=level,
        selected_topics=selected_topics,
        total_marks=25,
        allows_code=allows_code,
        code_kind=code_kind,
    )
