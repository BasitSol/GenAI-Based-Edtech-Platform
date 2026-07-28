"""Public generation boundary for quizzes and teacher assignments.

The shared assessment engine owns retrieval, provenance, validation, and
teacher-review safety. This module owns the Phase 2 quiz/assignment contract,
preventing API code from depending on the engine's internal implementation.
"""
from __future__ import annotations

from typing import Any

from backend.module2_generation.assessment_engine import generate_assessment


def generate_quiz(*, topic: str, difficulty: str, assessment_type: str,
                  question_count: int, question_format: str = "mixed",
                  level: str | None = None) -> dict[str, Any]:
    """Generate a grounded quiz or assignment for teacher review."""
    if assessment_type not in {"quiz", "assignment"}:
        raise ValueError("Quiz generation supports only quiz or assignment types.")
    return generate_assessment(
        topic=topic,
        difficulty=difficulty,
        assessment_type=assessment_type,
        question_count=question_count,
        question_format=question_format,
        level=level,
    )
