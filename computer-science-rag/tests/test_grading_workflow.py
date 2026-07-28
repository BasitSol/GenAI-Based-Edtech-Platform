"""Offline tests for typed answer extraction, review persistence, and agreement."""
from __future__ import annotations

from backend.module2_generation.grading.agreement import score_agreement
from evaluation.grading_eval import evaluate_grading_agreement
from backend.shared.persistence import PlatformStore
from backend.module2_generation.grading.agent import extract_answers, retrieve_grading_evidence


def test_extract_answers_segments_numbered_typed_submission():
    answers = extract_answers("Q1: First answer\nQuestion 2: Second answer", [1, 2])
    assert answers == {1: "First answer", 2: "Second answer"}


def test_extract_answers_uses_single_answer_fallback():
    assert extract_answers("One unlabelled response", [3]) == {3: "One unlabelled response"}


def test_grading_evidence_is_empty_without_provider_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    evidence = retrieve_grading_evidence([{"question": "Explain binary search."}], "O_LEVEL")
    assert evidence == [[]]


def test_teacher_review_preserves_ai_score_and_enforces_maximum(tmp_path):
    store = PlatformStore(tmp_path / "platform.sqlite")
    teacher = store.create_user("teacher@example.com", "hash", "teacher")
    student = store.create_user("student@example.com", "hash", "student")
    assessment = store.create_assessment(teacher["id"], "SQL", "BEGINNER", "quiz", {"questions": []}, "approved")
    submission = store.create_submission(assessment["id"], student["id"], "SELECT * FROM Student")
    grade = store.create_grade(submission["id"], 3, 4, "Good start", 0.8, {"items": []})
    assert store.review_grade(grade["id"], teacher["id"], 5) is None
    reviewed = store.review_grade(grade["id"], teacher["id"], 4, "Teacher confirmed")
    assert reviewed["ai_score"] == 3 and reviewed["human_score"] == 4


def test_score_agreement_reports_mae_and_safe_correlation():
    result = score_agreement([{"ai_score": 3, "human_score": 4}, {"ai_score": 5, "human_score": 5}])
    assert result["count"] == 2 and result["mae"] == 0.5 and result["pearson_correlation"] == 1.0


def test_grading_evaluation_reads_anonymised_score_csv(tmp_path):
    path = tmp_path / "agreement.csv"
    path.write_text("ai_score,human_score\n3,4\n5,5\n", encoding="utf-8")
    result = evaluate_grading_agreement(path)
    assert result["valid_rows"] == 2 and result["agreement"]["mae"] == 0.5
