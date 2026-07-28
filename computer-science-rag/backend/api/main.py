"""HTTP boundary for the educational RAG and Phase 2 platform capabilities.

Phase 1 remains available through its original ``/ask`` and ``/retrieve``
contracts.  Phase 2 adds an authenticated ``/chat`` wrapper and role-aware
platform endpoints without coupling any RAG implementation to FastAPI.
"""
from __future__ import annotations

import csv
import logging
import os
import sqlite3
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from backend.api.schemas import (AskRequest, AssessmentCreateRequest, ChatRequest, LoginRequest, MockTestCreateRequest,
                             GradeReviewRequest, RegisterRequest, SubmissionCreateRequest, TokenResponse, UserResponse)
from backend.api.security import (create_access_token, current_user, get_store, hash_password,
                              require_roles, verify_password)
from backend.module2_generation.assessments import export_assessment
from backend.module2_generation.assessments.syllabus_catalog import chapters_for_level, resolve_topics
from backend.shared.core import ROOT, current_build_path
from backend.module1_rag.chat.answer_service import answer_question
from backend.module1_rag.monitoring.telemetry import TelemetryStore
from backend.module1_rag.monitoring.tracing import langsmith_status
from backend.module1_rag.monitoring.live_excel import append_live_answer, live_workbook_path
from backend.shared.persistence import PlatformStore
from backend.module1_rag.chat.workflow import retrieve as retrieve_workflow
from backend.module2_generation.grading.agent import grade_typed_submission
from backend.module2_generation.mock_test import generate_mock_test
from backend.module2_generation.quiz import generate_quiz


app = FastAPI(
    title="GenAI Smart Education Platform",
    version="2.0.0",
    description="Phase 1 educational RAG plus the Phase 2 authenticated platform foundation.",
)
LOGGER = logging.getLogger(__name__)

# The React development server and FastAPI run on separate local origins.
# Keep the allow-list explicit instead of enabling wildcard origins with
# credentials, and permit deployments to override it through configuration.
_default_frontend_origins = "http://127.0.0.1:5173,http://localhost:5173"
_frontend_origins = [
    origin.strip()
    for origin in os.getenv("FRONTEND_ORIGINS", _default_frontend_origins).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_frontend_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


def _record_live_answer(result: dict, question: str) -> None:
    """Log UI/API answers without allowing a locked workbook to break chat."""
    try:
        append_live_answer({**result, "question": question})
    except OSError:
        LOGGER.warning("Live evaluation workbook could not be updated", exc_info=True)


@app.get("/health")
def health() -> dict:
    """Report build readiness without loading paid providers or credentials."""
    try:
        build = current_build_path()
        return {"status": "ready", "build_id": build.name}
    except RuntimeError as exc:
        return {"status": "not_ready", "reason": str(exc)}


@app.post("/ask")
def ask(request: AskRequest) -> dict:
    """Retained Phase 1 compatibility endpoint for existing clients/tests."""
    response = answer_question(**request.model_dump())
    _record_live_answer(response, request.query)
    return response


@app.post("/chat")
def chat(request: ChatRequest, user: dict = Depends(require_roles("teacher", "student"))) -> dict:
    """Run the unchanged RAG workflow in an authenticated platform session."""
    response = answer_question(**request.model_dump())
    _record_live_answer(response, request.query)
    response["platform_user"] = {"id": user["id"], "role": user["role"]}
    return response


@app.post("/retrieve")
def retrieve(request: AskRequest) -> dict:
    """Expose retrieval diagnostics independently from answer generation."""
    return retrieve_workflow(request.query, request.level, request.exam_year)


@app.get("/monitoring/summary")
def monitoring_summary(days: int = Query(default=30, ge=1, le=90)) -> dict:
    """Expose Module 1 operational metrics through the HTTP boundary."""
    return {"local_telemetry": TelemetryStore().summary(days), "langsmith": langsmith_status()}


@app.get("/monitoring/live-workbook")
def download_live_workbook() -> FileResponse:
    """Download the append-only question/answer evaluation workbook."""
    path = live_workbook_path()
    if not path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail="No live answers have been recorded yet.")
    return FileResponse(
        path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename="live_answers.xlsx",
    )


@app.post("/auth/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register(request: RegisterRequest, store: PlatformStore = Depends(get_store)) -> dict:
    """Create a local teacher or student account with a bcrypt password hash."""
    try:
        return store.create_user(str(request.email), hash_password(request.password), request.role)
    except sqlite3.IntegrityError:
        # Do not return storage internals or a password-related distinction.
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="An account with this email already exists.") from None


@app.post("/auth/login", response_model=TokenResponse)
def login(request: LoginRequest, store: PlatformStore = Depends(get_store)) -> dict:
    """Issue a short-lived signed JWT after verifying local credentials."""
    user = store.find_user_by_email(str(request.email))
    if not user or not verify_password(request.password, user["password_hash"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password.")
    token, expires_in = create_access_token(user)
    return {"access_token": token, "token_type": "bearer", "expires_in": expires_in}


@app.get("/auth/me", response_model=UserResponse)
def me(user: dict = Depends(current_user)) -> dict:
    """Return the authenticated user's safe public profile."""
    return user


@app.get("/dashboard/teacher")
def teacher_dashboard(user: dict = Depends(require_roles("teacher")),
                      store: PlatformStore = Depends(get_store)) -> dict:
    """Minimal teacher dashboard; Stage 4 will add assessment/review lists."""
    return {"role": user["role"], "summary": store.role_counts()}


@app.get("/dashboard/student")
def student_dashboard(user: dict = Depends(require_roles("student")),
                      store: PlatformStore = Depends(get_store)) -> dict:
    """Minimal student dashboard; Stage 4 will add approved work and grades."""
    return {"role": user["role"], "summary": store.role_counts()}


@app.post("/assessments", status_code=status.HTTP_201_CREATED)
def create_assessment(request: AssessmentCreateRequest, user: dict = Depends(require_roles("teacher")),
                      store: PlatformStore = Depends(get_store)) -> dict:
    """Generate a grounded draft and persist it as pending teacher review."""
    try:
        generated = generate_quiz(**request.model_dump())
    except Exception as exc:
        LOGGER.exception("Assessment generation pipeline failed")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail={
            "message": "Assessment generation failed before validation.",
            "error_category": type(exc).__name__,
            "action": "Check the API terminal traceback; no draft was saved.",
        }) from exc
    if not generated["validation"]["passed"]:
        content = generated.get("content", {})
        validation = generated.get("validation", {})
        insufficient = content.get("status") == "INSUFFICIENT_EVIDENCE"
        http_status = status.HTTP_422_UNPROCESSABLE_ENTITY if insufficient else status.HTTP_503_SERVICE_UNAVAILABLE
        LOGGER.warning(
            "Assessment generation rejected trace_id=%s content_status=%s validation_reason=%s",
            generated.get("trace_id"), content.get("status"), validation.get("reason"),
        )
        raise HTTPException(status_code=http_status, detail={
            "message": (
                "No usable textbook evidence was found for the requested assessment."
                if insufficient
                else "Assessment generation did not pass the production quality gates. No draft was saved."
            ),
            "error_code": content.get("error_category") or validation.get("error_category")
                          or validation.get("reason") or "ASSESSMENT_VALIDATION_FAILED",
            "retryable": not insufficient,
            "trace_id": generated.get("trace_id"),
            "content_status": content.get("status"),
            "content_reason": content.get("reason"),
            "validation": validation,
            "retry_count": generated.get("retry_count", 0),
        })
    blueprint, content = generated["blueprint"], generated["content"]
    return store.create_assessment(user["id"], blueprint["topic"], blueprint["difficulty"],
                                   blueprint["assessment_type"], content, status="pending_review")


@app.get("/syllabus/{level}/chapters")
def syllabus_chapters(level: str, user: dict = Depends(require_roles("teacher"))) -> dict:
    """Expose the active-build syllabus projection for the mock-test selector."""
    try:
        return chapters_for_level(level)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from None


@app.post("/mock-tests", status_code=status.HTTP_201_CREATED)
def create_mock_test(request: MockTestCreateRequest, user: dict = Depends(require_roles("teacher")),
                     store: PlatformStore = Depends(get_store)) -> dict:
    """Generate a 25-mark, selected-syllabus-topic mock test for teacher review.

    This endpoint intentionally cannot create a full paper. Its immutable
    25-mark and topic-scope contract is enforced again inside the LangGraph
    workflow before a draft can be persisted.
    """
    try:
        scope = resolve_topics(request.level, request.chapter_ids, request.topic_ids)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from None
    topic_names = [item["name"] for item in scope["topics"]]
    try:
        generated = generate_mock_test(
            topic_names=topic_names, difficulty=request.difficulty, level=scope["level"],
            selected_topics=scope["topics"], allows_code=bool(scope["allows_code"]),
            code_kind=scope.get("code_kind"),
        )
    except Exception as exc:
        # Keep credentials/provider bodies private while making a server-side
        # fault distinguishable from a normal evidence/quality rejection.
        LOGGER.exception("Mock-test generation pipeline failed")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail={
            "message": "Mock-test generation pipeline failed before validation.",
            "error_category": type(exc).__name__,
            "action": "Check the API terminal traceback; no draft was saved.",
        }) from exc
    if not generated["validation"]["passed"]:
        content = generated.get("content", {})
        validation = generated.get("validation", {})
        insufficient = content.get("status") == "INSUFFICIENT_EVIDENCE"
        http_status = status.HTTP_422_UNPROCESSABLE_ENTITY if insufficient else status.HTTP_503_SERVICE_UNAVAILABLE
        message = (
            "The selected topic has no usable textbook evidence in the active corpus."
            if insufficient
            else "Mock-test generation did not pass the production quality gates. No draft was saved."
        )
        LOGGER.warning(
            "Mock-test generation rejected trace_id=%s content_status=%s validation_reason=%s",
            generated.get("trace_id"), content.get("status"), validation.get("reason"),
        )
        raise HTTPException(status_code=http_status, detail={
            "message": message,
            "error_code": content.get("error_category") or validation.get("error_category")
                          or validation.get("reason") or "MOCK_TEST_VALIDATION_FAILED",
            "retryable": not insufficient,
            "trace_id": generated.get("trace_id"),
            "content_status": content.get("status"),
            "content_reason": content.get("reason"),
            "validation": validation,
            "blueprint": generated.get("blueprint"),
            "retry_count": generated.get("retry_count", 0),
        })
    blueprint, content = generated["blueprint"], generated["content"]
    content["mock_test_scope"] = {key: scope[key] for key in ("level", "source_documents", "chapter_ids", "chapter_names", "topic_ids", "allows_code", "code_kind")}
    return store.create_assessment(user["id"], blueprint["topic"], blueprint["difficulty"],
                                   "mock_test", content, status="pending_review")


@app.post("/assessment/{assessment_id}/approve")
def approve_assessment(assessment_id: int, user: dict = Depends(require_roles("teacher")),
                       store: PlatformStore = Depends(get_store)) -> dict:
    """Publish a reviewed assessment only when its owning teacher approves it."""
    assessment = store.approve_assessment(assessment_id, user["id"])
    if not assessment:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Assessment not found or not owned by this teacher.")
    return assessment


@app.delete("/assessment/{assessment_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_draft_assessment(assessment_id: int, user: dict = Depends(require_roles("teacher")),
                            store: PlatformStore = Depends(get_store)) -> None:
    """Remove only a teacher-owned unapproved draft with no student work."""
    if not store.delete_draft_assessment(assessment_id, user["id"]):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail="Only an unapproved draft with no submissions can be deleted.")


@app.get("/assessment/{assessment_id}/export/{output_format}")
def export_teacher_assessment(assessment_id: int, output_format: str, include_solutions: bool = True,
                              user: dict = Depends(require_roles("teacher")),
                              store: PlatformStore = Depends(get_store)) -> FileResponse:
    """Export only a teacher's own assessment; output never invokes the model."""
    if output_format not in {"docx", "pdf"}:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Output format must be docx or pdf.")
    assessment = store.get_assessment(assessment_id)
    if not assessment or assessment["teacher_id"] != user["id"]:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Assessment not found or not owned by this teacher.")
    try:
        path = export_assessment(assessment, output_format, include_solutions)  # type: ignore[arg-type]
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from None
    media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document" if output_format == "docx" else "application/pdf"
    return FileResponse(path, media_type=media_type, filename=path.name)


@app.get("/assessments/mine")
def teacher_assessments(user: dict = Depends(require_roles("teacher")),
                        store: PlatformStore = Depends(get_store)) -> list[dict]:
    return store.list_assessments(teacher_id=user["id"])


@app.get("/assessments/available")
def available_assessments(user: dict = Depends(require_roles("student")),
                          store: PlatformStore = Depends(get_store)) -> list[dict]:
    return store.list_assessments(approved_only=True)


@app.post("/assessment/{assessment_id}/submissions", status_code=status.HTTP_201_CREATED)
def submit_typed_work(assessment_id: int, request: SubmissionCreateRequest,
                      user: dict = Depends(require_roles("student")),
                      store: PlatformStore = Depends(get_store)) -> dict:
    """Accept a typed MVP submission and create an AI draft for teacher review."""
    assessment = store.get_assessment(assessment_id)
    if not assessment or assessment["status"] != "approved":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Approved assessment not found.")
    submission = store.create_submission(assessment_id, user["id"], request.answer_text)
    grading = grade_typed_submission(request.answer_text, assessment)
    evaluation = grading["evaluation"]
    if evaluation.get("status") == "PENDING_TEACHER_REVIEW":
        grade = store.create_grade(submission["id"], float(evaluation["total_score"]), float(evaluation["max_score"]),
                                   evaluation.get("comments", ""), float(evaluation["confidence_score"]), grading)
    else:
        # Persist the submission even if a provider is unavailable. A teacher
        # can later grade it manually; we do not manufacture an AI score.
        grade = None
    return {"submission": submission, "grading": grading, "grade": grade}


@app.get("/grades/pending-review")
def pending_grade_reviews(user: dict = Depends(require_roles("teacher")),
                          store: PlatformStore = Depends(get_store)) -> list[dict]:
    return [grade for grade in store.grades_for_teacher(user["id"]) if grade.get("human_score") is None]


@app.get("/student/grades")
def student_grades(user: dict = Depends(require_roles("student")),
                   store: PlatformStore = Depends(get_store)) -> list[dict]:
    """Give a student access only to their own reviewed/unreviewed grade records."""
    return store.grades_for_student(user["id"])


@app.post("/grade/{grade_id}/review")
def review_grade(grade_id: int, request: GradeReviewRequest,
                 user: dict = Depends(require_roles("teacher")),
                 store: PlatformStore = Depends(get_store)) -> dict:
    """Record the teacher's final score while retaining AI draft/audit details."""
    grade = store.review_grade(grade_id, user["id"], request.human_score, request.comments)
    if not grade:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail="Grade not found, not owned by this teacher, or score exceeds its maximum.")
    return grade


@app.get("/documents")
def documents() -> list[dict]:
    with (current_build_path() / "manifests/documents.csv").open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@app.get("/evaluation/status")
def evaluation_status() -> dict:
    results = sorted((ROOT / "evaluation/results").glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    return {
        "status": "available" if results else "not_run",
        "latest_result": str(results[0]) if results else None,
        "langsmith": langsmith_status(),
    }


@app.get("/observability/summary")
def observability_summary(days: int = 30) -> dict:
    return TelemetryStore().summary(days)
