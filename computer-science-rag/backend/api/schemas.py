"""Transport schemas for the Phase 1 compatibility and Phase 2 platform APIs."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# Avoid an unnecessary runtime dependency while still rejecting malformed
# account identifiers. Full provider/domain verification is intentionally not
# attempted by a local authentication service.
Email = str
EMAIL_PATTERN = r"^[^\s@]+@[^\s@]+\.[^\s@]+$"


class AskRequest(BaseModel):
    query: str
    level: str | None = None
    exam_year: int | None = None
    conversation_id: str | None = None
    difficulty: str | None = None


class RegisterRequest(BaseModel):
    email: Email = Field(pattern=EMAIL_PATTERN, max_length=254)
    password: str = Field(min_length=8, max_length=128)
    role: Literal["teacher", "student"]


class LoginRequest(BaseModel):
    email: Email = Field(pattern=EMAIL_PATTERN, max_length=254)
    password: str = Field(min_length=1, max_length=128)


class TokenResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int


class UserResponse(BaseModel):
    id: int
    email: Email
    role: Literal["teacher", "student"]
    created_at: str


class ChatRequest(AskRequest):
    """Authenticated Phase 2 wrapper around the unchanged Phase 1 contract."""


class AssessmentCreateRequest(BaseModel):
    topic: str = Field(min_length=2, max_length=200)
    difficulty: Literal["easy", "medium", "hard", "beginner", "intermediate", "advanced"]
    assessment_type: Literal["quiz", "assignment"]
    question_count: int = Field(ge=1, le=20)
    question_format: Literal["mcq", "short_answer", "long_answer", "mixed"] = "mixed"
    level: str | None = None


class MockTestCreateRequest(BaseModel):
    """Strict payload for a Phase 2 chapter/topic mock test, never a full exam."""
    level: Literal["O_LEVEL", "A_LEVEL"]
    chapter_ids: list[str] = Field(min_length=1, max_length=8)
    topic_ids: list[str] = Field(min_length=1, max_length=8)
    difficulty: Literal["easy", "medium", "hard", "beginner", "intermediate", "advanced"] = "medium"


class SubmissionCreateRequest(BaseModel):
    answer_text: str = Field(min_length=1, max_length=50_000)


class GradeReviewRequest(BaseModel):
    human_score: float = Field(ge=0)
    comments: str | None = Field(default=None, max_length=5_000)
