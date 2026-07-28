"""SQLite persistence for the Phase 2 education-platform workflow.

The Phase 1 corpus/index builds are immutable and can be rebuilt at any time.
User accounts, teacher approvals, submissions, and grades must survive those
rebuilds, so this store deliberately lives in ``data_processed/runtime``.
SQLite is sufficient for the single-node FYP deployment and the schema is kept
in one module to make a later PostgreSQL migration straightforward.
"""
from __future__ import annotations

import os
import sqlite3
import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from backend.shared.core import PROCESSED_ROOT


def utc_now() -> str:
    """Return an ISO timestamp suitable for audit fields and JSON responses."""
    return datetime.now(timezone.utc).isoformat()


def default_database_path() -> Path:
    """Resolve an optional explicit database path without loading environment."""
    configured = os.getenv("PLATFORM_DB_PATH", "").strip()
    return Path(configured).expanduser() if configured else PROCESSED_ROOT / "runtime" / "platform.sqlite"


class PlatformStore:
    """Small repository layer with explicit transactions and foreign keys."""

    def __init__(self, path: Path | None = None):
        self.path = path or default_database_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialise_schema()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialise_schema(self) -> None:
        """Create the complete Stage 0 schema idempotently for local deployment."""
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('teacher', 'student')),
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS quizzes_assignments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    teacher_id INTEGER NOT NULL,
                    topic TEXT NOT NULL,
                    difficulty TEXT NOT NULL,
                    type TEXT NOT NULL,
                    content_json TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('draft', 'pending_review', 'approved')),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (teacher_id) REFERENCES users(id)
                );

                CREATE TABLE IF NOT EXISTS submissions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    assignment_id INTEGER NOT NULL,
                    student_id INTEGER NOT NULL,
                    answer_text TEXT,
                    file_path TEXT,
                    submitted_at TEXT NOT NULL,
                    FOREIGN KEY (assignment_id) REFERENCES quizzes_assignments(id),
                    FOREIGN KEY (student_id) REFERENCES users(id)
                );

                CREATE TABLE IF NOT EXISTS grades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    submission_id INTEGER NOT NULL UNIQUE,
                    ai_score REAL,
                    human_score REAL,
                    max_score REAL NOT NULL,
                    comments TEXT,
                    confidence_score REAL,
                    reviewed_by INTEGER,
                    reviewed_at TEXT,
                    FOREIGN KEY (submission_id) REFERENCES submissions(id),
                    FOREIGN KEY (reviewed_by) REFERENCES users(id)
                );
                """
            )

    def create_user(self, email: str, password_hash: str, role: str) -> dict[str, Any]:
        """Persist a teacher/student account and return only safe public fields."""
        timestamp = utc_now()
        with self._connection() as connection:
            cursor = connection.execute(
                "INSERT INTO users (email, password_hash, role, created_at) VALUES (?, ?, ?, ?)",
                (email.strip().lower(), password_hash, role, timestamp),
            )
            user_id = int(cursor.lastrowid)
        return {"id": user_id, "email": email.strip().lower(), "role": role, "created_at": timestamp}

    def find_user_by_email(self, email: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT id, email, password_hash, role, created_at FROM users WHERE email = ?", (email.strip().lower(),)
            ).fetchone()
        return dict(row) if row else None

    def find_user(self, user_id: int) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT id, email, role, created_at FROM users WHERE id = ?", (user_id,)
            ).fetchone()
        return dict(row) if row else None

    def role_counts(self) -> dict[str, int]:
        """Provide lightweight dashboard-safe counts without exposing student data."""
        with self._connection() as connection:
            rows = connection.execute("SELECT role, COUNT(*) AS count FROM users GROUP BY role").fetchall()
            pending = connection.execute(
                "SELECT COUNT(*) AS count FROM quizzes_assignments WHERE status = 'pending_review'"
            ).fetchone()["count"]
            grades = connection.execute(
                "SELECT COUNT(*) AS count FROM grades WHERE human_score IS NULL"
            ).fetchone()["count"]
        counts = {str(row["role"]): int(row["count"]) for row in rows}
        return {"teachers": counts.get("teacher", 0), "students": counts.get("student", 0),
                "pending_assessments": int(pending), "pending_grade_reviews": int(grades)}

    def create_assessment(self, teacher_id: int, topic: str, difficulty: str, assessment_type: str,
                          content: dict[str, Any], status: str = "pending_review") -> dict[str, Any]:
        """Save an auditable generated assessment and its rubric/source trace."""
        timestamp = utc_now()
        with self._connection() as connection:
            cursor = connection.execute(
                """INSERT INTO quizzes_assignments
                   (teacher_id, topic, difficulty, type, content_json, status, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (teacher_id, topic, difficulty, assessment_type, json.dumps(content, ensure_ascii=False),
                 status, timestamp, timestamp),
            )
            assessment_id = int(cursor.lastrowid)
        return self.get_assessment(assessment_id)  # type: ignore[return-value]

    def get_assessment(self, assessment_id: int) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM quizzes_assignments WHERE id = ?", (assessment_id,)).fetchone()
        if not row:
            return None
        assessment = dict(row)
        assessment["content"] = json.loads(assessment.pop("content_json"))
        return assessment

    def approve_assessment(self, assessment_id: int, teacher_id: int) -> dict[str, Any] | None:
        """Approve only the record owned by the teacher performing the review."""
        with self._connection() as connection:
            cursor = connection.execute(
                """UPDATE quizzes_assignments SET status = 'approved', updated_at = ?
                   WHERE id = ? AND teacher_id = ?""",
                (utc_now(), assessment_id, teacher_id),
            )
            if cursor.rowcount != 1:
                return None
        return self.get_assessment(assessment_id)

    def delete_draft_assessment(self, assessment_id: int, teacher_id: int) -> bool:
        """Delete only an unapproved teacher-owned draft with no submissions.

        This deliberately refuses approved assessments so a teacher cannot
        accidentally erase work a student may already have seen or submitted.
        """
        with self._connection() as connection:
            cursor = connection.execute(
                """DELETE FROM quizzes_assignments
                   WHERE id = ? AND teacher_id = ? AND status IN ('draft', 'pending_review')
                   AND NOT EXISTS (SELECT 1 FROM submissions WHERE assignment_id = quizzes_assignments.id)""",
                (assessment_id, teacher_id),
            )
            return cursor.rowcount == 1

    def list_assessments(self, *, teacher_id: int | None = None, approved_only: bool = False) -> list[dict[str, Any]]:
        """Return assessment metadata/content filtered for the caller's role."""
        clauses, values = [], []
        if teacher_id is not None:
            clauses.append("teacher_id = ?")
            values.append(teacher_id)
        if approved_only:
            clauses.append("status = 'approved'")
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connection() as connection:
            rows = connection.execute(f"SELECT * FROM quizzes_assignments{where} ORDER BY created_at DESC", values).fetchall()
        results = []
        for row in rows:
            assessment = dict(row)
            assessment["content"] = json.loads(assessment.pop("content_json"))
            results.append(assessment)
        return results

    def create_submission(self, assignment_id: int, student_id: int, answer_text: str,
                          file_path: str | None = None) -> dict[str, Any]:
        """Persist typed MVP work; uploaded/OCR files can be linked later."""
        timestamp = utc_now()
        with self._connection() as connection:
            cursor = connection.execute(
                """INSERT INTO submissions (assignment_id, student_id, answer_text, file_path, submitted_at)
                   VALUES (?, ?, ?, ?, ?)""", (assignment_id, student_id, answer_text, file_path, timestamp)
            )
            submission_id = int(cursor.lastrowid)
        return self.get_submission(submission_id)  # type: ignore[return-value]

    def get_submission(self, submission_id: int) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM submissions WHERE id = ?", (submission_id,)).fetchone()
        return dict(row) if row else None

    def create_grade(self, submission_id: int, ai_score: float | None, max_score: float,
                     comments: str, confidence_score: float | None, details: dict[str, Any]) -> dict[str, Any]:
        """Store machine grading separately from the later human final score."""
        with self._connection() as connection:
            cursor = connection.execute(
                """INSERT INTO grades (submission_id, ai_score, human_score, max_score, comments, confidence_score)
                   VALUES (?, ?, NULL, ?, ?, ?)""",
                (submission_id, ai_score, max_score, comments, confidence_score),
            )
            grade_id = int(cursor.lastrowid)
            # Structured strengths/weaknesses are audit data; keep them in a
            # companion table without expanding the required plan schema.
            connection.execute("""CREATE TABLE IF NOT EXISTS grade_details (
                grade_id INTEGER PRIMARY KEY, details_json TEXT NOT NULL,
                FOREIGN KEY (grade_id) REFERENCES grades(id) ON DELETE CASCADE)""")
            connection.execute("INSERT INTO grade_details (grade_id, details_json) VALUES (?, ?)",
                               (grade_id, json.dumps(details, ensure_ascii=False)))
        return self.get_grade(grade_id)  # type: ignore[return-value]

    def get_grade(self, grade_id: int) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM grades WHERE id = ?", (grade_id,)).fetchone()
            details = connection.execute("SELECT details_json FROM grade_details WHERE grade_id = ?", (grade_id,)).fetchone()
        if not row:
            return None
        result = dict(row)
        result["details"] = json.loads(details["details_json"]) if details else {}
        return result

    def review_grade(self, grade_id: int, teacher_id: int, human_score: float, comments: str | None = None) -> dict[str, Any] | None:
        """Save a teacher-adjusted final score, preserving the original AI score."""
        with self._connection() as connection:
            grade = connection.execute(
                """SELECT g.max_score, qa.teacher_id FROM grades g
                   JOIN submissions s ON s.id = g.submission_id
                   JOIN quizzes_assignments qa ON qa.id = s.assignment_id WHERE g.id = ?""", (grade_id,)
            ).fetchone()
            if not grade or int(grade["teacher_id"]) != teacher_id or not 0 <= human_score <= float(grade["max_score"]):
                return None
            connection.execute(
                "UPDATE grades SET human_score = ?, comments = COALESCE(?, comments), reviewed_by = ?, reviewed_at = ? WHERE id = ?",
                (human_score, comments, teacher_id, utc_now(), grade_id),
            )
        return self.get_grade(grade_id)

    def grades_for_teacher(self, teacher_id: int) -> list[dict[str, Any]]:
        """List grade records only for assessments owned by the requesting teacher."""
        with self._connection() as connection:
            rows = connection.execute(
                """SELECT g.id FROM grades g JOIN submissions s ON s.id = g.submission_id
                   JOIN quizzes_assignments qa ON qa.id = s.assignment_id
                   WHERE qa.teacher_id = ? ORDER BY s.submitted_at DESC""", (teacher_id,)
            ).fetchall()
        return [grade for row in rows if (grade := self.get_grade(int(row["id"]))) is not None]

    def grades_for_student(self, student_id: int) -> list[dict[str, Any]]:
        """Return only grades belonging to the authenticated student."""
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT g.id FROM grades g JOIN submissions s ON s.id = g.submission_id WHERE s.student_id = ? ORDER BY s.submitted_at DESC",
                (student_id,),
            ).fetchall()
        return [grade for row in rows if (grade := self.get_grade(int(row["id"]))) is not None]
