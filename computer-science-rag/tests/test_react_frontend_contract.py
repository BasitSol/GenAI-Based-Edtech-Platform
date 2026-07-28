"""Contract tests for the React/FastAPI boundary without starting Node."""
from __future__ import annotations

from fastapi.middleware.cors import CORSMiddleware

from backend.api.main import app
from backend.shared.core import ROOT


def test_api_has_cors_middleware_for_the_react_development_origin():
    middleware = [item for item in app.user_middleware if item.cls is CORSMiddleware]
    assert len(middleware) == 1
    assert "http://127.0.0.1:5173" in middleware[0].kwargs["allow_origins"]


def test_react_routes_cover_both_platform_roles():
    source = (ROOT / "frontend" / "src" / "App.jsx").read_text(encoding="utf-8")
    assert "Student RAG Assistant" in source
    assert "Assessment generator" in source
    assert "25-mark mock test" in source
    assert "Grade review" in source
    assert "/assessments/available" in source
    assert "/assessments/mine" in source


def test_mock_test_selector_exposes_printed_coursebook_hierarchy():
    source = (ROOT / "frontend" / "src" / "App.jsx").read_text(encoding="utf-8")
    assert "Select chapter(s)" in source
    assert "Choose exact sections" in source
    assert "book_page_label" in source
    assert "not PDF viewer indexes" in source
    assert "MAX_MOCK_CHAPTERS = 8" in source
    assert "MAX_MOCK_TOPICS = 8" in source
    assert "section slots left" in source


def test_legacy_python_ui_entry_points_are_removed():
    legacy_framework = "stream" + "lit"
    assert not (ROOT / "scripts" / f"run_{legacy_framework}.py").exists()
    assert not (ROOT / "scripts" / "run_platform_dashboard.py").exists()
    assert legacy_framework not in (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
