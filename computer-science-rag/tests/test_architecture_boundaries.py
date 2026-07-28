"""Keep the Phase 1/Phase 2 package boundaries from drifting back together."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from backend.module2_generation.mock_test import generator as mock_test_generator
from backend.module2_generation.quiz import generator as quiz_generator
from backend.shared.core import ROOT


def _python_files(path: Path) -> list[Path]:
    return [item for item in path.rglob("*.py") if "__pycache__" not in item.parts]


def test_module1_does_not_depend_on_phase2_generation():
    for path in _python_files(ROOT / "backend" / "module1_rag"):
        assert "backend.module2_generation" not in path.read_text(encoding="utf-8"), path


def test_frontend_is_plain_react_and_uses_only_the_http_boundary():
    """The UI is a React application and must not import Python backend code."""
    package = json.loads((ROOT / "frontend" / "package.json").read_text(encoding="utf-8"))
    assert "react" in package["dependencies"]
    assert "react-dom" in package["dependencies"]
    assert "next" not in package["dependencies"]
    assert not list((ROOT / "frontend").rglob("*.py"))
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (ROOT / "frontend" / "src").rglob("*")
        if path.suffix in {".js", ".jsx"}
    )
    assert "backend/" not in source
    assert "backend." not in source


def test_legacy_src_imports_are_not_reintroduced():
    roots = ("backend", "frontend", "scripts", "evaluation", "ragas_evaluation", "tests")
    for root in roots:
        for path in _python_files(ROOT / root):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            modules = [
                node.module for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.module
            ]
            modules.extend(
                alias.name for node in ast.walk(tree)
                if isinstance(node, ast.Import) for alias in node.names
            )
            assert not any(module == "src" or module.startswith("src.") for module in modules), path


def test_quiz_service_rejects_mock_test_contract():
    with pytest.raises(ValueError, match="quiz or assignment"):
        quiz_generator.generate_quiz(
            topic="Databases",
            difficulty="medium",
            assessment_type="mock_test",
            question_count=5,
        )


def test_mock_test_service_owns_exact_structure(monkeypatch):
    captured = {}

    def fake_engine(**kwargs):
        captured.update(kwargs)
        return {"validation": {"passed": True}}

    monkeypatch.setattr(mock_test_generator, "generate_assessment", fake_engine)
    result = mock_test_generator.generate_mock_test(
        topic_names=["Database concepts"],
        difficulty="medium",
        level="O_LEVEL",
        selected_topics=[{"id": "ol_database_concepts"}],
        allows_code=False,
    )
    assert result["validation"]["passed"]
    assert captured["assessment_type"] == "mock_test"
    assert captured["question_count"] == 8
    assert captured["total_marks"] == 25
    assert captured["question_format"] == "mixed"
