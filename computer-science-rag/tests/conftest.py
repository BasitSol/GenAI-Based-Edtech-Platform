"""Keep unit/regression tests deterministic, offline and independent of large models."""
import os
from pathlib import Path

import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    # The default user Temp is unavailable in some profiles, while a fixed
    # basetemp can remain locked after a native Windows crash. A process-unique
    # workspace path avoids deleting or reusing either location.
    root = Path(__file__).resolve().parents[1]
    # CI/sandbox users can redirect temporary files when the repository ACL is
    # owned by another Windows account. Normal local runs keep the workspace
    # default and the process-unique directory below.
    temp_root = Path(os.getenv("PYTEST_TEMP_ROOT", str(root / ".pytest_runs")))
    temp_root.mkdir(parents=True, exist_ok=True)
    config.option.basetemp = str(temp_root / f"run_{os.getpid()}")


@pytest.fixture(autouse=True)
def _offline_test_retrieval_models(monkeypatch):
    # Real BGE model loading and ranking are covered by explicit smoke/evaluation
    # runs. Unit tests use the declared test-only embedding so `pytest` never
    # downloads models, consumes API credits or repeatedly allocates multi-GB
    # PyTorch weights on Windows.
    monkeypatch.setenv("EMBEDDING_MODEL", "local-hash-baseline")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "auto")
    monkeypatch.setenv("RERANKER_ENABLED", "false")
    monkeypatch.setenv("EMBEDDING_PROGRESS", "false")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    # A developer's real key may be loaded from .env during test collection.
    # Unit/regression tests must never make paid or nondeterministic API calls;
    # individual API-behaviour tests explicitly inject a fake key when needed.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
