"""Regression tests for Stage 0 auth, role enforcement, and chat wrapping."""
from __future__ import annotations

from fastapi.testclient import TestClient

from backend.api.main import app


def _client(monkeypatch, tmp_path) -> TestClient:
    monkeypatch.setenv("JWT_SECRET_KEY", "test-secret-that-is-long-enough-for-jwt-signing")
    monkeypatch.setenv("PLATFORM_DB_PATH", str(tmp_path / "platform.sqlite"))
    # The app persists its store by design; reset it so every test is isolated.
    app.state.platform_store = None
    monkeypatch.setattr("backend.api.main._record_live_answer", lambda *_: None)
    return TestClient(app)


def _register(client: TestClient, email: str, role: str) -> dict:
    response = client.post("/auth/register", json={"email": email, "password": "secure-password", "role": role})
    assert response.status_code == 201, response.text
    return response.json()


def _token(client: TestClient, email: str) -> str:
    response = client.post("/auth/login", json={"email": email, "password": "secure-password"})
    assert response.status_code == 200, response.text
    return response.json()["access_token"]


def test_register_login_and_current_user(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    created = _register(client, "teacher@example.com", "teacher")
    assert created["role"] == "teacher"

    token = _token(client, "teacher@example.com")
    response = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert response.json()["email"] == "teacher@example.com"


def test_role_enforcement_and_duplicate_registration(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    _register(client, "student@example.com", "student")
    duplicate = client.post("/auth/register", json={"email": "student@example.com", "password": "secure-password", "role": "student"})
    assert duplicate.status_code == 409

    token = _token(client, "student@example.com")
    response = client.get("/dashboard/teacher", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 403


def test_authenticated_chat_delegates_to_phase1_answer(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    _register(client, "teacher@example.com", "teacher")
    token = _token(client, "teacher@example.com")

    monkeypatch.setattr("backend.api.main.answer_question", lambda **kwargs: {"answer": "grounded answer", "received": kwargs})
    response = client.post("/chat", json={"query": "Explain binary search."},
                           headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200, response.text
    assert response.json()["answer"] == "grounded answer"
    assert response.json()["platform_user"]["role"] == "teacher"
