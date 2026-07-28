"""JWT authentication and role dependencies for the Phase 2 API.

Secrets are read only when a protected request is made.  This preserves the
Phase 1 guarantee that imports and offline tests never activate credentials.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Callable

import bcrypt
import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from backend.shared.persistence import PlatformStore


bearer_scheme = HTTPBearer(auto_error=False)


def _secret() -> str:
    secret = os.getenv("JWT_SECRET_KEY", "").strip()
    if not secret:
        raise RuntimeError("JWT_SECRET_KEY must be configured before using authenticated platform endpoints.")
    if len(secret) < 32:
        raise RuntimeError("JWT_SECRET_KEY must contain at least 32 characters.")
    return secret


def _algorithm() -> str:
    return os.getenv("JWT_ALGORITHM", "HS256")


def hash_password(password: str) -> str:
    """Hash passwords locally with bcrypt; plaintext is never persisted."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except (TypeError, ValueError):
        return False


def create_access_token(user: dict) -> tuple[str, int]:
    expires_in = max(1, int(os.getenv("JWT_EXPIRE_MINUTES", "480")))
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=expires_in)
    claims = {"sub": str(user["id"]), "email": user["email"], "role": user["role"], "exp": expires_at}
    return jwt.encode(claims, _secret(), algorithm=_algorithm()), expires_in * 60


def get_store(request: Request) -> PlatformStore:
    """Lazily initialise one store per FastAPI app, allowing isolated tests."""
    store = getattr(request.app.state, "platform_store", None)
    if store is None:
        store = PlatformStore()
        request.app.state.platform_store = store
    return store


def current_user(credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
                 store: PlatformStore = Depends(get_store)) -> dict:
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication is required.")
    try:
        payload = jwt.decode(credentials.credentials, _secret(), algorithms=[_algorithm()])
        user_id = int(payload["sub"])
    except (jwt.PyJWTError, KeyError, TypeError, ValueError):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired access token.") from None
    user = store.find_user(user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Account no longer exists.")
    return user


def require_roles(*allowed_roles: str) -> Callable:
    """Return a dependency that enforces platform roles on an endpoint."""
    def enforce(user: dict = Depends(current_user)) -> dict:
        if user["role"] not in allowed_roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Your role cannot access this resource.")
        return user
    return enforce
