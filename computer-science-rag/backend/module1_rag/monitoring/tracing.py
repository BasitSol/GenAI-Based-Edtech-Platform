"""Deliberately opt-in LangSmith tracing with a transparent local path."""
from __future__ import annotations

import os
from functools import wraps


def _enabled() -> bool:
    flag = os.getenv("LANGSMITH_TRACING", os.getenv("LANGCHAIN_TRACING_V2", "false")).lower()
    return flag in {"1", "true", "yes", "on"} and bool(os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY"))


def traced(name: str, run_type: str = "chain"):
    """Trace at call time only when both consent flag and credential exist."""
    def decorator(function):
        traced_function = None

        @wraps(function)
        def wrapper(*args, **kwargs):
            nonlocal traced_function
            if not _enabled():
                return function(*args, **kwargs)
            if traced_function is None:
                from langsmith import traceable
                traced_function = traceable(name=name, run_type=run_type)(function)
            return traced_function(*args, **kwargs)
        return wrapper
    return decorator


def langsmith_status() -> dict:
    return {"enabled": _enabled(),
            "configured": bool(os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")),
            "project": os.getenv("LANGSMITH_PROJECT", os.getenv("LANGCHAIN_PROJECT", "computer-science-rag-v2"))}
