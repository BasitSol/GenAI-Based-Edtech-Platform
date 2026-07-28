"""Conservative follow-up detection and standalone-query rewriting."""
from __future__ import annotations

import re


ANAPHORA = re.compile(
    r"\b(?:it|that|this|those|these|the second point|the previous answer|same example|another example)\b|"
    r"^\s*(?:simplify|summari[sz]e|rewrite|convert|continue|what about|why|show another)\b",
    re.I,
)


def depends_on_memory(query: str) -> bool:
    """Return true only for linguistically dependent messages.

    Verbs such as ``explain`` are intentionally not sufficient: "Explain
    binary search" is a complete new question and must never inherit SQL from
    the previous turn.
    """
    return bool(ANAPHORA.search(query.strip()))


def rewrite_followup(query: str, state: dict) -> str:
    if not depends_on_memory(query):
        return query.strip()
    recent = state.get("recent_messages", [])
    prior_user = next((item.get("content", "") for item in reversed(recent) if item.get("role") == "user"), "")
    prior_topic = state.get("active_topic") or prior_user
    if not prior_topic:
        return query.strip()
    return f"Previous topic: {prior_topic}\nCurrent request: {query.strip()}"
