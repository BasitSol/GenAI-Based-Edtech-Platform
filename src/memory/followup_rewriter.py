from __future__ import annotations
import re
def rewrite_followup(query: str, state: dict) -> str:
    """Deterministic rewrite avoids sending unrelated conversation history to retrieval."""
    if not state.get("recent_messages") or not re.match(r"^(it|that|this|and |what about|convert |explain )",query.strip(),re.I): return query
    prior=next((m["content"] for m in reversed(state["recent_messages"]) if m["role"]=="user"),"")
    return f"Context: {prior}\nFollow-up: {query}"
