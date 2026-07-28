"""Persistent hybrid memory with bounded recent turns and compact summaries."""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from backend.shared.core import PROCESSED_ROOT, tokens


class ConversationStore:
    def __init__(self, path: Path | None = None):
        path = path or PROCESSED_ROOT / "runtime" / "conversations.sqlite"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("CREATE TABLE IF NOT EXISTS conversations(conversation_id TEXT PRIMARY KEY, state TEXT NOT NULL, updated_at TEXT NOT NULL)")
        self.connection.commit()

    def get(self, conversation_id: str | None) -> dict:
        if not conversation_id:
            return {"conversation_id": str(uuid.uuid4()), "recent_messages": [], "summary": "", "entities": []}
        row = self.connection.execute("SELECT state FROM conversations WHERE conversation_id=?", (conversation_id,)).fetchone()
        return json.loads(row[0]) if row else {"conversation_id": conversation_id, "recent_messages": [], "summary": "", "entities": []}

    @staticmethod
    def relevant_messages(query: str, state: dict, limit: int = 4) -> list[dict]:
        query_terms = set(tokens(query))
        messages = state.get("recent_messages", [])
        scored = []
        for position, message in enumerate(messages):
            overlap = len(query_terms & set(tokens(message.get("content", ""))))
            recency = (position + 1) / max(1, len(messages))
            scored.append((overlap * 3 + recency, position, message))
        chosen = sorted(scored, reverse=True)[:limit]
        return [item for _, _, item in sorted(chosen, key=lambda value: value[1])]

    def record(self, state: dict, query: str, answer: str, profile: dict, sources: list[dict]) -> str:
        timestamp = datetime.now(timezone.utc).isoformat()
        messages = state.setdefault("recent_messages", [])
        messages.extend([
            {"role": "user", "content": query, "timestamp": timestamp},
            {"role": "assistant", "content": answer[:1600], "timestamp": timestamp,
             "source_ids": [item.get("chunk_id") for item in sources[:10]]},
        ])
        if len(messages) > 10:
            older, state["recent_messages"] = messages[:-8], messages[-8:]
            compact = " | ".join(f"{item['role']}: {' '.join(item.get('content','').split())[:240]}" for item in older)
            state["summary"] = " | ".join(filter(None, [state.get("summary", ""), compact]))[-2400:]
        state["active_topic"] = query if len(tokens(query)) >= 3 else state.get("active_topic")
        state["last_profile"] = {key: profile.get(key) for key in ("intent", "category", "level", "difficulty")}
        state["updated_at"] = timestamp
        self.connection.execute("INSERT OR REPLACE INTO conversations VALUES (?,?,?)", (
            state["conversation_id"], json.dumps(state, ensure_ascii=False), timestamp,
        ))
        self.connection.commit()
        return state["conversation_id"]
