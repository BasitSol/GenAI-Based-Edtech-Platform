"""Deterministic memory/rewrite evaluation with honest semantic gaps."""
from __future__ import annotations

from src.core import tokens
from src.memory.followup_rewriter import rewrite_followup


def _recall(expected: str, actual: str) -> float:
    terms = {term for term in tokens(expected) if len(term) > 2}
    return len(terms & set(tokens(actual))) / max(1, len(terms))


def evaluate(records: list[dict]) -> dict:
    rows = []
    for record in records:
        test = record.get("memory_test")
        if not test:
            continue
        prior, followup = test["prior_question"], test["followup"]
        state = {"recent_messages": [{"role": "user", "content": prior},
                                     {"role": "assistant", "content": "Prior grounded response"}],
                 "active_topic": prior}
        rewritten = rewrite_followup(followup, state)
        full_history_chars = sum(len(item["content"]) for item in state["recent_messages"])
        injected_chars = max(0, len(rewritten) - len(followup))
        rows.append({
            "id": record["id"], "followup": followup, "rewritten_query": rewritten,
            "query_rewrite_accuracy": _recall(prior, rewritten),
            "context_retention_accuracy": _recall(test["expected_topic"], rewritten),
            "context_compression_ratio": injected_chars / max(1, full_history_chars),
            "token_savings": 1 - min(1, injected_chars / max(1, full_history_chars)),
        })
    mean = lambda field: (sum(row[field] for row in rows) / len(rows)) if rows else None
    return {
        "status": "MEASURED" if rows else "NOT_MEASURED", "scored_count": len(rows),
        "query_rewrite_accuracy": mean("query_rewrite_accuracy"),
        "context_retention_accuracy": mean("context_retention_accuracy"),
        "context_compression_ratio": mean("context_compression_ratio"), "token_savings": mean("token_savings"),
        "followup_answer_accuracy": {"status": "NOT_MEASURED", "reason": "Requires a reviewed multi-turn gold answer set"},
        "conversation_coherence": {"status": "NOT_MEASURED", "reason": "Requires human or judge-model review"},
        "rows": rows,
    }
