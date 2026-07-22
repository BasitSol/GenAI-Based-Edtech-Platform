"""Cross-encoder reranking with an explicit, observable lexical fallback."""
from __future__ import annotations

import os
from functools import lru_cache

from src.core import tokens


@lru_cache(maxsize=1)
def _model(name: str, device: str | None, maximum_length: int):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(name, device=device, max_length=maximum_length)


def _lexical_score(query: str, chunk: dict) -> float:
    query_terms = set(tokens(query))
    chunk_terms = set(tokens(chunk.get("retrieval_text") or chunk.get("text", "")))
    overlap = len(query_terms & chunk_terms) / max(1, len(query_terms))
    return overlap + 0.01 * int(chunk.get("authority_level", 0))


def rerank_with_debug(query: str, chunks: list[dict], top_k: int = 8, scorer=None) -> tuple[list[dict], dict]:
    if not chunks:
        return [], {"model": None, "candidate_count": 0, "fallback": False, "error": None}
    model_name = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
    try:
        if os.getenv("RERANKER_ENABLED", "true").lower() not in {"1", "true", "yes", "on"} and scorer is None:
            raise RuntimeError("reranker disabled by configuration")
        encoder = scorer or _model(model_name, os.getenv("RERANKER_DEVICE") or None, int(os.getenv("RERANKER_MAX_LENGTH", "512")))
        passages = [item.get("retrieval_text") or item["text"] for item in chunks]
        scores = encoder.predict(
            [[query, passage] for passage in passages],
            batch_size=max(1, int(os.getenv("RERANKER_BATCH_SIZE", "4"))),
            show_progress_bar=False,
        )
        ranked = sorted(
            ({**item, "reranker_score": float(score)} for item, score in zip(chunks, scores)),
            key=lambda item: item["reranker_score"], reverse=True,
        )[:top_k]
        return ranked, {"model": model_name, "candidate_count": len(chunks), "fallback": False, "error": None}
    except Exception as exc:
        ranked = sorted(
            ({**item, "reranker_score": _lexical_score(query, item)} for item in chunks),
            key=lambda item: item["reranker_score"], reverse=True,
        )[:top_k]
        return ranked, {
            "model": "lexical_fallback", "candidate_count": len(chunks), "fallback": True,
            "error": f"{type(exc).__name__}: {str(exc)[:240]}",
        }


def rerank(query: str, chunks: list[dict], top_k: int = 8) -> list[dict]:
    return rerank_with_debug(query, chunks, top_k)[0]
