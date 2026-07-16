"""Second-stage BGE cross-encoder reranking with an observable safe fallback."""
from __future__ import annotations

from functools import lru_cache
import gc
import hashlib
import os
from collections import OrderedDict

from src.core import tokens
from src.indexing.embedding_service import release_local_embedding_model, require_available_memory
from src.indexing.contextual_enrichment import contextualized_text


_SCORE_CACHE: OrderedDict[str, tuple[float, ...]] = OrderedDict()


def _enabled() -> bool:
    return os.getenv("RERANKER_ENABLED", "true").lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=2)
def _load_cross_encoder(model_name: str, device: str | None):
    require_available_memory(model_name, int(os.getenv("RERANKER_MIN_AVAILABLE_MB", "1400")))
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RuntimeError("sentence-transformers is required for BGE cross-encoder reranking") from exc
    return CrossEncoder(model_name, max_length=int(os.getenv("RERANKER_MAX_LENGTH", "512")), device=device)


def release_reranker_model() -> None:
    _load_cross_encoder.cache_clear()
    gc.collect()


def _score_cache_key(model_name: str, query: str, passages: list[str]) -> str:
    identity = f"{model_name}\0{os.getenv('RERANKER_MAX_LENGTH','512')}\0{query}\0" + "\0".join(passages)
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _predict_cached(model, model_name: str, query: str, passages: list[str]) -> tuple[list[float], bool]:
    key = _score_cache_key(model_name, query, passages)
    cached = _SCORE_CACHE.get(key)
    if cached is not None:
        _SCORE_CACHE.move_to_end(key)
        return list(cached), True
    scores = model.predict(
        [[query, passage] for passage in passages],
        batch_size=max(1, int(os.getenv("RERANKER_BATCH_SIZE", "4"))),
        show_progress_bar=False,
    )
    values = tuple(float(score) for score in scores)
    _SCORE_CACHE[key] = values
    maximum = max(1, int(os.getenv("RERANKER_SCORE_CACHE_SIZE", "512")))
    while len(_SCORE_CACHE) > maximum:
        _SCORE_CACHE.popitem(last=False)
    return list(values), False


def _lexical_rerank(query: str, chunks: list[dict], top_k: int) -> list[dict]:
    query_terms = set(tokens(query))

    def score(chunk):
        overlap = len(query_terms & set(tokens(chunk.get("text", ""))))
        exact_bonus = 3 if chunk.get("document_type") == "MARK_SCHEME" else 0
        return overlap + exact_bonus + chunk.get("authority_level", 0) * .05

    result = []
    for chunk in sorted(chunks, key=score, reverse=True)[:top_k]:
        item = dict(chunk)
        item["reranker"] = "lexical_fallback"
        item["reranker_score"] = float(score(chunk))
        result.append(item)
    return result


def rerank_with_debug(query: str, chunks: list[dict], top_k: int = 6, scorer=None) -> tuple[list[dict], dict]:
    if not chunks:
        return [], {"reranker": "none", "reranker_candidates": 0, "reranker_error": None}
    model_name = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
    if not _enabled() and scorer is None:
        ranked = _lexical_rerank(query, chunks, top_k)
        return ranked, {"reranker": "lexical_disabled", "reranker_candidates": len(chunks), "reranker_error": None}
    try:
        if os.getenv("LOCAL_MODEL_MEMORY_MODE", "sequential").lower() == "sequential" and scorer is None:
            release_local_embedding_model()
        model = scorer or _load_cross_encoder(model_name, os.getenv("RERANKER_DEVICE") or None)
        # Give the cross-encoder the same contextual signals used to build the
        # dense index while preserving original text for answer citations.
        passages = [contextualized_text(chunk) for chunk in chunks]
        if scorer is None:
            scores, cache_hit = _predict_cached(model, model_name, query, passages)
        else:
            scores = model.predict(
                [[query, passage] for passage in passages],
                batch_size=max(1, int(os.getenv("RERANKER_BATCH_SIZE", "4"))),
                show_progress_bar=False,
            )
            cache_hit = False
        scored = []
        for chunk, score in zip(chunks, scores):
            item = dict(chunk)
            item["reranker"] = model_name
            item["reranker_score"] = float(score)
            scored.append(item)
        ranked = sorted(scored, key=lambda item: item["reranker_score"], reverse=True)[:top_k]
        return ranked, {"reranker": model_name, "reranker_candidates": len(chunks), "reranker_score_cache_hit": cache_hit, "reranker_error": None}
    except Exception as exc:
        ranked = _lexical_rerank(query, chunks, top_k)
        return ranked, {
            "reranker": "lexical_fallback",
            "reranker_candidates": len(chunks),
            "reranker_error": f"{type(exc).__name__}: {str(exc)[:300]}",
        }


def rerank(query: str, chunks: list[dict], top_k: int = 6) -> list[dict]:
    return rerank_with_debug(query, chunks, top_k)[0]
