"""Embedding providers used by the Chroma dense-retrieval index."""
from __future__ import annotations

import hashlib
import math
import os
import gc
from functools import lru_cache
from typing import Protocol

from src.core import tokens


def available_memory_mb() -> int | None:
    try:
        import psutil
        return int(psutil.virtual_memory().available / (1024 * 1024))
    except Exception:
        return None


def require_available_memory(model_label: str, minimum_mb: int) -> None:
    available = available_memory_mb()
    if available is not None and available < minimum_mb:
        raise RuntimeError(
            f"{model_label} requires approximately {minimum_mb} MB of available RAM in full-precision CPU mode; "
            f"only {available} MB is currently available. Close browsers, IDEs, Streamlit and other memory-heavy "
            "applications, then rerun the command. The model and index have not been downgraded."
        )


def release_local_embedding_model() -> None:
    """Release full-precision BGE weights before the full-precision reranker loads."""
    _load_sentence_transformer.cache_clear()
    gc.collect()


class EmbeddingService(Protocol):
    provider: str
    model: str
    def embed_many(self, texts: list[str]) -> list[list[float]]: ...
    def embed(self, text: str) -> list[float]: ...


class HashEmbeddingService:
    """Offline-only fallback for tests and source inspection; not a quality RAG embedding."""
    provider = "local_hash_fallback"
    model = "local-hash-baseline"
    dimensions = 384

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in tokens(text):
            value = int(hashlib.sha256(token.encode()).hexdigest()[:8], 16)
            vector[value % self.dimensions] += 1 if value & 1 else -1
        size = math.sqrt(sum(item * item for item in vector)) or 1
        return [item / size for item in vector]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


class OpenAIEmbeddingService:
    provider = "openai"

    def __init__(self, model: str | None = None):
        from openai import OpenAI
        self.model = model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.client = OpenAI()

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self.client.embeddings.create(model=self.model, input=texts, encoding_format="float")
        return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]


@lru_cache(maxsize=2)
def _load_sentence_transformer(model: str, device: str | None, max_length: int):
    """Share heavyweight encoders across indexes and application requests."""
    require_available_memory(model, int(os.getenv("EMBEDDING_MIN_AVAILABLE_MB", "2800")))
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required for local BGE embeddings. "
            "Install requirements.txt before building the index."
        ) from exc
    encoder = SentenceTransformer(model, device=device)
    encoder.max_seq_length = max_length
    return encoder


class SentenceTransformerEmbeddingService:
    """Local normalized dense embeddings, used for BAAI/BGE models."""
    provider = "sentence_transformers"

    def __init__(self, model: str = "BAAI/bge-m3"):
        self.model = model
        # BGE-M3 is a long-context embedding model. SentenceTransformers pads to
        # the longest item in each batch, not this ceiling, so retaining the
        # native 8192-token capacity does not force every short chunk to 8K.
        self.max_length = max(64, int(os.getenv("EMBEDDING_MAX_LENGTH", "8192")))

    @property
    def cache_identity(self) -> str:
        return f"{self.provider}:{self.model}:max_length={self.max_length}:normalized=true"

    def _load(self):
        # Keep BGE-M3's configured long-context capacity. Actual compute follows
        # the longest input in a batch because padding is dynamic.
        return _load_sentence_transformer(self.model, os.getenv("EMBEDDING_DEVICE") or None, self.max_length)

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        batch_size = max(1, int(os.getenv("EMBEDDING_BATCH_SIZE", "8")))
        show_progress = len(texts) >= 16 and os.getenv("EMBEDDING_PROGRESS", "true").lower() in {"1", "true", "yes", "on"}
        vectors = self._load().encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
        return vectors.tolist()

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]


def create_embedding_service() -> EmbeddingService:
    """Select the configured model without silently changing vector spaces."""
    model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    if model == "local-hash-baseline":
        return HashEmbeddingService()
    provider = os.getenv("EMBEDDING_PROVIDER", "auto").lower()
    is_openai_model = model.lower().startswith("text-embedding-")
    if model.lower().startswith("baai/bge") or (provider in {"sentence_transformers", "local", "bge"} and not is_openai_model):
        return SentenceTransformerEmbeddingService(model)
    if os.getenv("OPENAI_API_KEY"):
        return OpenAIEmbeddingService(model)
    return UnavailableOpenAIEmbeddingService(model)


class UnavailableOpenAIEmbeddingService:
    """Preserve OpenAI cache identity while making missing credentials explicit."""
    provider = "openai"

    def __init__(self, model: str):
        self.model=model

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError(
            f"OPENAI_API_KEY is required for uncached {self.model} embeddings. "
            "Set the key or use EMBEDDING_MODEL=local-hash-baseline only for tests."
        )

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]
