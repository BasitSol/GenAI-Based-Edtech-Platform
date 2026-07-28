"""Explicit embedding providers with reproducible model identities."""
from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
from dataclasses import dataclass

from backend.shared.core import PROCESSED_ROOT, tokens


@dataclass
class EmbeddingUsage:
    requests: int = 0
    input_tokens: int = 0
    cache_hits: int = 0
    cache_misses: int = 0


class HashEmbeddingService:
    """Deterministic test double; prohibited for production index manifests."""
    provider = "test"
    model = "local-hash-baseline"
    dimensions = 384
    usage = EmbeddingUsage()

    @property
    def identity(self) -> str:
        return f"{self.provider}:{self.model}:{self.dimensions}"

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vector = [0.0] * self.dimensions
            for token in tokens(text):
                value = int(hashlib.sha256(token.encode()).hexdigest()[:8], 16)
                vector[value % self.dimensions] += 1.0 if value & 1 else -1.0
            norm = math.sqrt(sum(value * value for value in vector)) or 1.0
            vectors.append([value / norm for value in vector])
        return vectors


class OpenAIEmbeddingService:
    provider = "openai"

    def __init__(self, model: str):
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(f"OPENAI_API_KEY is required to build/query {model} embeddings.")
        from openai import OpenAI
        self.model = model
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.usage = EmbeddingUsage()
        self.cache_path = PROCESSED_ROOT / "runtime" / "embedding_cache.sqlite"
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.cache_path) as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS embeddings(cache_key TEXT PRIMARY KEY, model TEXT NOT NULL, vector TEXT NOT NULL)")

    @property
    def identity(self) -> str:
        return f"{self.provider}:{self.model}"

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        keys = [hashlib.sha256(f"{self.model}\0{text}".encode("utf-8")).hexdigest() for text in texts]
        unique = list(dict.fromkeys(keys))
        cached: dict[str, list[float]] = {}
        with sqlite3.connect(self.cache_path) as connection:
            for offset in range(0, len(unique), 800):
                batch = unique[offset:offset + 800]
                placeholders = ",".join("?" for _ in batch)
                cached.update({key: json.loads(vector) for key, vector in connection.execute(
                    f"SELECT cache_key,vector FROM embeddings WHERE cache_key IN ({placeholders})", batch
                )})
        self.usage.cache_hits += sum(key in cached for key in keys)
        missing_keys = [key for key in unique if key not in cached]
        if missing_keys:
            text_by_key = dict(zip(keys, texts))
            response = self.client.embeddings.create(
                model=self.model, input=[text_by_key[key] for key in missing_keys], encoding_format="float"
            )
            usage = getattr(response, "usage", None)
            self.usage.requests += 1
            self.usage.cache_misses += len(missing_keys)
            self.usage.input_tokens += int(getattr(usage, "prompt_tokens", 0) or getattr(usage, "total_tokens", 0) or 0)
            vectors = [item.embedding for item in sorted(response.data, key=lambda item: item.index)]
            if len(vectors) != len(missing_keys):
                raise RuntimeError("Embedding provider returned an unexpected vector count")
            with sqlite3.connect(self.cache_path) as connection:
                connection.executemany("INSERT OR REPLACE INTO embeddings VALUES (?,?,?)", [
                    (key, self.model, json.dumps(vector)) for key, vector in zip(missing_keys, vectors)
                ])
            cached.update(dict(zip(missing_keys, vectors)))
        return [cached[key] for key in keys]


def create_embedding_service(*, allow_test_provider: bool = False):
    model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    if model == "local-hash-baseline":
        if not allow_test_provider:
            raise RuntimeError("local-hash-baseline is test-only and cannot build a production index.")
        return HashEmbeddingService()
    if not model.startswith("text-embedding-"):
        raise RuntimeError(
            f"Unsupported active embedding model {model!r}. This low-memory build requires an explicit OpenAI embedding model."
        )
    return OpenAIEmbeddingService(model)
