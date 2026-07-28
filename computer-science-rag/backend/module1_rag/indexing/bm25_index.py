"""Serializable BM25 index with scores and metadata filtering."""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

from backend.shared.core import tokens


class BM25Index:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.documents = [tokens(item.get("retrieval_text") or item["text"]) for item in chunks]
        self.document_frequency = Counter(term for document in self.documents for term in set(document))
        self.average_length = sum(map(len, self.documents)) / max(1, len(self.documents))

    @staticmethod
    def _matches(item: dict, filters: dict | None) -> bool:
        return not filters or all(value is None or item.get(key) == value for key, value in filters.items())

    def search(self, query: str, k: int = 30, filters: dict | None = None) -> list[dict]:
        query_terms = tokens(query)
        count = len(self.documents)
        ranked: list[tuple[float, dict]] = []
        for index, document in enumerate(self.documents):
            chunk = self.chunks[index]
            if not self._matches(chunk, filters):
                continue
            frequencies = Counter(document)
            score = 0.0
            for term in query_terms:
                if term not in frequencies:
                    continue
                inverse_frequency = math.log(1 + (count - self.document_frequency[term] + 0.5) / (self.document_frequency[term] + 0.5))
                numerator = frequencies[term] * 2.2
                denominator = frequencies[term] + 1.2 * (0.25 + 0.75 * len(document) / max(1.0, self.average_length))
                score += inverse_frequency * numerator / denominator
            if score:
                ranked.append((score, chunk))
        ranked.sort(key=lambda pair: pair[0], reverse=True)
        return [{**chunk, "bm25_score": score, "bm25_rank": rank} for rank, (score, chunk) in enumerate(ranked[:k], 1)]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.chunks, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        if not path.exists():
            raise RuntimeError("BM25 index is missing; rebuild the active index.")
        return cls(json.loads(path.read_text(encoding="utf-8")))
