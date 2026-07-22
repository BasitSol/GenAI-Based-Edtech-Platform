"""Dense Chroma index bound to one immutable corpus/model fingerprint."""
from __future__ import annotations

import os
from pathlib import Path

from .embedding_service import create_embedding_service


class ChromaIndex:
    COLLECTION = "educational_chunks"

    def __init__(self, path: Path, *, create: bool = False, allow_test_provider: bool = False):
        import chromadb
        self.path = path
        self.embedder = create_embedding_service(allow_test_provider=allow_test_provider)
        self.client = chromadb.PersistentClient(path=str(path))
        self.collection = self.client.get_or_create_collection(
            self.COLLECTION, embedding_function=None, metadata={"hnsw:space": "cosine"}
        ) if create else self.client.get_collection(self.COLLECTION)

    @staticmethod
    def indexable(chunks: list[dict]) -> list[dict]:
        return [item for item in chunks if item.get("content_type") != "PARENT_CONTEXT"]

    def rebuild(self, chunks: list[dict]) -> dict:
        try:
            self.client.delete_collection(self.COLLECTION)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            self.COLLECTION, embedding_function=None, metadata={"hnsw:space": "cosine"}
        )
        batch_size = max(1, int(os.getenv("EMBEDDING_CACHE_BATCH_SIZE", "250")))
        dimensions = None
        for offset in range(0, len(chunks), batch_size):
            batch = chunks[offset:offset + batch_size]
            vectors = self.embedder.embed_many([item["retrieval_text"] for item in batch])
            if vectors:
                dimensions = len(vectors[0])
            self.collection.add(
                ids=[item["chunk_id"] for item in batch],
                embeddings=vectors,
                documents=[item["text"] for item in batch],
                metadatas=[{
                    "document_id": item["document_id"],
                    "document_type": item["document_type"],
                    "level": item.get("level") or "",
                    "page_start": int(item["page_start"]),
                } for item in batch],
            )
        return {"vectors": len(chunks), "dimensions": dimensions}

    def search(self, query: str, k: int, filters: dict | None = None) -> list[dict]:
        vector = self.embedder.embed_many([query])[0]
        where = {key: value for key, value in (filters or {}).items() if key in {"document_type", "level"} and value is not None}
        result = self.collection.query(
            query_embeddings=[vector],
            n_results=k,
            where=where or None,
            include=["distances", "metadatas"],
        )
        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        return [
            {"chunk_id": chunk_id, "dense_distance": float(distance), "dense_score": 1.0 - float(distance), "dense_rank": rank}
            for rank, (chunk_id, distance) in enumerate(zip(ids, distances), 1)
        ]
