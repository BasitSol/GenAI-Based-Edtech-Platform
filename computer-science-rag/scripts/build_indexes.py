"""Build all indexes for one staged corpus and atomically promote it."""
from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.shared.core import PROCESSED_ROOT, load_runtime_environment, pending_build_path, read_jsonl, stable_hash, write_json
from backend.module1_rag.indexing.bm25_index import BM25Index
from backend.module1_rag.indexing.chroma_index import ChromaIndex
from backend.module1_rag.indexing.metadata_store import MetadataStore


def build_indexes() -> dict:
    # This is the only build step that needs the embedding credential. Loading
    # is explicit and occurs only after the user invokes this script.
    load_runtime_environment()
    build = pending_build_path()
    manifest_path = build / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "CORPUS_READY":
        raise RuntimeError(f"Pending build status must be CORPUS_READY, got {manifest.get('status')!r}.")
    chunks = read_jsonl(build / "chunks" / "all.jsonl")
    if not chunks:
        raise RuntimeError("Pending corpus contains no chunks.")

    indexes = build / "indexes"
    if indexes.exists():
        shutil.rmtree(indexes)
    indexes.mkdir(parents=True)

    searchable = [item for item in chunks if item.get("content_type") != "PARENT_CONTEXT"]
    BM25Index(searchable).save(indexes / "bm25.json")
    metadata = MetadataStore(indexes / "metadata.sqlite")
    metadata.rebuild(build / "manifests" / "documents.csv", build / "chunks" / "all.jsonl")
    metadata.connection.close()
    dense = ChromaIndex(indexes / "chroma", create=True)
    dense_report = dense.rebuild(ChromaIndex.indexable(chunks))

    embedding_model = dense.embedder.model
    embedding_tokens = dense.embedder.usage.input_tokens
    embedding_requests = dense.embedder.usage.requests
    price = float(os.getenv("EMBEDDING_INPUT_PRICE_PER_MILLION", "0.02"))
    index_identity = stable_hash({
        "build_id": manifest["build_id"],
        "embedding": dense.embedder.identity,
        "sparse": "bm25-v2",
        "vector_store": "chroma-cosine-v2",
    }, 20)
    ready = {
        **manifest,
        "status": "READY",
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "index_identity": index_identity,
        "embedding_provider": dense.embedder.provider,
        "embedding_model": embedding_model,
        "embedding_requests": embedding_requests,
        "embedding_input_tokens": embedding_tokens,
        "embedding_cache_hits": dense.embedder.usage.cache_hits,
        "embedding_cache_misses": dense.embedder.usage.cache_misses,
        "estimated_embedding_cost": round(embedding_tokens * price / 1_000_000, 8),
        "vector_database": "chromadb",
        "vector_count": dense_report["vectors"],
        "vector_dimensions": dense_report["dimensions"],
        "sparse_index": "bm25-v2",
        "reranker_model": os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base"),
        "contextual_enrichment": "deterministic-metadata-v2",
    }
    write_json(manifest_path, ready)
    # The tiny pointer replacement is the commit operation: no runtime can see
    # this build until every index and manifest above has succeeded.
    write_json(PROCESSED_ROOT / "current.json", {"build_id": manifest["build_id"], "index_identity": index_identity})
    (PROCESSED_ROOT / "pending.json").unlink(missing_ok=True)
    return ready


if __name__ == "__main__":
    print(build_indexes())
