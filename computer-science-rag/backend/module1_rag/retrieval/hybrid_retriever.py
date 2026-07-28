"""Adaptive hybrid retrieval over one validated immutable build."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from backend.shared.core import current_build_path
from backend.module1_rag.indexing.bm25_index import BM25Index
from backend.module1_rag.indexing.chroma_index import ChromaIndex
from backend.module1_rag.indexing.metadata_store import MetadataStore
from .adaptive_policy import plan_for, query_variants, type_priority
from .query_classifier import classify


RRF_CONSTANT = 60


def _rrf(rankings: list[list[dict]]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, 1):
            scores[item["chunk_id"]] = scores.get(item["chunk_id"], 0.0) + 1.0 / (RRF_CONSTANT + rank)
    return scores


class HybridRetriever:
    def __init__(self, build: Path | None = None):
        self.build = build or current_build_path()
        self.manifest = json.loads((self.build / "manifest.json").read_text(encoding="utf-8"))
        self.bm25 = BM25Index.load(self.build / "indexes" / "bm25.json")
        self.by_id = {item["chunk_id"]: item for item in self.bm25.chunks}
        self.metadata = MetadataStore(self.build / "indexes" / "metadata.sqlite")
        self.dense = ChromaIndex(self.build / "indexes" / "chroma")
        if self.dense.embedder.model != self.manifest.get("embedding_model"):
            raise RuntimeError(
                f"Active index uses {self.manifest.get('embedding_model')}, but runtime requests {self.dense.embedder.model}. "
                "Rebuild indexes or restore the matching EMBEDDING_MODEL."
            )

    def close(self) -> None:
        """Release request-scoped database handles deterministically."""
        self.metadata.close()

    def _exact(self, route: dict) -> list[dict]:
        required = ("subject_code", "year", "session", "component")
        if not all(route.get(key) is not None for key in required):
            return []
        return self.metadata.exact_paper_chunks(
            subject_code=route["subject_code"], year=int(route["year"]),
            session=route["session"], component=str(route["component"]),
            question_number=route.get("question_number"),
        )

    def retrieve(self, query: str, level: str | None = None, exam_year: int | None = None, result_limit: int = 24,
                 document_type: str | None = None) -> dict:
        started = time.perf_counter()
        usage_before = (
            self.dense.embedder.usage.requests, self.dense.embedder.usage.input_tokens,
            self.dense.embedder.usage.cache_hits, self.dense.embedder.usage.cache_misses,
        )
        route = classify(query, level, exam_year)
        plan = plan_for(route)
        variants = query_variants(query, route, plan)
        # Source-specific lanes are used by assessment generation to guarantee
        # that a "past-paper" request actually searches question-paper chunks,
        # rather than relying on mixed ranking to surface one incidentally.
        filters = {"level": route.get("level"), "document_type": document_type}
        sparse_rankings: list[list[dict]] = []
        dense_rankings: list[list[dict]] = []
        for variant in variants:
            sparse_rankings.append(self.bm25.search(variant, plan.sparse_k, filters))
            dense_hits = self.dense.search(variant, plan.dense_k, filters)
            dense_rankings.append([
                {**self.by_id[hit["chunk_id"]], **hit}
                for hit in dense_hits if hit["chunk_id"] in self.by_id
            ])

        exact = self._exact(route)
        exact_scheme = any(item["document_type"] == "MARK_SCHEME" for item in exact)
        scores = _rrf(sparse_rankings + dense_rankings)
        candidates = []
        seen = set()
        # Exact paper identity is deterministic and always precedes semantic
        # support, but only a matching scheme is factual answer authority.
        for item in sorted(exact, key=lambda chunk: (chunk["document_type"] != "MARK_SCHEME", chunk["page_start"])):
            if document_type and item.get("document_type") != document_type:
                continue
            if item["chunk_id"] not in seen:
                candidates.append({**item, "rrf_score": 1.0, "retrieval_route": "exact_metadata"})
                seen.add(item["chunk_id"])
        for chunk_id in sorted(scores, key=lambda value: scores[value] + type_priority(self.by_id[value], plan) * 1e-6, reverse=True):
            if chunk_id in seen:
                continue
            item = self.by_id[chunk_id]
            # Source authority is enforced before reranking.  Generic theory
            # cannot be answered from unrelated assessment material merely
            # because it shares vocabulary with the question.
            if ((not document_type and item.get("document_type") not in plan.preferred_types)
                    or (document_type and item.get("document_type") != document_type)):
                continue
            relationship = ("ASSESSMENT_PATTERN" if item.get("document_type") == "MARK_SCHEME"
                            else "QUESTION_IDENTITY" if item.get("document_type") == "QUESTION_PAPER"
                            else "CURRICULUM_EVIDENCE")
            candidates.append({**item, "relationship": relationship,
                               "rrf_score": scores[chunk_id], "retrieval_route": "hybrid_rrf"})
            seen.add(chunk_id)
            if len(candidates) >= result_limit:
                break

        usage_after = (
            self.dense.embedder.usage.requests, self.dense.embedder.usage.input_tokens,
            self.dense.embedder.usage.cache_hits, self.dense.embedder.usage.cache_misses,
        )
        request_delta, token_delta, hit_delta, miss_delta = [after - before for before, after in zip(usage_before, usage_after)]
        embedding_cost = token_delta * float(os.getenv("EMBEDDING_INPUT_PRICE_PER_MILLION", "0.02")) / 1_000_000
        return {
            "route": route,
            "chunks": candidates,
            "exact_mark_scheme_available": exact_scheme,
            "retrieval_debug": {
                "build_id": self.manifest["build_id"],
                "index_identity": self.manifest["index_identity"],
                "query_variants": variants,
                "metadata_filters": {key: value for key, value in filters.items() if value is not None},
                "exact_hits": len(exact),
                "sparse_hits": [len(items) for items in sparse_rankings],
                "dense_hits": [len(items) for items in dense_rankings],
                "embedding_requests": request_delta, "embedding_input_tokens": token_delta,
                "embedding_cache_hits": hit_delta, "embedding_cache_misses": miss_delta,
                "embedding_cost": round(embedding_cost, 8),
                "retrieval_plan": plan.to_dict(),
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            },
        }
