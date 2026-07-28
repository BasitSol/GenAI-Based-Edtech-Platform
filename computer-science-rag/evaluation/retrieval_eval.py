"""Deterministic retrieval evaluation with complete failure accounting."""
from __future__ import annotations

import math
import time
from pathlib import Path

from evaluation.benchmark import benchmark_status, measurable_records
from backend.module1_rag.retrieval.hybrid_retriever import HybridRetriever
from backend.module1_rag.chat.workflow import retrieve


def _relevant(chunk: dict, source: dict) -> bool:
    if chunk.get("document_id") != source.get("document_id"):
        return False
    if source.get("match_policy") == "PAGE":
        page = source.get("page", source.get("page_start"))
        end = source.get("page_end", page)
        return page is not None and page <= chunk.get("page_start", -1) <= end
    if source.get("chunk_id"):
        return chunk.get("chunk_id") in {source["chunk_id"], source.get("parent_chunk_id")}
    page = source.get("page", source.get("page_start"))
    end = source.get("page_end", page)
    return page is not None and page <= chunk.get("page_start", -1) <= end


def _mean(rows: list[dict], field: str) -> float | None:
    values = [row[field] for row in rows if row.get("execution_status") == "COMPLETED" and row.get(field) is not None]
    return sum(values) / len(values) if values else None


def evaluate(dataset: Path, limit: int | None = None) -> dict:
    records = measurable_records(dataset)
    records = records[:limit] if limit else records
    system = HybridRetriever()
    rows = []
    for record in records:
        started = time.perf_counter()
        base = {"id": record["id"], "question": record["question"]}
        try:
            result = retrieve(record["question"], record.get("level"), record.get("exam_year"),
                              maximum_chunks=10, retriever=system)
            chunks = result["chunks"][:10]
            gold = [dict(source, match_policy=record.get("gold_match_policy", "CHUNK"))
                    for source in record["ground_truth_references"]]
            # Count each gold source at most once. Without this assignment,
            # duplicate child chunks can make nDCG exceed its mathematical
            # maximum of 1.0 and inflate ranking quality.
            assigned: set[int] = set()
            relevance: list[int] = []
            matched_by_k: list[set[int]] = []
            for chunk in chunks:
                matches = [index for index, source in enumerate(gold)
                           if index not in assigned and _relevant(chunk, source)]
                if matches:
                    assigned.add(matches[0])
                    relevance.append(1)
                else:
                    relevance.append(0)
                matched_by_k.append(set(assigned))
            matched = lambda k: matched_by_k[min(k, len(matched_by_k)) - 1] if matched_by_k and k > 0 else set()
            first = next((index + 1 for index, value in enumerate(relevance) if value), None)
            dcg = sum(value / math.log2(index + 2) for index, value in enumerate(relevance))
            ideal_count = min(10, len(gold))
            idcg = sum(1 / math.log2(index + 2) for index in range(ideal_count)) or 1.0
            rows.append({
                **base, "execution_status": "COMPLETED", "error": None,
                "precision_at_5": sum(relevance[:5]) / 5,
                "precision_at_10": sum(relevance[:10]) / 10,
                "recall_at_5": len(matched(5)) / len(gold),
                "recall_at_10": len(matched(10)) / len(gold),
                "reciprocal_rank": 1 / first if first else 0.0,
                "ndcg_at_10": dcg / idcg,
                "exact_scheme_correct": bool(result["exact_mark_scheme_available"]) == bool(record.get("exact_mark_scheme_available")),
                "retrieved_chunk_ids": [chunk.get("chunk_id") for chunk in chunks],
                "retrieved_document_ids": list(dict.fromkeys(chunk.get("document_id") for chunk in chunks)),
                "retrieved_pages": list(dict.fromkeys(chunk.get("page_start") for chunk in chunks)),
                "route": result.get("route"), "retrieval_debug": result.get("retrieval_debug"),
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            })
        except Exception as exc:
            rows.append({**base, "execution_status": "FAILED", "error": f"{type(exc).__name__}: {str(exc)[:500]}",
                         "latency_ms": round((time.perf_counter() - started) * 1000, 2)})
    completed = sum(row["execution_status"] == "COMPLETED" for row in rows)
    return {
        "benchmark": benchmark_status(dataset), "attempted_count": len(rows), "completed_count": completed,
        "failure_count": len(rows) - completed, "coverage": completed / len(rows) if rows else 0.0,
        "precision_at_5": _mean(rows, "precision_at_5"), "precision_at_10": _mean(rows, "precision_at_10"),
        "recall_at_5": _mean(rows, "recall_at_5"), "recall_at_10": _mean(rows, "recall_at_10"),
        "mrr": _mean(rows, "reciprocal_rank"), "ndcg_at_10": _mean(rows, "ndcg_at_10"),
        "exact_scheme_retrieval_accuracy": _mean(rows, "exact_scheme_correct"), "rows": rows,
    }
