"""Answer execution and deterministic metrics without misleading proxies."""
from __future__ import annotations

import time
from pathlib import Path

from evaluation.benchmark import benchmark_status, measurable_records
from evaluation.citation_eval import evaluate_answer as evaluate_citations
from src.generation.answer_generator import answer_question


def _mean(rows: list[dict], field: str) -> float | None:
    values = [row[field] for row in rows if row.get("execution_status") == "COMPLETED" and row.get(field) is not None]
    return sum(values) / len(values) if values else None


def evaluate(dataset: Path, limit: int | None = None) -> dict:
    records = measurable_records(dataset)
    records = records[:limit] if limit else records
    rows = []
    for record in records:
        started = time.perf_counter()
        try:
            result = answer_question(record["question"], record.get("level"), record.get("exam_year"))
            metrics = evaluate_citations(result, record)
            rows.append({
                "id": record["id"], "question": record["question"],
                "execution_status": result.get("execution_status", "FAILED_UNKNOWN"),
                "error": result.get("generation_error"), "answer": result.get("answer"),
                "answer_type": result.get("answer_type"), "technical_failure": result.get("technical_failure", False),
                "latency_ms": result.get("latency_ms"), "input_tokens": result.get("input_tokens", 0),
                "output_tokens": result.get("output_tokens", 0), "estimated_cost": result.get("estimated_cost", 0.0),
                "generation_provider": result.get("generation_provider"), "contexts": [item.get("text", "") for item in result.get("retrieved_chunks", [])],
                "retrieved_chunk_ids": [item.get("chunk_id") for item in result.get("retrieved_chunks", [])],
                "citations": result.get("citations", []),
                "verification": result.get("verification"), **metrics,
            })
        except Exception as exc:
            rows.append({
                "id": record["id"], "question": record["question"], "execution_status": "FAILED_PIPELINE",
                "error": f"{type(exc).__name__}: {str(exc)[:500]}", "answer": None,
                "technical_failure": True, "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "input_tokens": 0, "output_tokens": 0, "estimated_cost": 0.0,
            })
    completed = sum(row.get("execution_status") == "COMPLETED" for row in rows)
    return {
        "benchmark": benchmark_status(dataset), "attempted_count": len(rows), "completed_count": completed,
        "failure_count": len(rows) - completed, "coverage": completed / len(rows) if rows else 0.0,
        "citation_identity_accuracy": _mean(rows, "citation_identity_accuracy"),
        "citation_gold_precision": _mean(rows, "citation_gold_precision"),
        "citation_coverage": _mean(rows, "citation_coverage"),
        "semantic_metrics": {
            "status": "NOT_MEASURED", "reason": "Run with --ragas for judge-based faithfulness, relevancy, and correctness.",
            "faithfulness": None, "answer_relevancy": None, "answer_correctness": None,
        },
        "technical_failure_rate": sum(bool(row.get("technical_failure")) for row in rows) / len(rows) if rows else None,
        "total_input_tokens": sum(row.get("input_tokens", 0) for row in rows),
        "total_output_tokens": sum(row.get("output_tokens", 0) for row in rows),
        "estimated_cost": sum(row.get("estimated_cost", 0.0) for row in rows), "rows": rows,
    }
