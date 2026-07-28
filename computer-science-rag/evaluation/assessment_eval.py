"""Retrieval evaluation for Phase 2 assessment generation evidence reuse."""
from __future__ import annotations

from pathlib import Path

from evaluation.benchmark import measurable_records
from backend.module1_rag.chat.workflow import retrieve


def evaluate_assessment_retrieval(dataset: Path, limit: int | None = None) -> dict:
    """Check whether assessment-style retrieval still surfaces Phase 1 gold sources.

    Each benchmark topic is transformed into the same assessment-generation
    retrieval request used by the workflow. No assessment or answer is
    generated, so this is a low-cost, reproducible source-reuse evaluation.
    """
    records = measurable_records(dataset)
    records = records[:limit] if limit else records
    rows = []
    for record in records:
        topic = " ".join(record.get("expected_topics", [])) or record["question"]
        query = f"Generate a quiz assessment about {topic} with question paper style and mark scheme criteria"
        try:
            result = retrieve(query, level=record.get("level"), maximum_chunks=12)
            retrieved = {chunk.get("chunk_id") for chunk in result.get("chunks", [])}
            gold = {item.get("chunk_id") for item in record.get("ground_truth_references", [])}
            rows.append({"id": record["id"], "topic": topic, "gold_chunk_ids": sorted(gold),
                         "retrieved_chunk_ids": sorted(retrieved), "gold_source_recalled": bool(gold & retrieved),
                         "retrieved_types": sorted({item.get("document_type") for item in result.get("chunks", [])}),
                         "retrieval_debug": result.get("retrieval_debug", {})})
        except Exception as exc:
            rows.append({"id": record["id"], "topic": topic, "gold_source_recalled": False,
                         "error": f"{type(exc).__name__}: {str(exc)[:300]}"})
    scored = [row for row in rows if "error" not in row]
    return {"attempted_count": len(rows), "scored_count": len(scored),
            "gold_source_recall": sum(row["gold_source_recalled"] for row in scored) / len(scored) if scored else None,
            "rows": rows}
