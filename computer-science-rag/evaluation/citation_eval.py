"""Canonical citation identity and gold-reference coverage metrics."""
from __future__ import annotations


def _identity(citation: dict) -> tuple:
    return citation.get("document_id"), citation.get("page"), citation.get("chunk_id")


def evaluate_answer(answer: dict, record: dict) -> dict:
    citations = answer.get("citations") or []
    retrieved = {(item.get("document_id"), item.get("page_start"), item.get("chunk_id")) for item in answer.get("retrieved_chunks", [])}
    expected = {(item.get("document_id"), item.get("page")) for item in record.get("expected_citations", [])}
    cited = {_identity(item) for item in citations}
    valid_count = sum((identity[0], identity[1], identity[2]) in retrieved for identity in cited)
    gold_count = sum((identity[0], identity[1]) in expected for identity in cited)
    cited_pages = {(identity[0], identity[1]) for identity in cited}
    return {
        "citation_count": len(citations),
        "citation_identity_accuracy": valid_count / len(cited) if cited else None,
        "citation_gold_precision": gold_count / len(cited) if cited else None,
        "citation_coverage": len(cited_pages & expected) / len(expected) if expected else None,
    }
