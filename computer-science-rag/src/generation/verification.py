"""Deterministic contract checks; semantic quality remains an evaluator task."""
from __future__ import annotations


def verify_response(answer: str, chunks: list[dict], route: dict, context_sufficient: bool,
                    citations: list[dict] | None = None, generation_completed: bool = True) -> dict:
    """Validate identities and completeness without pretending to judge truth."""
    citations = citations or []
    identities = {(item.get("document_id"), item.get("page_start"), item.get("chunk_id")) for item in chunks}
    invalid = [item for item in citations if (item.get("document_id"), item.get("page"), item.get("chunk_id")) not in identities]
    failure = None
    if not context_sufficient:
        failure = "MISSING_CONTEXT"
    elif not generation_completed:
        failure = "GENERATION_NOT_COMPLETED"
    elif not answer.strip():
        failure = "EMPTY_ANSWER"
    elif invalid:
        failure = "INVALID_CITATION_IDENTITY"
    elif route.get("needs_citations", True) and not citations:
        failure = "MISSING_CITATIONS"
    return {
        "passed": failure is None,
        "failure_category": failure,
        "invalid_citations": invalid,
        "citation_identity_valid": not invalid,
        "semantic_faithfulness": {"status": "NOT_MEASURED", "reason": "Requires RAGAS or human review"},
    }
