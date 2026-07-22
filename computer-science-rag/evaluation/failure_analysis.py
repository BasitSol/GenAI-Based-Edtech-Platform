"""Failure taxonomy based only on observed, defensible signals."""
from __future__ import annotations


def analyze(records: list[dict], retrieval_rows: list[dict], answer_rows: list[dict]) -> dict:
    retrieval = {row["id"]: row for row in retrieval_rows}
    answers = {row["id"]: row for row in answer_rows}
    output, counts = [], {}
    for record in records:
        row_r, row_a = retrieval.get(record["id"]), answers.get(record["id"])
        category, recommendation = None, None
        if not row_r or row_r.get("execution_status") != "COMPLETED":
            category, recommendation = "RETRIEVAL_EXECUTION_FAILURE", "Inspect the row error, build manifest, and provider configuration."
        elif not row_r.get("recall_at_10"):
            category, recommendation = "RETRIEVAL_RELEVANCE_FAILURE", "Inspect chunk boundaries, query variants, filters, fusion, and reranker ranks."
        elif row_a and row_a.get("execution_status") != "COMPLETED":
            category, recommendation = "ANSWER_EXECUTION_FAILURE", "Inspect generation status/error; do not score missing output as a semantic answer."
        elif row_a and row_a.get("citation_identity_accuracy") is not None and row_a["citation_identity_accuracy"] < 1:
            category, recommendation = "CITATION_IDENTITY_FAILURE", "Inspect source-key mapping and immutable chunk identities."
        if category:
            counts[category] = counts.get(category, 0) + 1
            output.append({"id": record["id"], "failure_category": category, "recommendation": recommendation,
                           "retrieval_error": (row_r or {}).get("error"), "answer_error": (row_a or {}).get("error")})
    return {"failure_count": len(output), "by_category": counts, "rows": output}
