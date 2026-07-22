"""Quality gates that cannot pass with partial or unavailable measurements."""
from __future__ import annotations

import csv

from src.core import ROOT


def _gate(value, operator: str, target: float, *, measured: bool = True, reason: str | None = None) -> dict:
    passed = None if value is None or not measured else (value >= target if operator == ">=" else value <= target)
    return {"value": value, "operator": operator, "target": target, "status": "MEASURED" if passed is not None else "NOT_MEASURED",
            "passed": passed, "reason": reason if passed is None else None}


def _informational(value, *, measured: bool, reason: str | None = None) -> dict:
    """Expose an observed metric without treating a chosen threshold as a failure."""
    return {"value": value, "operator": "INFORMATIONAL", "target": None,
            "status": "MEASURED" if measured and value is not None else "NOT_MEASURED",
            "passed": None, "reason": None if measured and value is not None else reason}


def _manual_review_metrics() -> dict[str, float | None]:
    folder = ROOT / "evaluation" / "review_packets"
    values: dict[str, float | None] = {"metadata_accuracy": None, "question_boundary_accuracy": None, "ocr_text_accuracy": None}
    for filename, metric in (("metadata_review.csv", "metadata_accuracy"), ("question_boundary_review.csv", "question_boundary_accuracy")):
        path = folder / filename
        if not path.exists():
            continue
        with path.open(encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        reviewed = [row for row in rows if row.get("reviewer") and row.get("reviewed_at") and row.get("is_correct") in {"0", "1"}]
        if reviewed and len(reviewed) == len(rows):
            values[metric] = sum(int(row["is_correct"]) for row in reviewed) / len(reviewed)
    path = folder / "ocr_review.csv"
    if path.exists():
        with path.open(encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        reviewed = [row for row in rows if row.get("reviewer") and row.get("reviewed_at") and row.get("reviewed_words")]
        if reviewed and len(reviewed) == len(rows):
            total = sum(int(row["reviewed_words"]) for row in reviewed)
            values["ocr_text_accuracy"] = sum(int(row["correct_words"]) for row in reviewed) / max(1, total)
    return values


def assess(report: dict) -> dict:
    benchmark, ingestion = report.get("benchmark") or {}, report.get("ingestion") or {}
    retrieval, answers, performance = report.get("retrieval") or {}, report.get("answers") or {}, report.get("performance") or {}
    ragas = report.get("ragas") or {}
    manual = _manual_review_metrics()
    gates = {
        "benchmark_size": _gate(benchmark.get("measurable_count"), ">=", 100),
        "retrieval_evaluation_coverage": _gate(retrieval.get("coverage"), ">=", 1.0),
        "answer_evaluation_coverage": _gate((answers or {}).get("coverage"), ">=", 1.0, measured=answers is not None, reason="Answer evaluation not run"),
        "page_extraction_coverage": _gate(ingestion.get("coverage"), ">=", .99),
        "recall_at_5": _gate(retrieval.get("recall_at_5"), ">=", .90),
        "recall_at_10": _gate(retrieval.get("recall_at_10"), ">=", .95),
        "mrr": _gate(retrieval.get("mrr"), ">=", .75),
        "ndcg_at_10": _gate(retrieval.get("ndcg_at_10"), ">=", .85),
        "exact_scheme_retrieval_accuracy": _gate(retrieval.get("exact_scheme_retrieval_accuracy"), ">=", .98),
        "citation_identity_accuracy": _gate((answers or {}).get("citation_identity_accuracy"), ">=", 1.0, measured=answers is not None),
        "citation_coverage": _gate((answers or {}).get("citation_coverage"), ">=", .90, measured=answers is not None),
        "technical_failure_rate": _gate((answers or {}).get("technical_failure_rate"), "<=", .01, measured=answers is not None),
        "median_latency_ms": _gate(performance.get("median_latency_ms"), "<=", 5000, measured=bool(performance)),
        "p95_latency_ms": _gate(performance.get("p95_latency_ms"), "<=", 10000, measured=bool(performance)),
        "metadata_accuracy": _gate(manual["metadata_accuracy"], ">=", .98, reason="Complete metadata_review.csv"),
        "question_boundary_accuracy": _gate(manual["question_boundary_accuracy"], ">=", .97, reason="Complete question_boundary_review.csv"),
        "ocr_text_accuracy": _gate(manual["ocr_text_accuracy"], ">=", .95, reason="Complete ocr_review.csv"),
    }
    ragas_measured = ragas.get("status") == "COMPLETED" and ragas.get("coverage") == 1.0
    for name in ("context_precision", "context_recall", "faithfulness", "answer_relevancy",
                 "answer_correctness", "noise_sensitivity"):
        gates[f"ragas_{name}"] = _informational(
            ragas.get(name), measured=ragas_measured,
            reason=ragas.get("reason", "RAGAS not run with full coverage"),
        )
    measured = [gate["passed"] for gate in gates.values() if gate["passed"] is not None]
    return {"measured_gate_count": len(measured), "total_gate_count": len(gates),
            "all_measured_gates_passed": bool(measured) and all(measured),
            "all_required_gates_passed": len(measured) == len(gates) and all(measured), "gates": gates}
