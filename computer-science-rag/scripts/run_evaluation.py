"""Run the versioned enterprise benchmark without hiding incomplete rows."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.answer_eval import evaluate as evaluate_answers
from evaluation.benchmark import benchmark_status, measurable_records
from evaluation.excel_report import generate_workbook
from evaluation.failure_analysis import analyze as analyze_failures
from evaluation.ingestion_eval import evaluate as evaluate_ingestion
from evaluation.memory_eval import evaluate as evaluate_memory
from evaluation.performance_eval import evaluate as evaluate_performance
from evaluation.quality_gates import assess
from ragas_evaluation.evaluator import evaluate_ragas
from evaluation.retrieval_eval import evaluate as evaluate_retrieval
from backend.shared.core import ROOT, current_build_path, load_runtime_environment, write_json
from backend.shared.prompts import PROMPT_LIBRARY_VERSION


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="evaluation/datasets/enterprise_benchmark.jsonl")
    parser.add_argument("--limit", type=int, help="Pilot only; never qualifies as final validation")
    parser.add_argument("--allow-partial", action="store_true", help="Allow a pilot dataset below 100 measurable rows")
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--ragas", action="store_true", help="Use paid judge calls for real RAGAS semantic metrics")
    parser.add_argument("--excel", action="store_true")
    args = parser.parse_args()

    # Credentials are loaded only because the user explicitly invoked this entry point.
    load_runtime_environment()
    dataset = Path(args.dataset)
    dataset = dataset if dataset.is_absolute() else ROOT / dataset
    status = benchmark_status(dataset)
    if status["measurable_count"] < 100 and not args.allow_partial:
        raise SystemExit(f"Benchmark has {status['measurable_count']} measurable rows; 100 are required. Use --allow-partial only for diagnostics.")
    build = current_build_path()
    manifest = json.loads((build / "manifest.json").read_text(encoding="utf-8"))
    records = measurable_records(dataset)
    records = records[:args.limit] if args.limit else records
    retrieval = evaluate_retrieval(dataset, args.limit)
    answers = None if args.retrieval_only else evaluate_answers(dataset, args.limit)
    ragas = ({"status": "NOT_MEASURED", "reason": "--ragas was not selected", "rows": []}
             if not args.ragas or answers is None else evaluate_ragas(records, answers["rows"]))
    report = {
        "report_schema_version": "2.0", "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset), "evaluation_limit": args.limit,
        "pilot_run": bool(args.limit or status["measurable_count"] < 100),
        "system_versions": {
            "build_id": build.name, "source_fingerprint": manifest.get("source_fingerprint"),
            "embedding_model": manifest.get("embedding_model"),
            "reranker_model": manifest.get("reranker_model"),
            "generator_model": __import__("os").getenv("GENERATOR_MODEL", "gpt-4.1-mini"),
            "prompt_version": PROMPT_LIBRARY_VERSION, "benchmark_version": "2.0",
        },
        "benchmark": status, "ingestion": evaluate_ingestion(), "retrieval": retrieval,
        "answers": answers, "ragas": ragas, "memory": evaluate_memory(records),
        "performance": evaluate_performance(answers["rows"]) if answers else None,
        "failure_analysis": analyze_failures(records, retrieval["rows"], (answers or {}).get("rows", [])),
    }
    report["quality_gates"] = assess(report)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = ROOT / "evaluation" / "results" / f"enterprise_benchmark_{stamp}.json"
    write_json(output, report)
    excel = generate_workbook(report, records, output.with_suffix(".xlsx")) if args.excel else None
    print(json.dumps({
        "result_path": str(output), "excel_path": str(excel) if excel else None,
        "pilot_run": report["pilot_run"], "quality_gates": report["quality_gates"],
    }, indent=2))


if __name__ == "__main__":
    main()
