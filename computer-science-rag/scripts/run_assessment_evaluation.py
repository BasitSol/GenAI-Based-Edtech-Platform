"""Evaluate Phase 2 assessment evidence retrieval against the Phase 1 benchmark."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.assessment_eval import evaluate_assessment_retrieval
from backend.shared.core import ROOT, load_runtime_environment, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate assessment retrieval against Phase 1 gold sources.")
    parser.add_argument("--dataset", type=Path, default=ROOT / "evaluation" / "datasets" / "enterprise_benchmark.jsonl")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    load_runtime_environment()
    result = evaluate_assessment_retrieval(args.dataset, args.limit)
    result["generated_at"] = datetime.now(timezone.utc).isoformat()
    path = ROOT / "evaluation" / "results" / f"assessment_retrieval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    write_json(path, result)
    print({**result, "result_path": str(path)})


if __name__ == "__main__":
    main()
