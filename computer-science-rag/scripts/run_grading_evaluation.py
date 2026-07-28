"""Run the non-generative Phase 2 grading-agreement evaluation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.grading_eval import evaluate_grading_agreement
from backend.shared.core import ROOT, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure AI/teacher grading agreement from anonymised CSV data.")
    parser.add_argument("--input", required=True, type=Path, help="CSV with ai_score,human_score columns; do not include student names.")
    args = parser.parse_args()
    result = evaluate_grading_agreement(args.input)
    result["generated_at"] = datetime.now(timezone.utc).isoformat()
    path = ROOT / "evaluation" / "results" / f"grading_agreement_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    write_json(path, result)
    print({**result, "result_path": str(path)})


if __name__ == "__main__":
    main()
