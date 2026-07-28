"""Generate an Excel workbook from an existing JSON evaluation result."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.benchmark import load_records
from evaluation.excel_report import generate_workbook
from backend.shared.core import ROOT


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    parser.add_argument("--dataset", help="Override the dataset path stored in the result")
    parser.add_argument("--output")
    args = parser.parse_args()
    result_path = Path(args.result); result_path = result_path if result_path.is_absolute() else ROOT / result_path
    report = json.loads(result_path.read_text(encoding="utf-8"))
    dataset_path = Path(args.dataset or report["dataset"]); dataset_path = dataset_path if dataset_path.is_absolute() else ROOT / dataset_path
    records = load_records(dataset_path)
    output = Path(args.output) if args.output else result_path.with_suffix(".xlsx"); output = output if output.is_absolute() else ROOT / output
    print({"excel_path": str(generate_workbook(report, records, output)), "records": len(records)})
