"""Version and upload an approved benchmark to LangSmith."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.benchmark import approved_records
from src.core import ROOT


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="evaluation/datasets/enterprise_benchmark.jsonl")
    parser.add_argument("--name", default="computer-science-rag-enterprise-v2")
    args = parser.parse_args()
    path = Path(args.dataset); path = path if path.is_absolute() else ROOT / path
    records = approved_records(path)
    try:
        from langsmith import Client
    except ImportError as exc:
        raise SystemExit("Install requirements.txt to use LangSmith dataset synchronization") from exc
    client = Client()
    existing = next((item for item in client.list_datasets(dataset_name=args.name)), None)
    dataset = existing or client.create_dataset(dataset_name=args.name, description="Enterprise educational RAG benchmark with gold answers and sources")
    client.create_examples(
        inputs=[{"question": item["question"], "level": item.get("level"), "exam_year": item.get("exam_year")} for item in records],
        outputs=[{"answer": item.get("gold_answer"), "sources": item.get("ground_truth_references"), "category": item.get("category"), "difficulty": item.get("difficulty")} for item in records],
        dataset_id=dataset.id,
    )
    print({"dataset": args.name, "records": len(records), "dataset_id": str(dataset.id)})
