"""Export reproducible human-review packets for non-automatable quality gates."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.shared.core import ROOT, current_build_path, read_jsonl


def _write(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=50)
    args = parser.parse_args()
    build, output = current_build_path(), ROOT / "evaluation" / "review_packets"
    with (build / "manifests" / "documents.csv").open(encoding="utf-8") as handle:
        documents = list(csv.DictReader(handle))
    chunks = read_jsonl(build / "chunks" / "all.jsonl")
    pages = [page for document in documents for page in read_jsonl(build / "pages" / document["document_id"] / "pages.jsonl")]

    metadata = [{**row, "is_correct": "", "reviewer": "", "reviewed_at": "", "notes": ""} for row in documents]
    boundaries = [{"chunk_id": row["chunk_id"], "document_id": row["document_id"],
                   "question_number": row.get("question_number"), "page": row["page_start"],
                   "text": row["text"], "is_correct": "", "reviewer": "", "reviewed_at": "", "notes": ""}
                  for row in chunks if row.get("content_type") in {"QUESTION", "MARK_SCHEME_ENTRY"}][:args.sample_size]
    ocr_candidates = sorted(pages, key=lambda row: (not row.get("ocr_attempted"), row.get("quality_score", 0)))[:args.sample_size]
    ocr = [{"document_id": row["document_id"], "page": row["page_number"], "ocr_engine": row.get("ocr_engine"),
            "quality_score": row.get("quality_score"), "text": row.get("clean_text"), "correct_words": "",
            "reviewed_words": "", "reviewer": "", "reviewed_at": "", "notes": ""} for row in ocr_candidates]
    _write(output / "metadata_review.csv", metadata, list(metadata[0]) if metadata else ["is_correct", "reviewer", "reviewed_at", "notes"])
    _write(output / "question_boundary_review.csv", boundaries, list(boundaries[0]) if boundaries else ["is_correct", "reviewer", "reviewed_at", "notes"])
    _write(output / "ocr_review.csv", ocr, list(ocr[0]) if ocr else ["correct_words", "reviewed_words", "reviewer", "reviewed_at", "notes"])
    print({"output": str(output), "metadata_rows": len(metadata), "boundary_rows": len(boundaries), "ocr_rows": len(ocr)})


if __name__ == "__main__":
    main()
