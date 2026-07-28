"""Ingestion coverage measured against the active immutable build."""
from __future__ import annotations

import csv

from backend.shared.core import current_build_path, read_jsonl


def evaluate() -> dict:
    build = current_build_path()
    with (build / "manifests" / "documents.csv").open(encoding="utf-8") as handle:
        documents = list(csv.DictReader(handle))
    pages = [page for document in documents for page in read_jsonl(build / "pages" / document["document_id"] / "pages.jsonl")]
    expected = sum(int(document["page_count"]) for document in documents)
    diagrams = [page for page in pages if page.get("contains_diagram")]
    attempts = [page for page in pages if page.get("ocr_attempted")]
    successes = [page for page in attempts if page.get("ocr_used")]
    return {
        "build_id": build.name, "documents": len(documents), "expected_pages": expected,
        "extracted_pages": len(pages), "coverage": len(pages) / max(1, expected),
        "readable_page_rate": sum(page.get("quality_score", 0) >= .75 for page in pages) / max(1, len(pages)),
        "ocr_candidate_count": len(attempts), "ocr_success_count": len(successes),
        "ocr_success_rate": len(successes) / max(1, len(attempts)),
        "diagram_page_count": len(diagrams),
        "diagram_image_coverage": sum(bool(page.get("figure_path")) for page in diagrams) / max(1, len(diagrams)),
    }
