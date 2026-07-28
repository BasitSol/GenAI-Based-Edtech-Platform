"""Reproducible PDF ingestion with selective OCR and build isolation."""
from __future__ import annotations

import csv
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import fitz

from backend.shared.core import (
    AUTHORITY, BUILD_SCHEMA_VERSION, PROCESSED_ROOT, ROOT, load_yaml,
    sha256, source_fingerprint, stable_hash, write_csv, write_json,
    write_jsonl,
)
from .cleaner import blocks, clean_text
from .document_classifier import classify_document
from .metadata_parser import extract_metadata
from .ocr import extract_with_fallback
from .text_quality import needs_ocr, quality_score


MANIFEST_FIELDS = [
    "document_id", "source_path", "source_filename", "checksum_sha256",
    "level", "qualification", "subject_code", "document_type", "year",
    "session", "paper_number", "component", "variant", "page_count",
    "authority_level", "exact_pair_id", "native_text_available",
    "ocr_required_pages", "processing_status",
]


def _configured_sources(config: dict):
    for group in config["sources"].values():
        yield from group.values()


def _source_for(path: Path, config: dict) -> dict:
    normalized = path.as_posix().lower()
    for source in _configured_sources(config):
        if source["folder"].lower() not in normalized:
            continue
        required = source.get("filename_contains", [])
        if not required or any(value.lower() in path.name.lower() for value in required):
            return source
    return {"level": "UNKNOWN", "subject_code": "", "expected_document_type": "UNKNOWN"}


def _secondary_text(pdf: Path, index: int) -> str:
    """Use pdfplumber only when PyMuPDF returns low-quality native text."""
    try:
        import pdfplumber
        with pdfplumber.open(pdf) as document:
            return document.pages[index].extract_text(layout=True) or ""
    except Exception:
        return ""


def _save_figure_page(page: fitz.Page, document_id: str, page_number: int, build: Path) -> str:
    target = build / "figures" / document_id / f"page_{page_number:04d}.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False).save(target)
    return target.relative_to(build).as_posix()


def _prepare_build(config_path: Path) -> tuple[dict, Path, str, list[dict], dict]:
    config = load_yaml(config_path)
    fingerprint, inventory = source_fingerprint((ROOT / config["raw_data_root"]).resolve())
    implementation_files = [
        ROOT / "backend" / "module1_rag" / "ingestion" / "pipeline.py",
        ROOT / "backend" / "module1_rag" / "ingestion" / "chunking" / "pipeline.py",
        ROOT / "configs" / "baseline.yaml",
    ]
    settings = {
        "source_fingerprint": fingerprint,
        "schema_version": BUILD_SCHEMA_VERSION,
        "config_sha256": sha256(config_path),
        "implementation_fingerprint": stable_hash(
            [{"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)} for path in implementation_files], 20
        ),
        "parser": "pymupdf+pdfplumber+selective-ocr-v2",
        "chunker": "educational-structure-v2",
    }
    build_id = stable_hash(settings, 20)
    build = PROCESSED_ROOT / "builds" / build_id
    # Never delete the active build.  A repeated command with identical source
    # and implementation fingerprints must be a no-op at the operator level,
    # not a window in which production loses all indexes.
    if build.exists():
        current = PROCESSED_ROOT / "current.json"
        if current.exists() and json.loads(current.read_text(encoding="utf-8")).get("build_id") == build_id:
            raise RuntimeError(
                f"Build {build_id} is already active and will not be overwritten. "
                "No corpus or index rebuild is required unless source/configuration/code changes."
            )
        # Only an inactive partial/staged directory may be safely recreated.
        shutil.rmtree(build)
    build.mkdir(parents=True, exist_ok=False)
    write_json(build / "source_inventory.json", inventory)
    return config, build, build_id, inventory, settings


def build_corpus(config_path: Path = ROOT / "configs/data_sources.yaml") -> dict:
    """Extract, quality-check, chunk, and stage a complete immutable corpus.

    The active index is never modified here.  ``build_indexes.py`` promotes the
    staged build only after sparse, dense, and metadata indexes all succeed.
    """
    config, build, build_id, inventory, settings = _prepare_build(config_path)
    raw_root = (ROOT / config["raw_data_root"]).resolve()
    documents: list[dict] = []
    ocr_rows: list[dict] = []
    unknown: list[dict] = []

    for pdf in sorted(raw_root.rglob("*.pdf")):
        source = _source_for(pdf.relative_to(raw_root), config)
        with fitz.open(pdf) as document:
            preview = "\n".join(document[index].get_text() for index in range(min(3, len(document))))
            document_type = classify_document(pdf.name, preview, str(pdf.parent))
            if document_type == "UNKNOWN":
                document_type = source.get("expected_document_type", "UNKNOWN")
            metadata = extract_metadata(pdf, preview, source["level"], source["subject_code"], document_type)
            pages: list[dict] = []
            ocr_required: list[int] = []

            for page_number, page in enumerate(document, 1):
                native_text = page.get_text("text") or ""
                selected_text = native_text
                secondary_used = False
                if needs_ocr(selected_text):
                    alternate = _secondary_text(pdf, page_number - 1)
                    if quality_score(alternate) > quality_score(selected_text):
                        selected_text, secondary_used = alternate, True

                images = page.get_images(full=True)
                drawings = page.get_drawings()
                ocr_result = None
                # OCR is deliberately selective. Vector-only diagrams are
                # rendered for multimodal questions but are not sent to OCR.
                if needs_ocr(selected_text) and images:
                    ocr_result = extract_with_fallback(page, selected_text)
                    ocr_required.append(page_number)
                    if ocr_result.text and quality_score(ocr_result.text) > quality_score(selected_text):
                        selected_text = ocr_result.text

                cleaned = clean_text(selected_text)
                score = quality_score(cleaned)
                contains_diagram = bool(images) or (document_type == "QUESTION_PAPER" and len(drawings) >= 15)
                figure_path = _save_figure_page(page, metadata["document_id"], page_number, build) if contains_diagram else None
                record = {
                    "document_id": metadata["document_id"],
                    "page_number": page_number,
                    "document_type": document_type,
                    "raw_text": selected_text,
                    "clean_text": cleaned,
                    "blocks": blocks(cleaned),
                    "native_text_available": bool(native_text.strip()),
                    "secondary_parser_used": secondary_used,
                    "ocr_attempted": ocr_result is not None,
                    "ocr_used": bool(ocr_result and ocr_result.text == selected_text),
                    "ocr_engine": ocr_result.engine if ocr_result else None,
                    "ocr_confidence": ocr_result.confidence if ocr_result else None,
                    "ocr_error": ocr_result.error if ocr_result else None,
                    "quality_score": score,
                    "contains_table": bool(re.search(r"(?:\S+\s{3,}){2,}|\bTable\s+\d+", cleaned)),
                    "contains_code": bool(re.search(r"\b(DECLARE|PROCEDURE|FUNCTION|WHILE|FOR|SELECT|FROM)\b", cleaned, re.I)),
                    "contains_diagram": contains_diagram,
                    "figure_path": figure_path,
                }
                pages.append(record)
                if ocr_result is not None or score < 0.75:
                    ocr_rows.append({key: record.get(key) for key in (
                        "document_id", "page_number", "ocr_attempted", "ocr_used",
                        "quality_score", "ocr_engine", "ocr_confidence", "ocr_error",
                    )})

        write_jsonl(build / "pages" / metadata["document_id"] / "pages.jsonl", pages)
        manifest = {
            **metadata,
            "source_path": pdf.relative_to(ROOT).as_posix(),
            "source_filename": pdf.name,
            "checksum_sha256": sha256(pdf),
            "page_count": len(pages),
            "authority_level": AUTHORITY.get(document_type, 0),
            "native_text_available": any(item["native_text_available"] for item in pages),
            "ocr_required_pages": ";".join(map(str, ocr_required)),
            "processing_status": "COMPLETE" if pages else "FAILED",
        }
        documents.append(manifest)
        if document_type == "UNKNOWN":
            unknown.append({"source_path": str(pdf), "reason": "Could not classify document"})

    write_csv(build / "manifests" / "documents.csv", documents, MANIFEST_FIELDS)
    write_csv(build / "manifests" / "ocr.csv", ocr_rows, [
        "document_id", "page_number", "ocr_attempted", "ocr_used",
        "quality_score", "ocr_engine", "ocr_confidence", "ocr_error",
    ])
    write_csv(build / "manifests" / "unknown.csv", unknown, ["source_path", "reason"])

    # Chunking is part of corpus construction so pages and chunks can never be
    # generated by different source versions.
    from backend.module1_rag.ingestion.chunking.pipeline import build_chunks
    chunk_report = build_chunks(build, load_yaml(ROOT / "configs" / "baseline.yaml"))
    report = {
        "build_id": build_id,
        "schema_version": BUILD_SCHEMA_VERSION,
        "status": "CORPUS_READY",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_files": len(inventory),
        "documents": len(documents),
        "pages": sum(item["page_count"] for item in documents),
        "unknown_documents": len(unknown),
        "low_quality_or_ocr_pages": len(ocr_rows),
        **settings,
        **chunk_report,
    }
    write_json(build / "manifest.json", report)
    write_json(PROCESSED_ROOT / "pending.json", {"build_id": build_id})
    return report
