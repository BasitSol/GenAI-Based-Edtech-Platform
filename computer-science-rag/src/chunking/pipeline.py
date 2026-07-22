"""Structure-aware chunking for textbooks, syllabuses, and assessment material."""
from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

from src.core import read_jsonl, stable_hash, token_count, write_csv, write_jsonl


def _documents(build: Path) -> list[dict]:
    with (build / "manifests" / "documents.csv").open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _base(document: dict, page: dict, text: str, content_type: str, sequence: int) -> dict:
    identity = {
        "document_id": document["document_id"],
        "page": int(page["page_number"]),
        "type": content_type,
        "sequence": sequence,
        "text": text,
    }
    header = " | ".join(filter(None, [
        f"Qualification: {document.get('level')}",
        f"Source: {document.get('document_type')}",
        f"Subject: {document.get('subject_code')}",
        f"Page: {page.get('page_number')}",
    ]))
    return {
        "chunk_id": f"{document['document_id']}_{content_type.lower()}_{sequence:04d}_{stable_hash(identity, 8)}",
        "document_id": document["document_id"],
        "document_type": document["document_type"],
        "level": document.get("level") or None,
        "subject_code": document.get("subject_code") or None,
        "year": int(document["year"]) if document.get("year") else None,
        "session": document.get("session") or None,
        "component": document.get("component") or None,
        "page_start": int(page["page_number"]),
        "page_end": int(page["page_number"]),
        "text": text.strip(),
        "retrieval_text": f"{header}\n{text.strip()}",
        "content_hash": stable_hash(" ".join(text.lower().split()), 24),
        "token_count": token_count(text),
        "authority_level": int(document.get("authority_level") or 0),
        "content_type": content_type,
        "contains_code": bool(page.get("contains_code")),
        "contains_table": bool(page.get("contains_table")),
        "contains_diagram": bool(page.get("contains_diagram")),
        "figure_path": page.get("figure_path"),
        "parent_chunk_id": None,
    }


def _word_windows(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    step = max(1, size - overlap)
    windows = []
    for start in range(0, len(words), step):
        value = " ".join(words[start:start + size]).strip()
        if value:
            windows.append(value)
        if start + size >= len(words):
            break
    return windows


def _textbook_chunks(document: dict, pages: list[dict], child_tokens: int = 360, overlap_tokens: int = 55) -> list[dict]:
    """Keep children page-local and attach the exact page as parent context.

    The old corpus created global word windows whose page labels drifted as a
    window crossed pages.  Page-local parents make every citation auditable.
    """
    chunks: list[dict] = []
    sequence = 0
    for page in pages:
        text = page.get("clean_text", "").strip()
        if token_count(text) < 20:
            continue
        sequence += 1
        parent = _base(document, page, text, "PARENT_CONTEXT", sequence)
        parent["chunk_id"] = f"{document['document_id']}_page_{page['page_number']:04d}_parent"
        chunks.append(parent)
        for child_text in _word_windows(text, size=child_tokens, overlap=overlap_tokens):
            if token_count(child_text) < 25:
                continue
            sequence += 1
            child = _base(document, page, child_text, "EXPLANATION", sequence)
            child["parent_chunk_id"] = parent["chunk_id"]
            chunks.append(child)
    return chunks


def _syllabus_chunks(document: dict, pages: list[dict]) -> list[dict]:
    chunks: list[dict] = []
    for page in pages:
        text = page.get("clean_text", "")
        sections = re.split(r"(?m)(?=^\s*\d+(?:\.\d+)+\s+)", text)
        if len(sections) == 1:
            sections = _word_windows(text, 320, 40)
        for section in sections:
            if token_count(section) < 18:
                continue
            chunks.append(_base(document, page, section, "LEARNING_OBJECTIVE", len(chunks) + 1))
    return chunks


PAGE_MARKER = re.compile(r"^\[PAGE:(\d+)\]$")
QUESTION_LABEL = re.compile(r"^\s*(\d{1,2})(?:\s*\(([a-z])\))?(?:\s*\(([ivx]+)\))?(?:\s+|\s*$)(.*)$", re.I)
SCHEME_LABEL = re.compile(r"^\s*(\d{1,2})(?:\s*\(([a-z])\))?(?:\s*\(([ivx]+)\))?\s*$", re.I)
SUBPART_LABEL = re.compile(r"^\s*\(([a-z])\)(?:\s*\(([ivx]+)\))?\s*(.*)$", re.I)
ROMAN_LABEL = re.compile(r"^\s*\(([ivx]+)\)\s*(.*)$", re.I)


def _label(number: str, letter: str | None, roman: str | None) -> str:
    return number + (f"({letter.lower()})" if letter else "") + (f"({roman.lower()})" if roman else "")


def _assessment_chunks(document: dict, pages: list[dict], *, mark_scheme: bool) -> list[dict]:
    """Parse assessment entries while retaining page and paper identity.

    Labels are accepted only in a plausible monotonic question range, which
    rejects pseudocode line numbers and printed page numbers without requiring
    hard-coded paper layouts.
    """
    lines: list[tuple[int, str]] = []
    for page in pages:
        lines.extend((int(page["page_number"]), line.rstrip()) for line in page.get("clean_text", "").splitlines())
    chunks: list[dict] = []
    active: dict | None = None
    last_main = 0

    def flush() -> None:
        nonlocal active
        if not active:
            return
        text = "\n".join(active["lines"]).strip()
        # Short one-mark answers (for example a gate name or SQL keyword) are
        # valid evidence; three lexical tokens is enough when identity is exact.
        if token_count(text) < 3:
            active = None
            return
        page = next(item for item in pages if int(item["page_number"]) == active["page"])
        content_type = "MARK_SCHEME_ENTRY" if mark_scheme else "QUESTION"
        chunk = _base(document, page, text, content_type, len(chunks) + 1)
        chunk["question_number"] = active["label"]
        marks = re.findall(r"\[\s*(\d+)\s*\]|\b(\d+)\s+marks?\b", text, re.I)
        mark_value = next((int(left or right) for left, right in reversed(marks)), None)
        chunk["maximum_marks" if mark_scheme else "marks"] = mark_value
        chunk["relationship"] = "EXACT_MARK_SCHEME" if mark_scheme else "QUESTION_IDENTITY"
        chunks.append(chunk)
        active = None

    current_letter: str | None = None
    for line_index, (page_number, line) in enumerate(lines):
        # Question papers commonly print the main number on one line and the
        # subpart on the next.  Capture those boundaries without treating a
        # printed page number as a new question.
        if not mark_scheme and active:
            subpart = SUBPART_LABEL.match(line)
            roman_only = ROMAN_LABEL.match(line)
            if roman_only and current_letter:
                flush()
                active = {"label": _label(str(last_main), current_letter, roman_only.group(1)),
                          "page": page_number, "lines": [line]}
                continue
            if subpart:
                flush()
                current_letter = subpart.group(1).lower()
                active = {"label": _label(str(last_main), current_letter, subpart.group(2)),
                          "page": page_number, "lines": [line]}
                continue
        match = (SCHEME_LABEL.fullmatch(line) if mark_scheme else QUESTION_LABEL.match(line))
        if match:
            number, letter, roman = match.group(1), match.group(2), match.group(3)
            main = int(number)
            tail = "" if mark_scheme else match.group(4).strip()
            plausible = 1 <= main <= 20 and (
                main == last_main or main == last_main + 1 or (last_main == 0 and main == 1)
            )
            # Question-paper labels need either an explicit subpart or useful
            # following text; this avoids treating table values as questions.
            following = next((value.strip() for _, value in lines[line_index + 1:line_index + 4] if value.strip()), "")
            useful = (mark_scheme or letter or roman or len(re.findall(r"[A-Za-z]+", tail)) >= 3
                      or bool(SUBPART_LABEL.match(following))
                      or (not tail and len(re.findall(r"[A-Za-z]+", following)) >= 3))
            if plausible and useful:
                flush()
                last_main = max(last_main, main)
                current_letter = letter.lower() if letter else None
                active = {"label": _label(number, letter, roman), "page": page_number, "lines": [line]}
                continue
        if active:
            active["lines"].append(line)
    flush()
    return chunks


def _examiner_chunks(document: dict, pages: list[dict]) -> list[dict]:
    chunks: list[dict] = []
    for page in pages:
        text = page.get("clean_text", "")
        sections = re.split(r"(?im)(?=^\s*Question\s+\d{1,2}\b)", text)
        for section in sections:
            if token_count(section) < 25:
                continue
            chunk = _base(document, page, section, "EXAMINER_COMMENT", len(chunks) + 1)
            match = re.search(r"(?i)Question\s+(\d{1,2})", section)
            chunk["question_number"] = match.group(1) if match else None
            chunks.append(chunk)
    return chunks


def build_chunks(build: Path, config: dict | None = None) -> dict:
    config = config or {}
    all_chunks: list[dict] = []
    for document in _documents(build):
        pages = read_jsonl(build / "pages" / document["document_id"] / "pages.jsonl")
        document_type = document["document_type"]
        if document_type == "TEXTBOOK":
            chunks = _textbook_chunks(
                document, pages,
                child_tokens=int(config.get("textbook_child_tokens", 360)),
                overlap_tokens=int(config.get("textbook_overlap_tokens", 55)),
            )
        elif document_type == "SYLLABUS":
            chunks = _syllabus_chunks(document, pages)
        elif document_type == "QUESTION_PAPER":
            chunks = _assessment_chunks(document, pages, mark_scheme=False)
        elif document_type == "MARK_SCHEME":
            chunks = _assessment_chunks(document, pages, mark_scheme=True)
        elif document_type == "EXAMINER_REPORT":
            chunks = _examiner_chunks(document, pages)
        else:
            chunks = []
        write_jsonl(build / "chunks" / f"{document['document_id']}.jsonl", chunks)
        all_chunks.extend(chunks)

    write_jsonl(build / "chunks" / "all.jsonl", all_chunks)
    hashes: dict[str, str] = {}
    duplicates: list[dict] = []
    for chunk in all_chunks:
        previous = hashes.get(chunk["content_hash"])
        if previous:
            duplicates.append({"chunk_id": chunk["chunk_id"], "duplicate_of": previous})
        else:
            hashes[chunk["content_hash"]] = chunk["chunk_id"]
    write_csv(build / "manifests" / "duplicates.csv", duplicates, ["chunk_id", "duplicate_of"])
    counts = Counter(item["document_type"] for item in all_chunks)
    write_csv(build / "manifests" / "chunk_counts.csv", [
        {"document_type": key, "chunk_count": value} for key, value in sorted(counts.items())
    ], ["document_type", "chunk_count"])
    return {
        "chunks": len(all_chunks),
        "dense_chunks": sum(item["content_type"] != "PARENT_CONTEXT" for item in all_chunks),
        "duplicate_chunks": len(duplicates),
        "chunk_counts": dict(counts),
    }


if __name__ == "__main__":
    raise SystemExit("Chunking is now part of scripts/build_corpus.py; run that command instead.")
