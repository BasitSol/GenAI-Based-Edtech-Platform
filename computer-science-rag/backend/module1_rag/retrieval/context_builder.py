"""Budgeted, deterministic and citation-preserving context construction."""
from __future__ import annotations

import os
import re

from backend.shared.core import tokens
from backend.module1_rag.monitoring.tracing import traced


PRESERVE_FULL_TYPES = {"MARK_SCHEME", "QUESTION_PAPER", "MARKING_PATTERN"}
COMPRESSION_STOP = {
    "what", "when", "where", "which", "that", "this", "with", "from", "into", "about",
    "explain", "define", "answer", "question", "state", "give", "describe", "identify",
}


def _compression_enabled() -> bool:
    return os.getenv("EXTRACTIVE_COMPRESSION_ENABLED", "true").lower() in {"1", "true", "yes", "on"}


def _segments(text: str) -> list[str]:
    blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    result = []
    for block in blocks:
        # Keep code, tables and short labelled lines intact. Split long prose.
        if "\n" in block or len(block) <= 420:
            result.append(block)
        else:
            result.extend(part.strip() for part in re.split(r"(?<=[.!?])\s+", block) if part.strip())
    return result or ([text.strip()] if text.strip() else [])


def extract_relevant_text(query: str, chunk: dict, maximum_chars: int = 2400) -> tuple[str, bool]:
    text = chunk.get("text", "")
    if not text or len(text) <= maximum_chars or not query or not _compression_enabled():
        return text[:maximum_chars], len(text) > maximum_chars
    if chunk.get("document_type") in PRESERVE_FULL_TYPES:
        return text[:maximum_chars], len(text) > maximum_chars

    query_terms = {term for term in tokens(query) if len(term) > 2 and term not in COMPRESSION_STOP}
    segments = _segments(text)
    scored = []
    for position, segment in enumerate(segments):
        segment_terms = set(tokens(segment))
        overlap = len(query_terms & segment_terms)
        coverage = overlap / max(1, len(query_terms))
        density = overlap / max(1, len(segment_terms))
        heading_bonus = .2 if position == 0 or len(segment) < 120 else 0
        scored.append((coverage * 4 + density * 2 + heading_bonus, position, segment))

    chosen = []
    used = 0
    for _, position, segment in sorted(scored, key=lambda item: (-item[0], item[1])):
        addition = len(segment) + (2 if chosen else 0)
        if chosen and used + addition > maximum_chars:
            continue
        if not chosen and len(segment) > maximum_chars:
            segment = segment[:maximum_chars]
            addition = len(segment)
        chosen.append((position, segment))
        used += addition
        if used >= maximum_chars * .85:
            break
    compressed = "\n\n".join(segment for _, segment in sorted(chosen))
    return compressed or text[:maximum_chars], len(compressed or text[:maximum_chars]) < len(text)


@traced("extractive_context_compression", run_type="tool")
def build_context(chunks: list[dict], query: str = "", maximum_chunks: int = 6, maximum_chars: int = 12000, maximum_chars_per_chunk: int = 2400) -> list[dict]:
    result = []
    chars = 0
    for chunk in chunks:
        if len(result) >= maximum_chunks or chars >= maximum_chars:
            break
        remaining = maximum_chars - chars
        text = chunk.get("text", "")
        if not text:
            continue
        limit = min(remaining, maximum_chars_per_chunk)
        selected, compressed = extract_relevant_text(query, chunk, limit)
        if not selected:
            continue
        item = dict(chunk)
        item["text"] = selected
        item["context_compressed"] = compressed
        item["original_text_chars"] = len(text)
        item["context_text_chars"] = len(selected)
        result.append(item)
        chars += len(selected)
    return result
