"""Deterministic metadata context added only to dense embedding input."""
from __future__ import annotations

import os


CONTEXTUAL_ENRICHMENT_VERSION = "metadata-v2"


def contextual_enrichment_enabled() -> bool:
    return os.getenv("CONTEXTUAL_ENRICHMENT_ENABLED", "true").lower() in {"1", "true", "yes", "on"}


def contextualized_text(chunk: dict) -> str:
    """Return a deterministic context-enriched view without changing cited text.

    The enrichment disambiguates qualification, source role, paper identity and
    local section/question context. It is used for embeddings and reranking;
    citations always retain the original chunk text.
    """
    text = chunk.get("text", "")
    if not text or not contextual_enrichment_enabled():
        return text

    labels = [
        ("Qualification", {"A_LEVEL": "Cambridge International A Level Computer Science", "O_LEVEL": "Cambridge O Level Computer Science"}.get(chunk.get("level"), chunk.get("level"))),
        ("Subject code", chunk.get("subject_code")),
        ("Source type", str(chunk.get("document_type", "")).replace("_", " ").title()),
        ("Content type", str(chunk.get("content_type", "")).replace("_", " ").title()),
        ("Document", chunk.get("document_id")),
        ("Year", chunk.get("year")),
        ("Session", chunk.get("session")),
        ("Component", chunk.get("component")),
        ("Question", chunk.get("question_number")),
        ("Page", chunk.get("page_start")),
    ]
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    if first_line and len(first_line) <= 160:
        labels.append(("Local section cue", first_line))
    header = "\n".join(f"{name}: {value}" for name, value in labels if value not in (None, ""))
    return f"{header}\nContent:\n{text}" if header else text
