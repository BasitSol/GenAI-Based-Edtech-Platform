"""Language-agnostic text readability heuristics for selective OCR."""
from __future__ import annotations


def quality_score(text: str) -> float:
    clean = text.strip()
    if not clean:
        return 0.0
    readable = sum(character.isalnum() or character.isspace() or character in ".,:;()[]{}+-*/=<>\"'?!%&£#_" for character in clean)
    density = min(len(clean) / 300, 1.0)
    return round((readable / len(clean)) * density, 3)


def needs_ocr(text: str, min_characters: int = 100) -> bool:
    return len(text.strip()) < min_characters or quality_score(text) < 0.75
