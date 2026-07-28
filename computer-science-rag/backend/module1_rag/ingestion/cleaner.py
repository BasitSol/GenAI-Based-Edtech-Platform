"""Conservative PDF text cleanup and block classification."""
from __future__ import annotations

import re


NOISE = [r"Do not write in this margin\.?", r"Candidate (?:name|number).*", r"© UCLES.*",
         r"\.{8,}", r"This document consists of.*"]


def clean_text(text: str) -> str:
    for pattern in NOISE:
        text = re.sub(pattern, "", text, flags=re.I)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def blocks(text: str) -> list[dict]:
    output = []
    for value in text.split("\n"):
        value = value.strip()
        if not value:
            continue
        kind = ("CODE" if re.search(r"\b(DECLARE|FOR|WHILE|IF|PROCEDURE|FUNCTION)\b", value, re.I)
                else "HEADING" if len(value) < 120 and (value.isupper() or re.match(r"^\d+(\.\d+)*\s", value))
                else "PARAGRAPH")
        output.append({"block_type": kind, "text": value})
    return output
