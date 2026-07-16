from __future__ import annotations
import re
NOISE = [r"Do not write in this margin\.?", r"Candidate (?:name|number).*", r"© UCLES.*", r"\.{8,}", r"This document consists of.*"]
def clean_text(text: str) -> str:
    for pattern in NOISE: text = re.sub(pattern, "", text, flags=re.I)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
def blocks(text: str) -> list[dict]:
    out=[]
    for item in text.split("\n"):
        item=item.strip()
        if not item: continue
        kind="CODE" if re.search(r"\b(DECLARE|FOR|WHILE|IF|PROCEDURE|FUNCTION)\b", item, re.I) else "HEADING" if len(item)<120 and (item.isupper() or re.match(r"^\d+(\.\d+)*\s", item)) else "PARAGRAPH"
        out.append({"block_type":kind,"text":item})
    return out
