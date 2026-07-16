from __future__ import annotations
from src.core import token_count
def windows(text: str, size: int=450, overlap: int=60):
    words=text.split(); step=max(1,size-overlap)
    for start in range(0,len(words),step):
        value=" ".join(words[start:start+size])
        if value: yield value
        if start+size>=len(words): break
def page_span(pages: list[dict], start: int, end: int) -> tuple[int,int]:
    return pages[start]["page_number"], pages[min(end,len(pages)-1)]["page_number"]
