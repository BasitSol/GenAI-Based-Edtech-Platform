from __future__ import annotations
import re
from pathlib import Path
from backend.shared.core import slug
def extract_metadata(path: Path, text: str, level: str, subject_code: str, doc_type: str) -> dict:
    value=f"{path.name} {text[:5000]}".lower()
    years=re.findall(r"\b(20\d{2})\b", value); year=int(years[0]) if years else None
    session = "MJ" if re.search(r"may\s*/\s*june", value) else "FM" if re.search(r"february\s*/\s*march", value) else "ON" if re.search(r"october\s*/\s*november", value) else None
    header=re.search(r"\b(2210|9618)\s*/\s*(\d{2})\b", value)
    paper=re.search(r"(?:paper|component)\s*([123])\b", value)
    component=header.group(2) if header else (paper.group(1) if paper else None)
    # A paper identity only exists for a question paper or its matching scheme.
    exact_pair=f"{subject_code}_{session}_{year}_{component}" if doc_type in {"QUESTION_PAPER","MARK_SCHEME"} and year and session and component else None
    suffix={"QUESTION_PAPER":"qp", "MARK_SCHEME":"ms"}.get(doc_type, slug(path.stem)[:20])
    doc_id=f"{exact_pair}_{suffix}" if exact_pair and suffix in {"qp","ms"} else f"{level.lower()}_{slug(path.stem)}"
    return {"document_id":doc_id, "level":level, "qualification":level, "subject_code":subject_code, "document_type":doc_type, "year":year, "session":session, "paper_number":component, "component":component, "variant":None, "syllabus_valid_from":None, "syllabus_valid_to":None, "authority_level":None, "exact_pair_id":exact_pair}
