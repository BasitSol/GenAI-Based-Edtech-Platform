from __future__ import annotations
def classify_document(filename: str, first_pages_text: str, folder_hint: str = "") -> str:
    name=filename.lower().replace("_", "-")
    value=f"{filename} {first_pages_text} {folder_hint}".lower()
    # Filenames and configured folders are stronger than incidental phrases in a PDF.
    if "mark-scheme" in name or "mark scheme" in name: return "MARK_SCHEME"
    if "examiner-report" in name or "examiner report" in name: return "EXAMINER_REPORT"
    if "syllabus" in name: return "SYLLABUS"
    if "book" in name: return "TEXTBOOK"
    if "examiner report" in value or "principal examiner" in value: return "EXAMINER_REPORT"
    if "mark scheme" in value or "maximum raw mark" in value: return "MARK_SCHEME"
    if "question paper" in value or "answer all questions" in value or "past paper" in value: return "QUESTION_PAPER"
    if "syllabus" in value: return "SYLLABUS"
    if "book" in value or "computer science" in value: return "TEXTBOOK"
    return "UNKNOWN"
