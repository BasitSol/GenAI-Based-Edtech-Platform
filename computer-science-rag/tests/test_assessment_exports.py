"""Ensure assessment exports are local, reproducible files with both formats."""
from __future__ import annotations

from xml.etree import ElementTree
from zipfile import ZipFile

import pdfplumber

from backend.module2_generation.assessments.exports import export_assessment, structured_blocks, structured_markdown


def _assessment() -> dict:
    return {"id": 7, "topic": "Binary search", "difficulty": "INTERMEDIATE", "type": "quiz", "status": "approved",
            "content": {"title": "Binary Search Quiz", "instructions": "Answer all questions.", "questions": [
                {"number": 1, "question": "Explain the sorted-data prerequisite.", "marks": 3,
                 "question_type": "MCQ", "options": ["Sorted", "Unsorted", "Random", "Empty"], "correct_option": "Sorted",
                 "model_answer": "Binary search requires sorted data.", "rubric": ["states sorted-data prerequisite"],
                 "citations": [{"chunk_id": "book_1"}]},
                {"number": 2, "question": (
                    "Complete the table by identifying a suitable data type.\n\n"
                    "| Field | Data type |\n"
                    "|---|---|\n"
                    "| SongNumber | |\n"
                    "| Title | |\n"
                    "| Recorded | |\n"
                    "| Minutes | |"
                ), "marks": 2, "question_type": "SHORT_ANSWER", "options": [], "correct_option": "",
                 "model_answer": "| Field | Data type |\n|---|---|\n| SongNumber | Integer |\n| Title | Text |",
                 "rubric": ["one valid type", "second valid type"], "citations": [{"chunk_id": "book_2"}]},
                {"number": 3, "question": "Complete the truth table.", "marks": 2,
                 "question_type": "SHORT_ANSWER", "options": [], "correct_option": "",
                 "model_answer": "A B C | X\n0 0 0 | 0\n0 0 1 | 1\n1 1 0 | 0\n1 1 1 | 1",
                 "rubric": ["correct first rows", "correct final rows"], "citations": [{"chunk_id": "book_3"}]},
            ]}}


def test_assessment_exports_create_docx_and_pdf(monkeypatch, tmp_path):
    monkeypatch.setattr("backend.module2_generation.assessments.exports.export_directory", lambda: tmp_path)
    docx = export_assessment(_assessment(), "docx", include_solutions=True)
    pdf = export_assessment(_assessment(), "pdf", include_solutions=True)
    assert docx.exists() and docx.read_bytes()[:2] == b"PK"
    assert pdf.exists() and pdf.read_bytes()[:4] == b"%PDF"
    with ZipFile(docx) as archive:
        document_xml = archive.read("word/document.xml")
        numbering_xml = archive.read("word/numbering.xml")
        header_xml = archive.read("word/header1.xml")
        footer_xml = archive.read("word/footer1.xml")
        settings_xml = archive.read("word/settings.xml")
    assert b"Correct answer: A. Sorted" in document_xml
    assert b"upperLetter" in numbering_xml and b"bullet" in numbering_xml and b"decimal" in numbering_xml
    numbering = ElementTree.fromstring(numbering_xml)
    tags = [element.tag.rsplit("}", 1)[-1] for element in numbering]
    assert max(index for index, tag in enumerate(tags) if tag == "abstractNum") < min(
        index for index, tag in enumerate(tags) if tag == "num"
    )
    assert b"COMPUTER SCIENCE" in header_xml
    assert b"PAGE" in footer_xml and b"NUMPAGES" in footer_xml
    assert b"updateFields" in settings_xml
    assert b"<w:tbl>" in document_xml
    assert b"SongNumber" in document_xml and b"Data type" in document_xml
    assert b"<w:t>A</w:t>" in document_xml and b"<w:t>X</w:t>" in document_xml
    assert b"|---|---|" not in document_xml
    assert b"<w:tblGrid>" in document_xml and b"<w:tcMar>" in document_xml and b'w:w="9360"' in document_xml
    assert b"TableGrid" in document_xml and b"<w:keepLines" in document_xml
    with pdfplumber.open(pdf) as exported_pdf:
        pdf_text = "\n".join(page.extract_text() or "" for page in exported_pdf.pages)
    assert "SongNumber" in pdf_text and "Data type" in pdf_text
    assert "A B C X" in pdf_text
    assert "|---|---|" not in pdf_text
    assert "Computer Science Assessment" in pdf_text and "Page 1" in pdf_text


def test_compact_truth_table_is_canonicalised_for_all_presentations():
    value = "A B C | X\n0 0 0 | 0\n0 0 1 | 1\n1 1 0 | 0\n1 1 1 | 1"
    blocks = structured_blocks(value)
    assert blocks == [{"type": "table", "rows": [
        ["A", "B", "C", "X"],
        ["0", "0", "0", "0"],
        ["0", "0", "1", "1"],
        ["1", "1", "0", "0"],
        ["1", "1", "1", "1"],
    ]}]
    markdown = structured_markdown(value)
    assert "| A | B | C | X |" in markdown
    assert "| --- | --- | --- | --- |" in markdown


def test_markdown_lists_are_preserved_as_native_list_blocks():
    value = "Remember:\n\n- Validate input\n- Store the result\n\n1. Read\n2. Process"
    blocks = structured_blocks(value)
    assert blocks == [
        {"type": "paragraph", "text": "Remember:"},
        {"type": "list", "ordered": False, "items": ["Validate input", "Store the result"]},
        {"type": "list", "ordered": True, "items": ["Read", "Process"]},
    ]
    assert structured_markdown(value) == (
        "Remember:\n\n- Validate input\n- Store the result\n\n1. Read\n2. Process"
    )
