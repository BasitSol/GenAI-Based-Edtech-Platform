from backend.module1_rag.ingestion.chunking.pipeline import _assessment_chunks, _textbook_chunks


DOCUMENT = {"document_id": "paper", "document_type": "QUESTION_PAPER", "level": "A_LEVEL",
            "subject_code": "9618", "year": "2024", "session": "MJ", "component": "11", "authority_level": "3"}


def test_question_boundaries_include_split_subparts_and_keep_pages():
    pages = [{"page_number": 2, "clean_text": "1\n(a) Explain virtual memory.\n[2]\n(b) State one benefit.\n[1]\n2\nDescribe a compiler.\n[2]"}]
    chunks = _assessment_chunks(DOCUMENT, pages, mark_scheme=False)
    assert [chunk["question_number"] for chunk in chunks] == ["1(a)", "1(b)", "2"]
    assert all(chunk["page_start"] == 2 for chunk in chunks)


def test_question_parser_ignores_printed_page_number():
    pages = [{"page_number": 2, "clean_text": "2\n9618/11/M/J/24\n[Turn over]\n1 Explain an interrupt.\n[2]"}]
    chunks = _assessment_chunks(DOCUMENT, pages, mark_scheme=False)
    assert [chunk["question_number"] for chunk in chunks] == ["1"]


def test_textbook_children_are_page_local_and_have_exact_parent():
    pages = [{"page_number": 10, "clean_text": "compiler " * 500}, {"page_number": 11, "clean_text": "interpreter " * 500}]
    chunks = _textbook_chunks({**DOCUMENT, "document_type": "TEXTBOOK", "year": ""}, pages)
    children = [item for item in chunks if item["content_type"] == "EXPLANATION"]
    assert all(("compiler" in item["text"]) == (item["page_start"] == 10) for item in children)
    assert all(item["parent_chunk_id"].endswith(f"page_{item['page_start']:04d}_parent") for item in children)
