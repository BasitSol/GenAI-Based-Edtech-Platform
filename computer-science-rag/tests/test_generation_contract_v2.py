import pytest

from src.generation.answer_generator import _canonical_citations, _parse_generation
from src.generation.verification import verify_response


CHUNK = {"document_id": "book", "page_start": 42, "chunk_id": "book_42_1", "document_type": "TEXTBOOK", "text": "Binary search needs sorted data."}


def test_structured_answer_separates_prose_from_sources():
    answer, citations = _parse_generation('{"answer_markdown":"```sql\\nSELECT * FROM Student;\\n```","source_keys":["S1"]}', {"S1": CHUNK})
    assert "S1" not in answer and citations[0]["chunk_id"] == "book_42_1"


def test_unknown_or_inline_source_key_is_rejected():
    with pytest.raises(ValueError):
        _parse_generation('{"answer_markdown":"Claim [S1]","source_keys":["S1"]}', {"S1": CHUNK})
    with pytest.raises(ValueError):
        _canonical_citations(["S2"], {"S1": CHUNK})


def test_verification_checks_identity_but_does_not_fake_faithfulness():
    citation = {"document_id": "book", "page": 42, "chunk_id": "book_42_1"}
    result = verify_response("Binary search needs sorted data.", [CHUNK], {"needs_citations": True}, True, [citation], True)
    assert result["passed"]
    assert result["semantic_faithfulness"]["status"] == "NOT_MEASURED"
