from src.indexing.contextual_enrichment import contextualized_text
from src.indexing.chroma_index import ChromaIndex
from src.retrieval.context_builder import extract_relevant_text
from src.retrieval.reranker import rerank_with_debug


def test_contextual_embedding_view_preserves_original_text(monkeypatch):
    monkeypatch.setenv("CONTEXTUAL_ENRICHMENT_ENABLED", "true")
    chunk = {
        "text": "An interrupt is a signal sent to the processor.",
        "document_id": "a_level_book",
        "document_type": "TEXTBOOK",
        "level": "A_LEVEL",
        "subject_code": "9618",
        "page_start": 42,
        "content_type": "EXPLANATION",
    }
    embedded = contextualized_text(chunk)
    assert "Qualification: Cambridge International A Level Computer Science" in embedded
    assert "Source type: Textbook" in embedded
    assert embedded.endswith(chunk["text"])
    assert chunk["text"] == "An interrupt is a signal sent to the processor."


class _FakeCrossEncoder:
    def predict(self, pairs, **kwargs):
        return [0.1 if "unrelated" in passage else 0.9 for _, passage in pairs]


def test_cross_encoder_scores_control_semantic_order(monkeypatch):
    monkeypatch.setenv("RERANKER_ENABLED", "true")
    chunks = [
        {"chunk_id": "weak", "text": "unrelated material", "authority_level": 5},
        {"chunk_id": "strong", "text": "relevant interrupt handling", "authority_level": 4},
    ]
    ranked, debug = rerank_with_debug("interrupt handling", chunks, 2, scorer=_FakeCrossEncoder())
    assert [item["chunk_id"] for item in ranked] == ["strong", "weak"]
    assert ranked[0]["reranker_score"] == 0.9
    assert debug["reranker_error"] is None
    assert debug["reranker_score_cache_hit"] is False


def test_extractive_compression_selects_relevant_passage(monkeypatch):
    monkeypatch.setenv("EXTRACTIVE_COMPRESSION_ENABLED", "true")
    irrelevant = "Storage devices retain data after power is removed. " * 18
    relevant = "An interrupt causes the processor to suspend the current task and execute an interrupt service routine."
    text = irrelevant + "\n\n" + relevant + "\n\n" + irrelevant
    selected, compressed = extract_relevant_text("How does a processor handle an interrupt?", {"text": text, "document_type": "TEXTBOOK"}, 500)
    assert compressed
    assert "interrupt service routine" in selected
    assert len(selected) <= 500


def test_official_evidence_is_not_semantically_rewritten(monkeypatch):
    monkeypatch.setenv("EXTRACTIVE_COMPRESSION_ENABLED", "true")
    text = "official mark scheme line " * 100
    selected, compressed = extract_relevant_text("answer", {"text": text, "document_type": "MARK_SCHEME"}, 300)
    assert selected == text[:300]
    assert compressed


def test_dense_index_excludes_records_with_deterministic_routes():
    chunks = [
        {"chunk_id": "child", "document_type": "TEXTBOOK", "content_type": "EXPLANATION"},
        {"chunk_id": "parent", "document_type": "TEXTBOOK", "content_type": "PARENT_CONTEXT"},
        {"chunk_id": "scheme", "document_type": "MARK_SCHEME", "content_type": "MARK_SCHEME_ENTRY"},
        {"chunk_id": "pattern", "document_type": "MARKING_PATTERN", "content_type": "MARK_SCHEME_ENTRY"},
    ]
    assert [item["chunk_id"] for item in ChromaIndex.indexable_chunks(chunks)] == ["child"]
