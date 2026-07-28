from backend.module1_rag.retrieval.context_builder import extract_relevant_text
from backend.module1_rag.indexing.chroma_index import ChromaIndex
from backend.module1_rag.retrieval.hybrid_retriever import _rrf
from backend.module1_rag.retrieval.query_classifier import classify
from backend.module1_rag.retrieval.reranker import rerank_with_debug


def test_structural_classification_for_problematic_queries(monkeypatch):
    monkeypatch.setenv("INTENT_CLASSIFIER_MODE", "deterministic")
    assert classify("Explain how binary search works.")["category"] == "PROGRAMMING"
    assert classify("Write an SQL query to display students whose Mark is greater than 70.")["category"] == "SQL"
    assert classify("Explain the difference between a compiler and an interpreter.")["category"] == "COMPARISON"


def test_rrf_rewards_items_found_by_both_retrievers():
    scores = _rrf([[{"chunk_id": "both"}, {"chunk_id": "sparse"}],
                   [{"chunk_id": "both"}, {"chunk_id": "dense"}]])
    assert scores["both"] > scores["sparse"] and scores["both"] > scores["dense"]


def test_chroma_search_combines_level_and_document_type_with_and_filter():
    """Chroma rejects multiple top-level metadata fields; use one $and clause."""
    seen = {}

    class Embedder:
        def embed_many(self, _): return [[0.1, 0.2]]

    class Collection:
        def query(self, **kwargs):
            seen.update(kwargs)
            return {"ids": [["chunk"]], "distances": [[0.1]]}

    index = object.__new__(ChromaIndex)
    index.embedder, index.collection = Embedder(), Collection()
    hits = index.search("binary search", 4, {"level": "O_LEVEL", "document_type": "QUESTION_PAPER"})
    assert seen["where"] == {"$and": [{"level": "O_LEVEL"}, {"document_type": "QUESTION_PAPER"}]}
    assert hits[0]["chunk_id"] == "chunk"


class FakeCrossEncoder:
    def predict(self, pairs, **kwargs):
        return [0.9 if "binary search" in passage else 0.1 for _, passage in pairs]


def test_cross_encoder_controls_final_order():
    chunks = [{"chunk_id": "wrong", "text": "linear scan"},
              {"chunk_id": "right", "text": "binary search halves sorted data"}]
    ranked, debug = rerank_with_debug("binary search", chunks, 2, scorer=FakeCrossEncoder())
    assert [item["chunk_id"] for item in ranked] == ["right", "wrong"]
    assert not debug["fallback"]


def test_extractive_compression_keeps_relevant_passage(monkeypatch):
    monkeypatch.setenv("EXTRACTIVE_COMPRESSION_ENABLED", "true")
    text = ("Storage is non-volatile. " * 30) + "\n\nBinary search repeatedly discards half of a sorted list.\n\n" + ("Networks use protocols. " * 30)
    selected, compressed = extract_relevant_text("How does binary search work?", {"text": text, "document_type": "TEXTBOOK"}, 400)
    assert compressed and "discards half" in selected and len(selected) <= 400
