from src.retrieval.context_builder import extract_relevant_text
from src.retrieval.hybrid_retriever import _rrf
from src.retrieval.query_classifier import classify
from src.retrieval.reranker import rerank_with_debug


def test_structural_classification_for_problematic_queries(monkeypatch):
    monkeypatch.setenv("INTENT_CLASSIFIER_MODE", "deterministic")
    assert classify("Explain how binary search works.")["category"] == "PROGRAMMING"
    assert classify("Write an SQL query to display students whose Mark is greater than 70.")["category"] == "SQL"
    assert classify("Explain the difference between a compiler and an interpreter.")["category"] == "COMPARISON"


def test_rrf_rewards_items_found_by_both_retrievers():
    scores = _rrf([[{"chunk_id": "both"}, {"chunk_id": "sparse"}],
                   [{"chunk_id": "both"}, {"chunk_id": "dense"}]])
    assert scores["both"] > scores["sparse"] and scores["both"] > scores["dense"]


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
