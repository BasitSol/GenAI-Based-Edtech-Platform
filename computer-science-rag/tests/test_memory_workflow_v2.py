from pathlib import Path

from src.memory.conversation_store import ConversationStore
from src.memory.followup_rewriter import rewrite_followup
from src.workflows.rag_graph import retrieve


def test_standalone_question_does_not_inherit_previous_sql():
    state = {"recent_messages": [{"role": "user", "content": "Write SQL"}, {"role": "assistant", "content": "SELECT *"}]}
    assert rewrite_followup("Explain how binary search works.", state) == "Explain how binary search works."


def test_persistent_memory_is_bounded(tmp_path: Path):
    store, state = ConversationStore(tmp_path / "memory.sqlite"), None
    state = store.get(None)
    for index in range(7):
        store.record(state, f"Question {index}", f"Answer {index}", {"category": "THEORY"}, [])
    saved = store.get(state["conversation_id"])
    assert len(saved["recent_messages"]) <= 10 and saved["summary"]


class Metadata:
    def parent(self, _):
        return None


class CorrectiveFakeRetriever:
    metadata = Metadata()

    def __init__(self):
        self.calls = 0

    def retrieve(self, query, level, year, result_limit=24):
        self.calls += 1
        text = "irrelevant network material" if self.calls == 1 else "binary search sorted list middle discard half"
        return {"route": {"category": "PROGRAMMING"}, "chunks": [{"chunk_id": f"c{self.calls}", "text": text, "retrieval_text": text,
                 "document_id": "book", "document_type": "TEXTBOOK", "page_start": 1, "parent_chunk_id": None}],
                "exact_mark_scheme_available": False,
                "retrieval_debug": {"retrieval_plan": {"context_chunks": 4, "context_chars": 4000}}}


def test_corrective_workflow_retries_at_most_once(monkeypatch):
    monkeypatch.setenv("RERANKER_ENABLED", "false")
    fake = CorrectiveFakeRetriever()
    result = retrieve("Explain binary search sorted list", retriever=fake)
    assert fake.calls == 2
    assert result["retrieval_debug"]["retry_count"] == 1
    assert result["chunks"]
