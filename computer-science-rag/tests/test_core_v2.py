import importlib
import os

from backend.module1_rag.indexing.embedding_service import EmbeddingUsage, OpenAIEmbeddingService, create_embedding_service


def test_importing_core_does_not_load_dotenv(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    import backend.shared.core
    importlib.reload(backend.shared.core)
    assert "OPENAI_API_KEY" not in os.environ


def test_hash_embeddings_are_test_only(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL", "local-hash-baseline")
    try:
        create_embedding_service()
        raised = False
    except RuntimeError:
        raised = True
    assert raised
    assert create_embedding_service(allow_test_provider=True).provider == "test"


def test_embedding_cache_is_model_and_text_addressed(tmp_path):
    class Embeddings:
        calls = 0
        def create(self, model, input, encoding_format):
            self.calls += 1
            data = [type("Item", (), {"index": index, "embedding": [float(len(text)), 1.0]}) for index, text in enumerate(input)]
            return type("Response", (), {"data": data, "usage": type("Usage", (), {"prompt_tokens": 4})()})
    service = OpenAIEmbeddingService.__new__(OpenAIEmbeddingService)
    service.model, service.client, service.usage = "text-embedding-test", type("Client", (), {"embeddings": Embeddings()})(), EmbeddingUsage()
    service.cache_path = tmp_path / "cache.sqlite"
    import sqlite3
    with sqlite3.connect(service.cache_path) as connection:
        connection.execute("CREATE TABLE embeddings(cache_key TEXT PRIMARY KEY, model TEXT NOT NULL, vector TEXT NOT NULL)")
    first = service.embed_many(["same text", "same text"])
    second = service.embed_many(["same text"])
    assert first == second * 2
    assert service.client.embeddings.calls == 1
    assert service.usage.cache_hits == 1 and service.usage.cache_misses == 1
