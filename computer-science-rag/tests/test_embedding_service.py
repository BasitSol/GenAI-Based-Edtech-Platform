import pytest
from src.indexing.embedding_service import HashEmbeddingService, create_embedding_service

def test_hash_embedding_is_normalized_and_stable():
    service=HashEmbeddingService()
    assert service.embed('compiler') == service.embed('compiler')
    assert len(service.embed('compiler')) == 384

def test_no_key_does_not_silently_change_embedding_model(monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.setenv('EMBEDDING_MODEL','text-embedding-3-small')
    service=create_embedding_service()
    assert service.provider == 'openai'
    assert service.model == 'text-embedding-3-small'
    with pytest.raises(RuntimeError,match='OPENAI_API_KEY'):
        service.embed('compiler')

def test_hash_fallback_requires_explicit_model(monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.setenv('EMBEDDING_MODEL','local-hash-baseline')
    assert create_embedding_service().provider == 'local_hash_fallback'
