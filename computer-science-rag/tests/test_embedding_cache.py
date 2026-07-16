from pathlib import Path
from src.indexing.chroma_index import ChromaIndex

def test_embedding_cache_reuses_identical_text(tmp_path, monkeypatch):
    monkeypatch.setenv('EMBEDDING_MODEL','local-hash-baseline')
    index=ChromaIndex(tmp_path)
    first=index._embed_cached(['same text'])
    second=index._embed_cached(['same text'])
    assert first == second
    assert index.cache.execute('SELECT COUNT(*) FROM embeddings').fetchone()[0] == 1
