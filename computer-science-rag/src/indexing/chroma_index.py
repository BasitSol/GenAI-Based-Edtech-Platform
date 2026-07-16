from __future__ import annotations
import hashlib
import json
import os
import sqlite3
from pathlib import Path
from src.indexing.embedding_service import create_embedding_service
from src.indexing.contextual_enrichment import CONTEXTUAL_ENRICHMENT_VERSION, contextual_enrichment_enabled, contextualized_text
class ChromaIndex:
    """Persistent Chroma collection with explicit, reproducible embeddings."""
    def __init__(self,path:Path,collection='curriculum_chunks'):
        import chromadb
        self.path=path
        self.client=chromadb.PersistentClient(path=str(path))
        self.embedder=create_embedding_service()
        enrichment=f'{contextual_enrichment_enabled()}:{CONTEXTUAL_ENRICHMENT_VERSION}'
        fingerprint=hashlib.sha256(f'{self._cache_identity()}:{enrichment}'.encode('utf-8')).hexdigest()[:12]
        self.collection_name=f'{collection}_{fingerprint}'
        self.collection=self.client.get_or_create_collection(self.collection_name,embedding_function=None)
        self.last_error=None
        self.cache=sqlite3.connect(path/'embedding_cache.sqlite'); self.cache.execute('CREATE TABLE IF NOT EXISTS embeddings (cache_key TEXT PRIMARY KEY, model TEXT NOT NULL, vector TEXT NOT NULL)'); self.cache.commit()
    def _cache_key(self,text):
        return hashlib.sha256(f'{self._cache_identity()}\0{text}'.encode('utf-8')).hexdigest()
    def _cache_identity(self):
        return getattr(self.embedder,'cache_identity',f'{self.embedder.provider}:{self.embedder.model}')
    @staticmethod
    def indexable_chunks(chunks):
        """Keep semantic candidates; deterministic/style/expanded records use other routes."""
        return [c for c in chunks if c.get('content_type')!='PARENT_CONTEXT' and c.get('document_type') not in {'MARK_SCHEME','MARKING_PATTERN'}]
    def _embed_cached(self,texts):
        keys=[self._cache_key(text) for text in texts]; cached={row[0]:json.loads(row[1]) for row in self.cache.execute(f"SELECT cache_key, vector FROM embeddings WHERE cache_key IN ({','.join('?' for _ in keys)})",keys)} if keys else {}
        missing=[text for key,text in zip(keys,texts) if key not in cached]
        if missing:
            if os.getenv('LOCAL_MODEL_MEMORY_MODE','sequential').lower()=='sequential':
                # A low-RAM CPU cannot retain BGE-M3 and the cross-encoder
                # simultaneously. Release reranker weights before embedding a
                # genuinely new query; cache hits do not trigger model swaps.
                try:
                    from src.retrieval.reranker import release_reranker_model
                    release_reranker_model()
                except ImportError:
                    pass
            vectors=self.embedder.embed_many(missing)
            for text,vector in zip(missing,vectors):
                key=self._cache_key(text); cached[key]=vector; self.cache.execute('INSERT OR REPLACE INTO embeddings VALUES (?,?,?)',(key,self._cache_identity(),json.dumps(vector)))
            self.cache.commit()
        return [cached[key] for key in keys]
    def prepare_embeddings(self,chunks):
        """Resolve every vector before replacement, committing resumable local batches."""
        vectors=[]
        cache_batch=max(1,int(os.getenv('EMBEDDING_CACHE_BATCH_SIZE','16')))
        for offset in range(0,len(chunks),cache_batch):
            vectors.extend(self._embed_cached([contextualized_text(c) for c in chunks[offset:offset+cache_batch]]))
        return vectors
    def rebuild(self,chunks,vectors=None):
        vectors=vectors if vectors is not None else self.prepare_embeddings(chunks)
        try: self.client.delete_collection(self.collection.name)
        except Exception: pass
        self.collection=self.client.get_or_create_collection(self.collection.name,embedding_function=None)
        for offset in range(0,len(chunks),250):
            batch=chunks[offset:offset+250]
            self.collection.add(ids=[c['chunk_id'] for c in batch],documents=[c['text'] for c in batch],embeddings=vectors[offset:offset+len(batch)],metadatas=[{'document_id':c['document_id'],'document_type':c['document_type'],'level':c['level'],'page_start':c['page_start']} for c in batch])
    def search(self,query,k=15):
        try:
            self.last_error=None
            # Query vectors share the same model-keyed persistent cache as corpus
            # vectors. Repeated evaluation and follow-ups therefore avoid paid,
            # network-bound embedding calls without changing vector spaces.
            vector=self._embed_cached([query])[0]
            return self.collection.query(query_embeddings=[vector],n_results=k,include=['documents','metadatas'])
        except Exception as error:
            # A changed embedding model requires a Chroma rebuild. BM25 remains
            # available so a user gets a useful, observable degraded response.
            self.last_error=f"Dense index unavailable: {error}"
            return {'ids':[[]]}
