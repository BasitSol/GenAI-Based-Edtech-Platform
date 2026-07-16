from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.core import ROOT,read_jsonl
import json
import os
from src.indexing.bm25_index import BM25Index
from src.indexing.metadata_store import MetadataStore
from src.indexing.chroma_index import ChromaIndex
from src.indexing.contextual_enrichment import CONTEXTUAL_ENRICHMENT_VERSION, contextual_enrichment_enabled
if __name__=='__main__':
    processed=ROOT/'data_processed'; chunks=read_jsonl(processed/'chunks'/'all_chunks.jsonl') + read_jsonl(processed/'chunks'/'marking_patterns.jsonl')
    if not chunks: raise SystemExit('No chunks found. Run scripts/build_corpus.py then python -m src.chunking.pipeline.')
    chroma=ChromaIndex(processed/'indexes'/'chroma')
    dense_chunks=chroma.indexable_chunks(chunks)
    # Preflight all dense vectors first. Missing credentials or an API failure
    # cannot leave BM25/metadata newer than a stale or deleted vector index.
    vectors=chroma.prepare_embeddings(dense_chunks)
    BM25Index(chunks).save(processed/'indexes'/'bm25'/'index.json'); MetadataStore().rebuild(processed/'manifests'/'documents_manifest.csv',processed/'chunks'/'all_chunks.jsonl'); chroma.rebuild(dense_chunks,vectors)
    cache_count=chroma.cache.execute('SELECT COUNT(*) FROM embeddings WHERE model=?',(chroma._cache_identity(),)).fetchone()[0]
    report={'chunks':len(chunks),'dense_chunks':len(dense_chunks),'bm25':'built','chroma':'built','chroma_collection':chroma.collection_name,'metadata':'built','embedding_provider':chroma.embedder.provider,'embedding_model':chroma.embedder.model,'embedding_max_length':getattr(chroma.embedder,'max_length',None),'contextual_enrichment':contextual_enrichment_enabled(),'contextual_enrichment_version':CONTEXTUAL_ENRICHMENT_VERSION if contextual_enrichment_enabled() else None,'reranker_enabled':os.getenv('RERANKER_ENABLED','true').lower() in {'1','true','yes','on'},'reranker_model':os.getenv('RERANKER_MODEL','BAAI/bge-reranker-base'),'reranker_max_length':int(os.getenv('RERANKER_MAX_LENGTH','512')),'reranker_candidates':int(os.getenv('RERANKER_CANDIDATES','24')),'extractive_compression':os.getenv('EXTRACTIVE_COMPRESSION_ENABLED','true').lower() in {'1','true','yes','on'},'cached_embeddings':cache_count}
    (processed/'manifests'/'index_manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8'); print(report)
