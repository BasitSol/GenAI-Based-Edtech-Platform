from __future__ import annotations
from pathlib import Path
from evaluation.benchmark import approved_records, benchmark_status
from src.retrieval.hybrid_retriever import HybridRetriever
from src.workflows.rag_graph import retrieve

def _relevant(chunk:dict, gold:list[dict]) -> bool:
    for source in gold:
        if chunk['document_id'] != source.get('document_id'): continue
        if source.get('chunk_id'):
            if chunk.get('chunk_id')==source['chunk_id']: return True
            if source.get('parent_chunk_id') and chunk.get('chunk_id')==source['parent_chunk_id']: return True
            continue
        start=source.get('page_start',source.get('page')); end=source.get('page_end',start)
        if start is None or chunk.get('page_start') is None or start <= chunk['page_start'] <= end: return True
    return False

def _dcg(relevances:list[int]) -> float:
    import math
    return sum(value/math.log2(index+2) for index,value in enumerate(relevances))

def evaluate(dataset:Path, limit:int|None=None):
    records=approved_records(dataset); records=records[:limit] if limit else records; system=HybridRetriever(); rows=[]
    for record in records:
        if not record['answerable'] or not record.get('gold_curriculum_sources'): continue
        # Score the production retrieval stack, including contextual BGE-M3,
        # cross-encoder reranking and parent-context expansion. Previously this
        # stopped before the reranker, so the reranker upgrade was not measured.
        result=retrieve(record['question'],record.get('level'),record.get('exam_year'),maximum_chunks=10,retriever=system)
        relevances=[int(_relevant(chunk,record['gold_curriculum_sources'])) for chunk in result['chunks']]
        ranks=[index+1 for index,value in enumerate(relevances) if value]
        # One gold page may legitimately produce multiple relevant chunks.
        # The ideal ranking therefore has the same number of relevant results.
        ideal=[1]*min(10,max(1,sum(relevances)))
        expected_scheme=record.get('exact_mark_scheme_available')
        found_types={chunk['document_type'] for chunk in result['chunks']}
        rows.append({'id':record['id'],'recall_at_5':int(any(relevances[:5])),'recall_at_10':int(any(relevances[:10])),'reciprocal_rank':1/ranks[0] if ranks else 0.0,'ndcg_at_10':_dcg(relevances)/_dcg(ideal),'source_routing_correct':not record.get('expected_source_types') or set(record['expected_source_types']).issubset(found_types),'exact_scheme_correct':result['exact_mark_scheme_available'] if expected_scheme is True else None,'retrieved_chunk_ids':[chunk['chunk_id'] for chunk in result['chunks']]})
    def mean(field):
        values=[row[field] for row in rows if row[field] is not None]
        return sum(values)/len(values) if values else None
    return {'benchmark':benchmark_status(dataset),'scored_count':len(rows),'recall_at_5':mean('recall_at_5'),'recall_at_10':mean('recall_at_10'),'mrr':mean('reciprocal_rank'),'ndcg_at_10':mean('ndcg_at_10'),'source_routing_accuracy':mean('source_routing_correct'),'exact_scheme_retrieval_accuracy':mean('exact_scheme_correct'),'rows':rows}
