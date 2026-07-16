"""Deterministic-first workflow; intentionally no LangGraph dependency until measured."""
from src.memory.followup_rewriter import rewrite_followup
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import rerank_with_debug
from src.retrieval.context_builder import build_context
from src.core import tokens
STOP={'what','when','where','which','that','this','with','from','into','about','explain','define','answer','question','computer','science','using'}
def retrieve(query, level=None, exam_year=None, conversation_state=None, maximum_chunks=6, retriever=None):
    standalone=rewrite_followup(query,conversation_state or {})
    import os
    retriever=retriever or HybridRetriever()
    candidate_count=max(maximum_chunks,int(os.getenv('RERANKER_CANDIDATES','24')))
    result=retriever.retrieve(standalone,level,exam_year,result_limit=candidate_count)
    semantic_query=result.get('semantic_query') or standalone
    route=result['route']
    identity_keys=('subject_code','year','session','component')
    identity_complete=all(route.get(key) is not None for key in identity_keys)
    exact_identity=[]
    if identity_complete:
        exact_identity=[chunk for chunk in result['chunks'] if all(str(chunk.get(key))==str(route.get(key)) for key in identity_keys)]
    exact_ids={chunk['chunk_id'] for chunk in exact_identity}
    semantic_candidates=[chunk for chunk in result['chunks'] if chunk['chunk_id'] not in exact_ids]
    deterministic_complete=bool(exact_identity) and (result.get('exact_mark_scheme_available') or route.get('intent')!='EXAM_ANSWER')
    high_retriever_agreement=result['retrieval_debug'].get('dense_bm25_overlap_at_10',0)>=float(os.getenv('RERANKER_BYPASS_OVERLAP','0.5'))
    if deterministic_complete:
        semantic=[]
        rerank_debug={'reranker':'bypassed_exact_identity','reranker_candidates':0,'reranker_error':None}
    elif os.getenv('RERANKER_CONDITIONAL','true').lower() in {'1','true','yes','on'} and high_retriever_agreement:
        semantic=semantic_candidates[:maximum_chunks]
        rerank_debug={'reranker':'bypassed_high_retriever_agreement','reranker_candidates':len(semantic_candidates),'reranker_error':None}
    else:
        semantic,rerank_debug=rerank_with_debug(semantic_query,semantic_candidates,maximum_chunks)
    ranked=(exact_identity+semantic)[:maximum_chunks]
    if route.get('intent')=='EXAM_ANSWER' and not result.get('exact_mark_scheme_available'):
        # Preserve all three roles required by the plan: exact question identity,
        # curriculum facts, and a style-only assessment pattern.
        for required_types in ({'QUESTION_PAPER'},{'TEXTBOOK','SYLLABUS'},{'MARKING_PATTERN'}):
            if any(item.get('document_type') in required_types for item in ranked): continue
            candidate=next((item for item in result['chunks'] if item.get('document_type') in required_types),None)
            if candidate:
                if len(ranked)>=maximum_chunks: ranked[-1]=candidate
                else: ranked.append(candidate)
        ordered=[]
        for required_types in ({'QUESTION_PAPER'},{'TEXTBOOK','SYLLABUS'},{'MARKING_PATTERN'}):
            candidate=next((item for item in ranked if item.get('document_type') in required_types),None)
            if candidate and candidate not in ordered: ordered.append(candidate)
        ordered.extend(item for item in ranked if item not in ordered)
        ranked=ordered[:maximum_chunks]
    query_terms={term for term in tokens(semantic_query) if len(term)>2 and term not in STOP}
    overlap=max((len(query_terms & set(tokens(chunk['text'])))/max(1,len(query_terms)) for chunk in ranked),default=0.0)
    exact_reference=identity_complete
    requested_year=result['route'].get('year')
    unavailable_year=bool(requested_year) and not exact_identity and not any(str(chunk.get('year'))==str(requested_year) for chunk in ranked)
    disallowed_request=any(phrase in standalone.lower() for phrase in ('complete source code','proprietary source code'))
    factual_chunks=[item for item in ranked if item.get('document_type') not in {'QUESTION_PAPER','MARKING_PATTERN'}]
    factual_overlap=max((len(query_terms & set(tokens(item.get('text',''))))/max(1,len(query_terms)) for item in factual_chunks),default=0.0)
    if route.get('intent')=='EXAM_ANSWER' and not result.get('exact_mark_scheme_available'):
        sufficient=bool(factual_chunks) and factual_overlap>=.12 and not unavailable_year and not disallowed_request
    else:
        sufficient=(exact_reference or overlap>=.12) and not unavailable_year and not disallowed_request
    expanded=[]
    if sufficient:
        expanded.extend(ranked)
        for chunk in ranked:
            parent_id=chunk.get('parent_chunk_id')
            if parent_id and parent_id in retriever.by_id and all(item['chunk_id']!=parent_id for item in expanded): expanded.append(retriever.by_id[parent_id])
    result["chunks"]=build_context(expanded,semantic_query,maximum_chunks=maximum_chunks)
    result['retrieval_debug'].update(rerank_debug)
    result['retrieval_debug']['compressed_chunks']=sum(1 for item in result['chunks'] if item.get('context_compressed'))
    result['retrieval_debug']['context_chars']=sum(len(item.get('text','')) for item in result['chunks'])
    result['retrieval_debug']['lexical_sufficiency']=round(overlap,3); result['retrieval_debug']['context_sufficient']=sufficient
    result['retrieval_debug']['factual_evidence_overlap']=round(factual_overlap,3)
    result["standalone_query"]=standalone
    return result
