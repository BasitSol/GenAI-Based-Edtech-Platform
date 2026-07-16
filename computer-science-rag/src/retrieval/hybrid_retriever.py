from __future__ import annotations
from pathlib import Path
from src.core import ROOT
from src.indexing.bm25_index import BM25Index
from src.indexing.chroma_index import ChromaIndex
from src.indexing.metadata_store import MetadataStore
from .authority_controller import filter_and_sort
from .query_classifier import classify
from src.core import tokens

EXAM_QUERY_STOP={'answer','question','state','what','meant','define','describe','explain','give','identify','write','complete','calculate','show','working','marks','mark','page'}

def _focus_exam_query(text:str)->str:
    focused=[term for term in tokens(text) if term not in EXAM_QUERY_STOP and not term.isdigit()]
    return ' '.join(focused) or text
class HybridRetriever:
    def __init__(self,processed=ROOT/'data_processed'):
        self.bm25=BM25Index.load(processed/'indexes'/'bm25'/'index.json'); self.by_id={c['chunk_id']:c for c in self.bm25.chunks}; self.chroma=ChromaIndex(processed/'indexes'/'chroma'); self.store=MetadataStore(processed/'databases'/'metadata.sqlite')
        self.patterns=[item for item in self.bm25.chunks if item.get('document_type')=='MARKING_PATTERN']
        self.pattern_bm25=BM25Index(self.patterns)
    def retrieve(self,query,level=None,exam_year=None,result_limit=8):
        route=classify(query,level,exam_year); metadata={k:route.get(k) for k in ('level','year','session','component','question_number') if route.get(k) is not None}
        semantic_query=query
        paper_identity_complete=all(route.get(key) is not None for key in ('subject_code','year','session','component'))
        exact=self.store.exact_chunks(**metadata) if paper_identity_complete else []
        if route['intent']=='EXAMINER_FEEDBACK' and route.get('question_number'):
            exact=self.store.exact_chunks(level=route.get('level'),question_number=route['question_number'])
        # Parsing a subquestion is best-effort; retain paper-level evidence when
        # the PDF layout prevented a subquestion boundary from being extracted.
        if not exact and metadata.get('question_number'):
            exact=self.store.exact_chunks(**{k:v for k,v in metadata.items() if k!='question_number'})
        # Fetch a wider candidate set before level filtering; otherwise results
        # from the other qualification can crowd out the requested curriculum.
        bm=self.bm25.search(query,200); dense_ids=self.chroma.search(query,200)['ids'][0]; dense=[self.by_id[x] for x in dense_ids if x in self.by_id]
        if route.get('level'):
            bm=[x for x in bm if x.get('level')==route['level']]
            dense=[x for x in dense if x.get('level')==route['level']]
        if paper_identity_complete:
            identity=('subject_code','year','session','component')
            bm=[x for x in bm if all(str(x.get(key))==str(route.get(key)) for key in identity)]
            dense=[x for x in dense if all(str(x.get(key))==str(route.get(key)) for key in identity)]
            if route.get('question_number'):
                question=route['question_number']
                bm=[x for x in bm if x.get('question_number') == question or str(x.get('question_number','')).startswith(f'{question}(')]
                dense=[x for x in dense if x.get('question_number') == question or str(x.get('question_number','')).startswith(f'{question}(')]
        dense_bm25_overlap_at_10=len({item['chunk_id'] for item in dense[:10]} & {item['chunk_id'] for item in bm[:10]})/10
        broad_candidates=dense+bm
        bm=bm[:30]; dense=dense[:30]
        # Exact evidence leads. The remaining candidates use reciprocal-rank fusion.
        scores={}
        for ranking in (dense,bm):
            for rank,chunk in enumerate(ranking,1): scores[chunk['chunk_id']]=scores.get(chunk['chunk_id'],0)+1/(60+rank)
        fused=list(exact)
        seen={x['chunk_id'] for x in fused}
        for chunk_id in sorted(scores,key=scores.get,reverse=True):
            if chunk_id not in seen: fused.append(self.by_id[chunk_id]); seen.add(chunk_id)
        exact_scheme=paper_identity_complete and any(c['document_type']=='MARK_SCHEME' for c in exact)
        # When a referenced paper has no matching mark scheme, the paper text is
        # the question, not evidence for an answer. Retrieve curriculum support
        # using the extracted question text and keep the QP chunk for identity.
        # This is deliberately separate from the paper-identity-filtered search.
        if paper_identity_complete and route['intent']=='EXAM_ANSWER' and not exact_scheme:
            question_chunks=[item for item in exact if item.get('document_type')=='QUESTION_PAPER']
            support_query=_focus_exam_query(' '.join(item.get('text','') for item in question_chunks) or query)
            semantic_query=support_query
            support_bm=self.bm25.search(support_query,120)
            support_dense_ids=self.chroma.search(support_query,120)['ids'][0]
            support_dense=[self.by_id[item] for item in support_dense_ids if item in self.by_id]
            support_types={'TEXTBOOK','SYLLABUS','MARKING_PATTERN'}
            curriculum_scores={}; patterns=[]; support_seen=set()
            for ranking in (support_bm,support_dense):
                for rank,candidate in enumerate(ranking,1):
                    if candidate.get('document_type') not in support_types: continue
                    if route.get('level') and candidate.get('level')!=route['level']: continue
                    if candidate.get('document_type')=='MARKING_PATTERN':
                        if candidate['chunk_id'] not in support_seen: patterns.append(candidate); support_seen.add(candidate['chunk_id'])
                    else:
                        curriculum_scores[candidate['chunk_id']]=curriculum_scores.get(candidate['chunk_id'],0)+1/(60+rank)
            curriculum=[self.by_id[item] for item in sorted(curriculum_scores,key=curriculum_scores.get,reverse=True)]
            pattern_ranked=self.pattern_bm25.search(support_query,40)
            family=str(route.get('component') or '')[:1]
            pattern_fallback=[item for item in self.patterns if (not route.get('level') or item.get('level')==route['level']) and (not family or str(item.get('component') or '').startswith(family))]
            for candidate in pattern_ranked+pattern_fallback:
                if route.get('level') and candidate.get('level')!=route['level']: continue
                if candidate['chunk_id'] in support_seen: continue
                patterns.append(candidate); support_seen.add(candidate['chunk_id'])
            # Curriculum supplies factual evidence. Patterns supply assessment
            # structure only and are kept as a distinct, lower-authority group.
            fused.extend(curriculum[:20]); fused.extend(patterns[:8])
        allowed=filter_and_sort(fused,route['intent'],exact_scheme)
        allowed_ids={item['chunk_id'] for item in allowed}
        for item in filter_and_sort(broad_candidates,route['intent'],exact_scheme):
            if item['chunk_id'] not in allowed_ids:
                allowed.append(item); allowed_ids.add(item['chunk_id'])
        results=allowed[:result_limit]
        if route['intent']=='EXAM_ANSWER' and not exact_scheme:
            for required_types in ({'QUESTION_PAPER'},{'TEXTBOOK','SYLLABUS'},{'MARKING_PATTERN'}):
                if any(item['document_type'] in required_types for item in results): continue
                candidate=next((item for item in allowed if item['document_type'] in required_types),None)
                if candidate:
                    if len(results)>=result_limit: results[-1]=candidate
                    else: results.append(candidate)
        if route['intent'] in {'DEFINITION','CONCEPT_EXPLANATION','COMPARISON'}:
            for document_type in ('TEXTBOOK','SYLLABUS'):
                if any(item['document_type']==document_type for item in results): continue
                candidate=next((item for item in allowed if item['document_type']==document_type),None)
                if candidate:
                    if len(results)>=result_limit: results[-1]=candidate
                    else: results.append(candidate)
        return {'route':route,'chunks':results,'semantic_query':semantic_query,'exact_mark_scheme_available':exact_scheme,'retrieval_debug':{'metadata_hits':len(exact),'dense_hits':len(dense),'bm25_hits':len(bm),'dense_bm25_overlap_at_10':round(dense_bm25_overlap_at_10,3),'dense_index_error':self.chroma.last_error,'semantic_query':semantic_query}}
