from __future__ import annotations
from pathlib import Path
import hashlib
import json
import os
from evaluation.benchmark import approved_records, benchmark_status
from evaluation.citation_eval import evaluate_answer as evaluate_citations
from src.generation.answer_generator import answer_question
from src.generation.grounding_checker import faithfulness
from src.core import ROOT, tokens

def _answer_cache_key(item:dict)->str:
    digest=hashlib.sha256()
    digest.update(json.dumps(item,sort_keys=True,ensure_ascii=False).encode('utf-8'))
    digest.update(os.getenv('GENERATOR_MODEL','gpt-4.1-mini').encode('utf-8'))
    manifest=ROOT/'data_processed/manifests/index_manifest.json'
    if manifest.exists(): digest.update(manifest.read_bytes())
    for folder in ('src/generation','src/retrieval','src/workflows'):
        for path in sorted((ROOT/folder).glob('*.py')):
            digest.update(path.name.encode('utf-8')); digest.update(path.read_bytes())
    return digest.hexdigest()

def _cached_answer(item:dict)->tuple[dict,bool]:
    cache_dir=ROOT/'evaluation/cache/answers'; cache_dir.mkdir(parents=True,exist_ok=True)
    path=cache_dir/f'{_answer_cache_key(item)}.json'
    if path.exists(): return json.loads(path.read_text(encoding='utf-8')),True
    answer=None; conversation_id=None
    for turn in item.get('conversation_turns',[item['question']]):
        answer=answer_question(turn,item.get('level'),item.get('exam_year'),conversation_id)
        conversation_id=answer['conversation_id']
    path.write_text(json.dumps(answer,ensure_ascii=False),encoding='utf-8')
    return answer,False

def _key_point_coverage(answer:str, points:list[str]) -> float|None:
    if not points: return None
    answer_terms=set(tokens(answer)); covered=0
    for point in points:
        terms={term for term in tokens(point) if len(term)>2}
        if terms and len(terms&answer_terms)/len(terms)>=.65: covered+=1
    return covered/len(points)

def evaluate(dataset:Path, limit:int|None=None):
    records=approved_records(dataset); records=records[:limit] if limit else records; rows=[]
    for item in records:
        if item.get('evaluation_scope')=='RETRIEVAL_ONLY': continue
        answer,cache_hit=_cached_answer(item)
        citations=evaluate_citations(answer,item)
        abstained=answer['answer_type']=='INSUFFICIENT_SOURCE'
        if abstained:
            citations={**citations,'citation_precision':None,'citation_coverage':None,'claim_citation_coverage':None}
        rows.append({'id':item['id'],'answer':answer['answer'],'answer_type':answer['answer_type'],'status_matches':not item.get('expected_answer_type') or answer['answer_type']==item['expected_answer_type'],'exact_scheme_status_matches':item.get('exact_mark_scheme_available') is None or answer['exact_mark_scheme_available']==item['exact_mark_scheme_available'],'key_point_coverage':_key_point_coverage(answer['answer'],item.get('required_key_points',[])) if not abstained else None,'faithfulness':faithfulness(answer['answer'],answer['retrieved_chunks']) if not abstained else None,'abstention_correct':abstained if not item['answerable'] else not abstained,'technical_failure':answer.get('technical_failure',False),'latency_ms':answer['latency_ms'],'input_tokens':answer.get('input_tokens',0),'output_tokens':answer.get('output_tokens',0),'estimated_cost':answer.get('estimated_cost',0.0),'generation_provider':answer['generation_provider'],'answer_cache_hit':cache_hit,**citations})
    def mean(field):
        values=[row[field] for row in rows if row[field] is not None]
        return sum(values)/len(values) if values else None
    return {'benchmark':benchmark_status(dataset),'scored_count':len(rows),'answer_cache_hits':sum(row['answer_cache_hit'] for row in rows),'official_status_accuracy':mean('status_matches'),'exact_scheme_status_accuracy':mean('exact_scheme_status_matches'),'citation_identity_accuracy':mean('identity_valid'),'citation_precision':mean('citation_precision'),'citation_coverage':mean('citation_coverage'),'gold_source_coverage':mean('gold_source_coverage'),'claim_citation_coverage':mean('claim_citation_coverage'),'key_point_coverage':mean('key_point_coverage'),'faithfulness':mean('faithfulness'),'abstention_accuracy':mean('abstention_correct'),'technical_failure_rate':mean('technical_failure'),'total_input_tokens':sum(row['input_tokens'] for row in rows),'total_output_tokens':sum(row['output_tokens'] for row in rows),'estimated_cost':sum(row['estimated_cost'] for row in rows),'evaluation_run_api_cost':sum(row['estimated_cost'] for row in rows if not row['answer_cache_hit']),'rows':rows}
