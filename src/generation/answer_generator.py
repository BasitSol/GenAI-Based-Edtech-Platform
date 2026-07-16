from __future__ import annotations
import base64,os,time
from pathlib import Path
from src.core import ROOT
from src.workflows.rag_graph import retrieve
from src.memory.conversation_store import ConversationStore
from .citation_validator import citations_for,validate
from .prompts import SYSTEM_PROMPT
from .grounding_checker import enforce_grounding

ABSTENTION_PHRASES=('insufficient evidence','not enough information','cannot answer from the supplied','not supported by the supplied','sources do not contain')

def _is_abstention(answer:str)->bool:
    lowered=answer.lower()
    return any(phrase in lowered for phrase in ABSTENTION_PHRASES)

def _official_scheme_answer(chunks:list[dict])->str:
    """Return the matching scheme entry directly; no generative paraphrase needed."""
    entries=[item for item in chunks if item.get('document_type')=='MARK_SCHEME']
    lines=[]
    for item in entries:
        text=' '.join(item.get('text','').replace('ï‚·','•').replace('âˆ’','−').split())
        if text:
            lines.append(f"{text} [{item['document_id']} p.{item['page_start']}]")
    return '\n\n'.join(lines)

def answer_question(query:str,level:str|None=None,exam_year:int|None=None,conversation_id:str|None=None)->dict:
    started=time.perf_counter(); memory=ConversationStore(); state=memory.get(conversation_id); retrieval=retrieve(query,level or state.get('selected_level'),exam_year or state.get('selected_exam_year'),state); chunks=retrieval['chunks']; route=retrieval['route']; exact=retrieval['exact_mark_scheme_available']
    marks=route.get('marks') or max((int(item.get('marks') or 0) for item in chunks if item.get('document_type')=='QUESTION_PAPER'),default=0) or None
    route['marks']=marks
    usage={'input_tokens':0,'output_tokens':0}; generation_error=None
    if not chunks:
        answer='I could not find enough information in the available curriculum material to answer this reliably.'; kind='INSUFFICIENT_SOURCE'; disclosure=None; confidence=.0
    else:
        kind='OFFICIAL_MARK_SCHEME_SUPPORTED_ANSWER' if exact else ('AI_GENERATED_MODEL_ANSWER' if route['intent']=='EXAM_ANSWER' else 'CURRICULUM_EXPLANATION')
        disclosure=None if exact or kind=='CURRICULUM_EXPLANATION' else 'The exact mark scheme for this question is not included in the available material. This model answer is generated from retrieved curriculum evidence and assessment-pattern examples.'
        # Safe local baseline: presents grounded evidence verbatim until an API key is deliberately configured.
        evidence='\n\n'.join(c['text'][:1200] for c in chunks[:3])
        generation_provider='retrieval_only'
        try:
            if exact:
                answer=_official_scheme_answer(chunks)
                generation_provider='deterministic_mark_scheme'
            elif os.getenv('OPENAI_API_KEY'):
                answer,usage=_generate_with_openai(query,chunks,kind,disclosure,route)
                answer=enforce_grounding(answer,chunks,maximum_claims=max(1,min(6,marks or 6)))
                generation_provider='openai'
                if not answer or _is_abstention(answer):
                    answer='I could not find enough information in the available curriculum material to answer this reliably.'
                    kind='INSUFFICIENT_SOURCE'; disclosure=None; confidence=0.0
            else: answer=evidence
        except Exception as exc:
            answer=evidence; generation_error=f"{type(exc).__name__}: {str(exc)[:300]}"
        confidence=min(.95,.45+.06*len(chunks));
    if not chunks: generation_provider='none'
    citations=citations_for(chunks,answer)
    estimated_cost=_estimated_cost(os.getenv('GENERATOR_MODEL','gpt-4.1-mini'),usage)
    conversation_id=memory.record(state['conversation_id'],query,answer,route)
    return {'answer':answer,'answer_type':kind,'exact_mark_scheme_available':exact,'disclosure':disclosure,'generation_provider':generation_provider,'generator_model':os.getenv('GENERATOR_MODEL') if generation_provider=='openai' else None,'estimated_marks_addressed':marks,'maximum_marks':marks,'confidence':round(confidence,2),'citations':citations,'retrieved_chunks':chunks,'conversation_id':conversation_id,'latency_ms':round((time.perf_counter()-started)*1000),'input_tokens':usage['input_tokens'],'output_tokens':usage['output_tokens'],'estimated_cost':estimated_cost,'technical_failure':bool(generation_error),'generation_error':generation_error,'citation_valid':validate(citations,chunks),'retrieval_debug':retrieval['retrieval_debug']}
def _generate_with_openai(query,chunks,kind,disclosure,route):
    from openai import OpenAI
    # Keep the exact question, then provide several factual sources and at most
    # one style-only pattern. A fixed first-three policy could crowd factual
    # evidence out with a question chunk and a marking-pattern chunk.
    question_sources=[item for item in chunks if item.get('document_type')=='QUESTION_PAPER'][:1]
    factual_sources=[item for item in chunks if item.get('document_type') not in {'QUESTION_PAPER','MARKING_PATTERN'}][:4]
    pattern_sources=[item for item in chunks if item.get('document_type')=='MARKING_PATTERN'][:1]
    selected=[]
    for item in question_sources+factual_sources+pattern_sources:
        if item not in selected: selected.append(item)
    context='\n\n'.join(f"[{c['document_id']} p.{c['page_start']}; type={c['document_type']}; relationship={c.get('relationship','EVIDENCE')}] {c['text'][:1800]}" for c in selected)
    prompt=(f'Answer type: {kind}\nDisclosure: {disclosure or "none"}\nUser reference: {query}\n'
            'For a referenced exam question, read the full question wording from the supplied QUESTION_PAPER source. '
            'Use TEXTBOOK or SYLLABUS sources for factual support when no exact mark scheme is supplied.\n'
            f'Sources:\n{context}')
    content=[{'type':'text','text':prompt}]
    visual_terms=('diagram','figure','logic circuit','flowchart','flow chart','table below','shown below','supplied figure')
    question_text=' '.join(item.get('text','').lower() for item in chunks if item.get('document_type')=='QUESTION_PAPER')
    figure_required=route.get('intent')=='EXAM_ANSWER' and any(term in (query.lower()+' '+question_text) for term in visual_terms) and any(item.get('contains_diagram') and item.get('document_type')=='QUESTION_PAPER' for item in chunks)
    for chunk in ([item for item in chunks if item.get('figure_path') and item.get('document_type')=='QUESTION_PAPER'][:1] if figure_required else []):
        path=ROOT/chunk['figure_path']
        if path.exists():
            encoded=base64.b64encode(path.read_bytes()).decode('ascii')
            content.append({'type':'image_url','image_url':{'url':f'data:image/png;base64,{encoded}','detail':'high'}})
    output_limit=350 if route.get('intent') in {'PSEUDOCODE','CALCULATION'} else 220
    response=OpenAI().chat.completions.create(model=os.getenv('GENERATOR_MODEL','gpt-4.1-mini'),temperature=0,max_tokens=output_limit,messages=[{'role':'system','content':SYSTEM_PROMPT},{'role':'user','content':content}])
    usage=response.usage
    return response.choices[0].message.content,{'input_tokens':getattr(usage,'prompt_tokens',0) or 0,'output_tokens':getattr(usage,'completion_tokens',0) or 0}

def _estimated_cost(model:str,usage:dict)->float:
    prices={'gpt-4.1-mini':(.40,1.60),'gpt-4.1':(2.00,8.00)}
    input_price,output_price=prices.get(model,(0.0,0.0))
    return round((usage['input_tokens']*input_price+usage['output_tokens']*output_price)/1_000_000,8)
