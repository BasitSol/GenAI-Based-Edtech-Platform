from __future__ import annotations
from evaluation.retrieval_eval import _relevant
from src.generation.citation_validator import validate
from src.generation.grounding_checker import claim_citation_coverage, claim_units, cited_chunks, claim_supported
def evaluate_answer(answer:dict, record:dict|None=None)->dict:
    citations=answer.get('citations',[]); chunks=answer.get('retrieved_chunks',[])
    # The plan defines precision as citations supporting their associated claims,
    # not citations that happen to be the annotator's single gold retrieval page.
    inline_claims=[claim for claim in claim_units(answer.get('answer','')) if cited_chunks(claim,chunks)]
    supported_inline=[claim for claim in inline_claims if claim_supported(claim,cited_chunks(claim,chunks))]
    gold=record.get('gold_curriculum_sources',[]) if record else []
    covered=sum(any(c['document_id']==s.get('document_id') and (s.get('page_start',s.get('page')) is None or s.get('page_start',s.get('page'))<=c['page']<=s.get('page_end',s.get('page_start',s.get('page')))) for c in citations) for s in gold)
    claim_coverage=claim_citation_coverage(answer.get('answer',''))
    return {'citation_count':len(citations),'identity_valid':validate(citations,chunks),'citation_precision':len(supported_inline)/len(inline_claims) if inline_claims else None,'citation_coverage':claim_coverage,'gold_source_coverage':covered/len(gold) if gold else None,'claim_citation_coverage':claim_coverage}
