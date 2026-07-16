"""Deterministic grounding checks used alongside human/LLM rubric review."""
from __future__ import annotations
import re
from src.core import tokens

STOPWORDS={'the','and','for','that','this','with','from','into','are','was','were','has','have','had','will','would','could','should','can','may','its','their','then','than','each','one','two','three','also','because','therefore','answer','question','example','page'}
CITATION_RE=re.compile(r"\[([^\]]+)\s+p\.(\d+)\]",re.I)

def claim_units(answer:str)->list[str]:
    """Split prose into factual units while excluding labels and bare headings."""
    units=[]
    for line in answer.splitlines():
        line=line.strip()
        if not line or line.startswith('```') or line.endswith(':'): continue
        line=re.sub(r"^[-*•\d.)\s]+","",line).strip()
        for part in re.split(r"(?<=[.!?])\s+",line):
            plain=CITATION_RE.sub('',part).strip()
            if len(_content_terms(plain))>=2: units.append(part.strip())
    return units

def _content_terms(text:str)->set[str]:
    return {term for term in tokens(text) if len(term)>2 and term not in STOPWORDS}

def cited_chunks(claim:str,chunks:list[dict])->list[dict]:
    labels={(document.lower(),int(page)) for document,page in CITATION_RE.findall(claim)}
    return [item for item in chunks if (str(item.get('document_id','')).lower(),int(item.get('page_start',-1))) in labels]

def claim_supported(claim:str,chunks:list[dict],threshold:float=.30)->bool:
    terms=_content_terms(CITATION_RE.sub('',claim))
    if not terms or not chunks: return False
    evidence=_content_terms(' '.join(item.get('text','') for item in chunks))
    return len(terms & evidence)/len(terms)>=threshold

def enforce_grounding(answer:str,chunks:list[dict],threshold:float=.35,maximum_claims:int=6)->str:
    """Keep only evidence-supported generated claims and assign one best citation.

    This is a safety guard, not a metric shortcut: unsupported model output is
    removed rather than being decorated with an arbitrary citation.
    """
    grounded=[]
    for line in answer.splitlines():
        if len(grounded)>=maximum_claims: break
        stripped=line.strip()
        if not stripped: continue
        if stripped.startswith('```'): continue
        if stripped.startswith('#'):
            grounded.append(stripped); continue
        units=claim_units(stripped)
        if not units:
            plain=re.sub(r"^[-*•\d.)\s]+","",CITATION_RE.sub('',stripped)).strip(' :')
            if _content_terms(plain): units=[plain]
            else: continue
        for claim in units:
            if len(grounded)>=maximum_claims: break
            terms=_content_terms(CITATION_RE.sub('',claim))
            candidates=[]
            for chunk in chunks:
                if chunk.get('document_type')=='MARKING_PATTERN': continue
                evidence=_content_terms(chunk.get('text',''))
                score=len(terms & evidence)/len(terms) if terms else 0.0
                candidates.append((score,chunk))
            score,best=max(candidates,key=lambda item:item[0],default=(0.0,None))
            if best is None or score<threshold: continue
            plain=CITATION_RE.sub('',claim).strip()
            punctuation=plain[-1] if plain and plain[-1] in '.!?' else ''
            if punctuation: plain=plain[:-1].rstrip()
            grounded.append(f"{plain} [{best['document_id']} p.{best['page_start']}]{punctuation}")
    return '\n'.join(grounded).strip()

def _claims(answer:str)->list[str]:
    return [CITATION_RE.sub('',part).strip() for part in claim_units(answer)]

def faithfulness(answer:str,chunks:list[dict],threshold:float=.35)->float|None:
    claims=claim_units(answer)
    if not claims: return None
    supported=sum(claim_supported(claim,cited_chunks(claim,chunks) or chunks,threshold) for claim in claims)
    return supported/len(claims)

def claim_citation_coverage(answer:str)->float|None:
    claims=claim_units(answer)
    if not claims: return None
    return sum(bool(CITATION_RE.search(claim)) for claim in claims)/len(claims)
