from __future__ import annotations
import re
def citations_for(chunks,answer_text:str|None=None):
    seen=set(); result=[]
    selected=[]
    if answer_text:
        for c in chunks:
            pattern=rf"\[{re.escape(c['document_id'])}\s+p\.{c['page_start']}\]"
            if re.search(pattern,answer_text,re.I): selected.append(c)
    # A missing inline citation should not fabricate support from every chunk.
    if not selected and chunks: selected=[chunks[0]]
    for c in selected:
        key=(c['document_id'],c['page_start'],c['chunk_id'])
        if key not in seen:
            seen.add(key); result.append({'document_id':c['document_id'],'page':c['page_start'],'chunk_id':c['chunk_id'],'relationship':'EXACT_EVIDENCE' if c['document_type']=='MARK_SCHEME' else 'CURRICULUM_EVIDENCE'})
    return result
def validate(citations,chunks):
    """Citations must identify a retrieved chunk, its document, and a valid page."""
    return all(any(x['chunk_id']==c['chunk_id'] and x['document_id']==c['document_id'] and x['page_start']==c['page'] for x in chunks) for c in citations)
