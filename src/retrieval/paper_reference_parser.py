from __future__ import annotations
import re
def parse_reference(query: str) -> dict:
    result={}
    code=re.search(r"\b(2210|9618)\s*/?\s*(\d{2})?\s*/?\s*(M/J|MJ|F/M|FM|O/N|ON)?\s*/?\s*(20\d{2}|\d{2})?",query,re.I)
    if code:
        result['subject_code']=code.group(1); result['component']=code.group(2); result['session']=(code.group(3) or '').replace('/','').upper() or None
        if code.group(4): result['year']=int(('20' if len(code.group(4))==2 else '')+code.group(4))
    for pattern,key in [(r"(?:question|q\.)\s*(\d+(?:\s*\([a-z0-9]+\))*)",'question_number'),(r"\b(20\d{2})\b",'year'),(r"\bpaper\s*(\d)",'component'),(r"\b(\d+)\s*marks?\b",'marks')]:
        m=re.search(pattern,query,re.I)
        if m: result[key]=int(m.group(1)) if key in {'year','marks'} else re.sub(r'\s+','',m.group(1))
    return {k:v for k,v in result.items() if v is not None}
