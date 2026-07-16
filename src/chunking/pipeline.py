from __future__ import annotations
import re
from pathlib import Path
from src.core import ROOT, read_jsonl, write_jsonl, token_count, write_csv
from src.ingestion.pipeline import MANIFEST_FIELDS
from .base_chunker import windows
import csv

def _manifest(processed: Path):
    with (processed/"manifests"/"documents_manifest.csv").open(encoding="utf-8") as f: return list(csv.DictReader(f))
def _base(doc,page,text,n,page_record=None):
    return {"chunk_id":f"{doc['document_id']}_{n:04d}","document_id":doc["document_id"],"document_type":doc["document_type"],"level":doc["level"],"subject_code":doc["subject_code"],"year":doc["year"] or None,"session":doc["session"] or None,"component":doc["component"] or None,"page_start":page,"page_end":page,"text":text,"token_count":token_count(text),"authority_level":int(doc["authority_level"] or 0),"contains_diagram":bool((page_record or {}).get("contains_diagram")),"figure_path":(page_record or {}).get("figure_path")}
def _question_chunks(doc,pages):
    chunks=[]; combined="\n".join(f"[Page {p['page_number']}]\n{p['clean_text']}" for p in pages)
    raw_candidates=[]
    # Cambridge uses both `1` on its own line and `10 Question text`.
    # Sequential numbering plus header rejection distinguishes questions from
    # printed page numbers, table cells and pseudocode line numbers.
    for match in re.finditer(r"(?m)^\s*(\d{1,2})(?:[ \t]+(?=\S)|[ \t]*$)",combined):
        if len(match.group(1))>1 and match.group(1).startswith('0'):
            continue
        following=combined[match.end():].lstrip().splitlines()
        next_line=next((line.strip() for line in following if line.strip()),"")
        if re.match(r"(?:2210|9618)/\d{2}|\[Turn over|BLANK PAGE|\[Page \d+\]|\d{1,2}\s+\S",next_line,re.I):
            continue
        if re.match(r"(?:DECLARE|CONSTANT|INPUT|OUTPUT|IF|THEN|ELSE|ENDIF|FOR|NEXT|WHILE|ENDWHILE|REPEAT|UNTIL|CASE|PROCEDURE|FUNCTION|RETURN)\b",next_line):
            continue
        if len(re.findall(r"[A-Za-z]+",next_line))<3:
            continue
        raw_candidates.append(match)
    # Main Cambridge questions are sequential. This rejects page numbers,
    # marks and table values that happen to be followed by a subquestion.
    candidates=[]; expected=1
    for match in raw_candidates:
        value=int(match.group(1))
        if value==expected:
            candidates.append(match); expected+=1
    for i,match in enumerate(candidates):
        text=combined[match.start():candidates[i+1].start() if i+1<len(candidates) else len(combined)].strip()
        if len(text)<20: continue
        preceding_pages=re.findall(r"\[Page (\d+)\]", combined[:match.start()+1])
        page=int(preceding_pages[-1]) if preceding_pages else pages[0]["page_number"]
        # Top-level parts in these papers use (a)-(h); (i), (ii), ... are
        # Roman-numeral children and remain inside their parent part.
        subparts=list(re.finditer(r"(?m)^\s*\(([a-h])\)\s*",text,re.I))
        parts=[]
        if subparts:
            stem=text[:subparts[0].start()].strip()
            for index,subpart in enumerate(subparts):
                body=text[subpart.start():subparts[index+1].start() if index+1<len(subparts) else len(text)].strip()
                parts.append((f"{match.group(1)}({subpart.group(1).lower()})",f"{stem}\n{body}".strip(),match.start()+subpart.start()))
        else:
            parts=[(match.group(1),text,match.start())]
        for question_number,part,absolute_start in parts:
            part_pages=re.findall(r"\[Page (\d+)\]",combined[:absolute_start+1]); part_page=int(part_pages[-1]) if part_pages else page
            page_record=next((p for p in pages if p['page_number']==part_page),None)
            # A figure may belong to the shared stem on the main-question page.
            figure_record=next((p for p in pages if p['page_number'] in {page,part_page} and p.get('contains_diagram')),None)
            marks=re.findall(r"\[\s*(\d+)\s*\]",part)
            x=_base(doc,part_page,part,len(chunks)+1,page_record)
            if figure_record:
                x['contains_diagram']=True; x['figure_path']=figure_record.get('figure_path')
            x.update({"question_number":question_number,"marks":int(marks[-1]) if marks else None,"content_type":"QUESTION"}); chunks.append(x)
    return chunks
def _scheme_chunks(doc,pages):
    chunks=[]
    for p in pages:
        active_question=None; buffer=[]
        def flush():
            if not active_question or len("\n".join(buffer).strip())<15: return
            part="\n".join(buffer).strip(); marks=re.findall(r"\[\s*(\d+)\s*\]|\b(\d+)\s+marks?\b",part,re.I)
            mark_value=next((int(a or b) for a,b in marks),None)
            x=_base(doc,p["page_number"],part,len(chunks)+1,p); x.update({"question_number":active_question,"maximum_marks":mark_value,"content_type":"MARK_SCHEME_ENTRY","relationship":"EXACT_MATCH_CANDIDATE"}); chunks.append(x)
        for line in p["clean_text"].splitlines():
            # Cambridge mark schemes place the question/subquestion label on its
            # own line. "1 mark" is explicitly not a question label.
            label=re.fullmatch(r"\s*(\d{1,2}\([a-z]\)(?:\([ivx]+\))?)\s*",line,re.I)
            if label:
                flush(); active_question=label.group(1); buffer=[line]
            elif active_question:
                buffer.append(line)
        flush()
    return chunks
def _examiner_report_chunks(doc,pages):
    chunks=[]; combined="\n".join(f"[Page {p['page_number']}]\n{p['clean_text']}" for p in pages)
    matches=list(re.finditer(r"(?im)^\s*(?:question\s+)?(\d{1,2})(?:\s*\([a-z]\))?\s*$",combined))
    for index,match in enumerate(matches):
        text=combined[match.start():matches[index+1].start() if index+1<len(matches) else len(combined)].strip()
        if len(text.split())<25: continue
        preceding=re.findall(r"\[Page (\d+)\]",combined[:match.start()+1]); page=int(preceding[-1]) if preceding else 1
        page_record=next((p for p in pages if p['page_number']==page),None)
        x=_base(doc,page,text,len(chunks)+1,page_record)
        x.update({"question_number":match.group(1),"content_type":"EXAMINER_COMMENT","relationship":"EXAMINER_GUIDANCE"}); chunks.append(x)
    # Reports with narrative section headings still remain retrievable.
    if not chunks:
        for p in pages:
            for part in windows(p['clean_text'],500,60):
                if len(part.split())<25: continue
                x=_base(doc,p['page_number'],part,len(chunks)+1,p); x.update({"content_type":"EXAMINER_COMMENT","relationship":"EXAMINER_GUIDANCE"}); chunks.append(x)
    return chunks
def _syllabus_chunks(doc,pages):
    chunks=[]
    for p in pages:
        parts=re.split(r"(?m)(?=^\s*\d+(?:\.\d+)+\s)",p["clean_text"])
        for part in parts:
            if len(part.strip())>30:
                x=_base(doc,p["page_number"],part.strip(),len(chunks)+1,p); x.update({"content_type":"LEARNING_OBJECTIVE"}); chunks.append(x)
    return chunks
def _textbook_chunks(doc,pages):
    chunks=[]; words=[]; page_offsets=[]
    for page_record in pages:
        page_words=page_record['clean_text'].split()
        if page_words:
            page_offsets.append((len(words),page_record['page_number']))
            words.extend(page_words)
    if not words: return chunks
    def page_for(offset):
        return next((page for start,page in reversed(page_offsets) if start<=offset),pages[0]['page_number'])
    child_number=0
    for parent_index,start in enumerate(range(0,len(words),1500),1):
        parent_words=words[start:start+1500]
        if not parent_words: continue
        parent=' '.join(parent_words); pid=f"{doc['document_id']}_parent_{parent_index:03d}"; page=page_for(start)
        page_record=next((p for p in pages if p['page_number']==page),None)
        x=_base(doc,page,parent,-parent_index,page_record); x.update({'chunk_id':pid,'content_type':'PARENT_CONTEXT','parent_chunk_id':None}); chunks.append(x)
        for child_start in range(0,len(parent_words),390):
            child_words=parent_words[child_start:child_start+450]
            if not child_words: continue
            child_number+=1; child=' '.join(child_words); child_page=page_for(start+child_start)
            page_record=next((p for p in pages if p['page_number']==child_page),None)
            x=_base(doc,child_page,child,child_number,page_record); x.update({'chunk_id':f'{pid}_child_{child_number:03d}','content_type':'EXPLANATION','parent_chunk_id':pid}); chunks.append(x)
            if child_start+450>=len(parent_words): break
    return chunks
def build_chunks(processed: Path=ROOT/"data_processed") -> dict:
    all_chunks=[]
    for doc in _manifest(processed):
        pages=read_jsonl(processed/"pages"/doc["document_id"] / "pages.jsonl")
        typ=doc["document_type"]
        if typ=="TEXTBOOK": chunks=_textbook_chunks(doc,pages)
        elif typ=="SYLLABUS": chunks=_syllabus_chunks(doc,pages)
        elif typ=="QUESTION_PAPER": chunks=_question_chunks(doc,pages)
        elif typ=="MARK_SCHEME": chunks=_scheme_chunks(doc,pages)
        elif typ=="EXAMINER_REPORT": chunks=_examiner_report_chunks(doc,pages)
        else: chunks=[]
        write_jsonl(processed/"chunks"/f"{doc['document_id']}.jsonl",chunks); all_chunks.extend(chunks)
    write_jsonl(processed/"chunks"/"all_chunks.jsonl",all_chunks)
    patterns=[]
    for c in all_chunks:
        if c["document_type"]=="MARK_SCHEME": patterns.append({**c,"chunk_id":"pattern_"+c["chunk_id"],"document_type":"MARKING_PATTERN","relationship":"PATTERN_EXAMPLE","authority_level":2})
    write_jsonl(processed/"chunks"/"marking_patterns.jsonl",patterns)
    by_type={}
    seen={}; duplicates=[]
    for chunk in all_chunks:
        by_type[chunk["document_type"]]=by_type.get(chunk["document_type"],0)+1
        fingerprint=" ".join(chunk["text"].lower().split())
        if fingerprint in seen: duplicates.append({"chunk_id":chunk["chunk_id"],"duplicate_of":seen[fingerprint]})
        else: seen[fingerprint]=chunk["chunk_id"]
    write_csv(processed/"manifests"/"chunk_statistics.csv",[{"document_type":k,"chunk_count":v} for k,v in by_type.items()],["document_type","chunk_count"])
    write_csv(processed/"manifests"/"duplicate_chunks.csv",duplicates,["chunk_id","duplicate_of"])
    return {"chunks":len(all_chunks),"patterns":len(patterns),"duplicates":len(duplicates)}

if __name__ == "__main__": print(build_chunks())
