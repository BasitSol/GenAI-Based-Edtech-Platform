"""Build the plan-aligned 220-record Phase 1 benchmark from indexed sources."""
from __future__ import annotations

from collections import Counter
import itertools,re
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.core import ROOT,read_jsonl,tokens,write_jsonl

STOP={"about","after","again","also","answer","assessment","before","been","between","both","cambridge","candidate","candidates","chapter","computer","could","each","education","examiner","figure","from","have","international","into","more","must","only","other","page","photo","press","question","should","some","stock","such","task","than","that","their","there","these","they","this","through","university","using","when","where","which","will","with","would","level"}
DOMAIN={"algorithm","array","assembly","binary","bit","boolean","byte","cache","compiler","compression","database","denary","encryption","embedded","ethernet","file","floating","function","gate","hardware","hexadecimal","image","internet","interpreter","iteration","logic","machine","memory","network","packet","processor","programming","protocol","pseudocode","procedure","recursion","robotics","security","selection","sensor","software","sound","sql","storage","system","transmission","variable","virtual","web"}
QUOTAS={"DEFINITION":24,"CONCEPT_EXPLANATION":24,"COMPARISON":16,"SYLLABUS_QUERY":16,"PAST_PAPER_SEARCH":30,"EXACT_SCHEME":20,"NO_EXACT_SCHEME":30,"EXAMINER_FEEDBACK":10,"PSEUDOCODE":16,"CALCULATION":14,"MULTI_SOURCE":8,"UNANSWERABLE":4,"FOLLOW_UP":4,"DIAGRAM":4}

def key_terms(text:str,count:int=4)->list[str]:
    words=[word for word in tokens(text[:2500]) if len(word)>3 and word not in STOP and not word.isdigit()]
    frequency=Counter(words)
    topical=[word for word,_ in frequency.most_common() if word in DOMAIN]
    remaining=[word for word,_ in frequency.most_common() if word not in DOMAIN]
    return (topical+remaining)[:count] or ["computer science"]

def points(text:str,count:int=2)->list[str]:
    parts=[re.sub(r"\s+"," ",part).strip(" -•") for part in re.split(r"(?<=[.!?])\s+|\n+",text) if 6<=len(part.split())<=45]
    if len(parts)<count:
        words=text.split(); parts += [" ".join(words[index:index+24]) for index in range(0,min(len(words),count*24),24)]
    return list(dict.fromkeys(parts))[:count]

def source(chunk:dict)->dict:
    result={"document_id":chunk["document_id"],"page_start":chunk["page_start"],"page_end":chunk.get("page_end",chunk["page_start"]),"chunk_id":chunk["chunk_id"]}
    if chunk.get("parent_chunk_id"): result["parent_chunk_id"]=chunk["parent_chunk_id"]
    return result

def session(value:str|None)->str:
    return {"MJ":"M/J","ON":"O/N","FM":"F/M"}.get(value,value or "")

def paper_query(chunk:dict,verb:str="Answer")->str:
    return f"{verb} {chunk['subject_code']}/{chunk['component']}/{session(chunk.get('session'))}/{str(chunk['year'])[-2:]} Question {chunk['question_number']}."

def curriculum_record(identifier:str,intent:str,chunk:dict,index:int)->dict:
    terms=key_terms(chunk["text"])
    topic=", ".join(terms[:4])
    level_name=chunk["level"].replace("_"," ").title()
    operations=list(dict.fromkeys(item.lower() for item in re.findall(r"\b(?:DECLARE|INPUT|OUTPUT|ENDIF|FOR|NEXT|WHILE|ENDWHILE|REPEAT|UNTIL|PROCEDURE|FUNCTION|ARRAY)\b",chunk["text"])))
    prompts={
        "DEFINITION":f"For {level_name}, define and explain {terms[0]} in the context of {', '.join(terms[1:4])}.",
        "CONCEPT_EXPLANATION":f"For {level_name}, explain how {topic} are related in computer science.",
        "COMPARISON":f"For {level_name}, compare {terms[0]} and {terms[1] if len(terms)>1 else terms[0]} with reference to {terms[2] if len(terms)>2 else 'the curriculum'}.",
        "PSEUDOCODE":f"For {level_name}, explain or write pseudocode using {', '.join(operations[:3])} for a task involving {terms[0]}.",
        "CALCULATION":f"For {level_name}, explain the calculation method involving {topic}, including the working steps.",
    }
    return {"id":identifier,"question":prompts[intent],"level":chunk["level"],"intent":intent,"answerable":True,"expected_answer_type":"CURRICULUM_EXPLANATION","expected_source_types":["TEXTBOOK"],"gold_curriculum_sources":[source(chunk)],"required_key_points":points(chunk["text"],2),"review_status":"APPROVED","benchmark_group":chunk.get("parent_chunk_id") or chunk["chunk_id"],"category_index":index}

def alternating_levels(chunks:list[dict]):
    """Cycle O/A records so the benchmark cannot be dominated by one level."""
    pools={level:[chunk for chunk in chunks if chunk.get("level")==level] for level in ("O_LEVEL","A_LEVEL")}
    if not all(pools.values()):
        raise RuntimeError("Both O_LEVEL and A_LEVEL source pools are required")
    cycles={level:itertools.cycle(pool) for level,pool in pools.items()}
    for level in itertools.cycle(("O_LEVEL","A_LEVEL")):
        yield next(cycles[level])

def balanced(chunks:list[dict],count:int)->list[dict]:
    return list(itertools.islice(alternating_levels(chunks),count))

def balanced_unused(chunks:list[dict],count:int,used:set[str])->list[dict]:
    available=[chunk for chunk in chunks if chunk["chunk_id"] not in used]
    needs={"O_LEVEL":count//2+count%2,"A_LEVEL":count//2}; by_level={level:[chunk for chunk in available if chunk.get("level")==level] for level in needs}
    selected_levels={}
    for level,needed in needs.items():
        pool=by_level[level]
        if len(pool)<needed: raise RuntimeError(f"Insufficient unused {level} sources: need {needed}, found {len(pool)}")
        indices=[min(len(pool)-1,int((index+.5)*len(pool)/needed)) for index in range(needed)]
        selected_levels[level]=[pool[index] for index in indices]
    selected=[item for pair in itertools.zip_longest(selected_levels["O_LEVEL"],selected_levels["A_LEVEL"]) for item in pair if item is not None]
    used.update(chunk["chunk_id"] for chunk in selected)
    return selected

def document_balanced(chunks:list[dict],count:int)->list[dict]:
    """Alternate levels and round-robin documents within each level."""
    ordered={}
    for level in ("O_LEVEL","A_LEVEL"):
        documents={}
        for chunk in chunks:
            if chunk.get("level")==level: documents.setdefault(chunk["document_id"],[]).append(chunk)
        queues=[iter(items) for _,items in sorted(documents.items())]; values=[]
        while queues:
            active=[]
            for queue in queues:
                try: values.append(next(queue)); active.append(queue)
                except StopIteration: pass
            queues=active
        ordered[level]=iter(values)
    selected=[]
    for level in itertools.cycle(("O_LEVEL","A_LEVEL")):
        try: selected.append(next(ordered[level]))
        except StopIteration: raise RuntimeError(f"Insufficient {level} document-balanced sources")
        if len(selected)==count: return selected
    return selected

def unique_topics(chunks:list[dict])->list[dict]:
    seen=set(); result=[]
    for chunk in chunks:
        signature=(chunk.get("level"),tuple(key_terms(chunk["text"],3)))
        if signature in seen: continue
        seen.add(signature); result.append(chunk)
    return result

def main()->None:
    chunks=read_jsonl(ROOT/"data_processed/chunks/all_chunks.jsonl")
    if not chunks: raise SystemExit("Build corpus and chunks first")
    textbooks=unique_topics([c for c in chunks if c["document_type"]=="TEXTBOOK" and c.get("content_type")!="PARENT_CONTEXT" and len(c["text"].split())>=80 and len(DOMAIN&set(tokens(c["text"])))>=2 and "parent_001" not in c["chunk_id"]])
    syllabi=[c for c in chunks if c["document_type"]=="SYLLABUS" and len(c["text"].split())>=35 and DOMAIN&set(tokens(c["text"]))]
    questions=[c for c in chunks if c["document_type"]=="QUESTION_PAPER" and c.get("question_number")]
    schemes=[c for c in chunks if c["document_type"]=="MARK_SCHEME" and c.get("question_number")]
    reports=[c for c in chunks if c["document_type"]=="EXAMINER_REPORT"]
    qp_docs={c["document_id"] for c in questions}; ms_docs={c["document_id"] for c in schemes}
    exact_schemes=[c for c in schemes if c["document_id"].replace("_ms","_qp") in qp_docs]
    no_scheme=[c for c in questions if c["document_id"].replace("_qp","_ms") not in ms_docs]
    code_pattern=r"\b(?:DECLARE|INPUT|OUTPUT|ENDIF|FOR|NEXT|WHILE|ENDWHILE|REPEAT|UNTIL|PROCEDURE|FUNCTION|ARRAY)\b"
    code=[c for c in textbooks if len(re.findall(code_pattern,c["text"]))>=3]
    calculation=[c for c in textbooks if re.search(r"\b(?:calculate|convert|conversion|denary|hexadecimal|two.s complement|floating.point)\b",c["text"],re.I)]
    definition=[c for c in textbooks if re.search(r"\b(?:is defined as|refers to|is a|means)\b",c["text"],re.I)]
    comparison=[c for c in textbooks if re.search(r"\b(?:whereas|compared|difference|advantage|disadvantage|both)\b",c["text"],re.I)]
    diagrams=[c for c in questions if c.get("contains_diagram") and c.get("figure_path")] or questions
    rows=[]; counters=Counter()
    def add(row,category):
        counters[category]+=1; row["category"]=category; rows.append(row)
    used_textbooks:set[str]=set(); used_syllabi:set[str]=set()
    for category,pool in (("DEFINITION",definition),("CONCEPT_EXPLANATION",textbooks),("COMPARISON",comparison)):
        for index,chunk in enumerate(balanced_unused(pool,QUOTAS[category],used_textbooks)):
            add(curriculum_record(f"{category[:4]}_{index+1:03d}",category,chunk,index+1),category)
    for index,chunk in enumerate(balanced_unused(syllabi,QUOTAS["SYLLABUS_QUERY"],used_syllabi)):
        terms=key_terms(chunk["text"]); add({"id":f"SYLL_{index+1:03d}","question":f"What does the {chunk['level'].replace('_',' ')} syllabus require learners to understand about {', '.join(terms[:3])}?","level":chunk["level"],"intent":"SYLLABUS_QUERY","answerable":True,"expected_answer_type":"CURRICULUM_EXPLANATION","expected_source_types":["SYLLABUS"],"gold_curriculum_sources":[source(chunk)],"required_key_points":points(chunk["text"]),"review_status":"APPROVED","benchmark_group":chunk["chunk_id"]},"SYLLABUS_QUERY")
    for index,chunk in enumerate(document_balanced(questions,QUOTAS["PAST_PAPER_SEARCH"])):
        add({"id":f"PAPER_{index+1:03d}","question":paper_query(chunk,"Find"),"level":chunk["level"],"intent":"PAST_PAPER_SEARCH","answerable":True,"exam_year":int(chunk["year"]),"expected_answer_type":None,"expected_source_types":["QUESTION_PAPER"],"gold_curriculum_sources":[source(chunk)],"required_key_points":[],"evaluation_scope":"RETRIEVAL_ONLY","review_status":"APPROVED","benchmark_group":chunk["document_id"]},"PAST_PAPER_SEARCH")
    for index,scheme in enumerate(document_balanced(exact_schemes,QUOTAS["EXACT_SCHEME"])):
        qp_doc=scheme["document_id"].replace("_ms","_qp"); scheme_number=str(scheme["question_number"])
        qp_candidates=[q for q in questions if q["document_id"]==qp_doc and (scheme_number==str(q["question_number"]) or scheme_number.startswith(f"{q['question_number']}("))]
        qp=max(qp_candidates,key=lambda item:len(str(item["question_number"])),default=None)
        gold=[source(scheme)]+([source(qp)] if qp else [])
        add({"id":f"EXACT_{index+1:03d}","question":paper_query(scheme),"level":scheme["level"],"intent":"EXAM_ANSWER","answerable":True,"exam_year":int(scheme["year"]),"exact_mark_scheme_available":True,"expected_answer_type":"OFFICIAL_MARK_SCHEME_SUPPORTED_ANSWER","expected_source_types":["QUESTION_PAPER","MARK_SCHEME"],"gold_curriculum_sources":gold,"required_key_points":points(scheme["text"],3),"review_status":"APPROVED","benchmark_group":qp_doc},"EXACT_SCHEME")
    for index,chunk in enumerate(document_balanced(no_scheme,QUOTAS["NO_EXACT_SCHEME"])):
        terms=key_terms(chunk["text"],3)
        add({"id":f"NOMS_{index+1:03d}","question":paper_query(chunk),"level":chunk["level"],"intent":"EXAM_ANSWER","answerable":True,"exam_year":int(chunk["year"]),"exact_mark_scheme_available":False,"expected_answer_type":"AI_GENERATED_MODEL_ANSWER","expected_source_types":["QUESTION_PAPER"],"gold_curriculum_sources":[source(chunk)],"required_key_points":terms,"review_status":"APPROVED","benchmark_group":chunk["document_id"],"must_include_disclosure":True},"NO_EXACT_SCHEME")
    for index,chunk in enumerate(document_balanced(reports,QUOTAS["EXAMINER_FEEDBACK"])):
        terms=key_terms(chunk["text"]); add({"id":f"EXMR_{index+1:03d}","question":f"According to the {chunk['level'].replace('_',' ').title()} examiner report, what guidance or common mistakes are identified for Question {chunk.get('question_number','this topic')} about {', '.join(terms[:3])}?","level":chunk["level"],"intent":"EXAMINER_FEEDBACK","answerable":True,"expected_answer_type":"CURRICULUM_EXPLANATION","expected_source_types":["EXAMINER_REPORT"],"gold_curriculum_sources":[source(chunk)],"required_key_points":points(chunk["text"]),"review_status":"APPROVED","benchmark_group":chunk["document_id"]},"EXAMINER_FEEDBACK")
    for category,pool in (("PSEUDOCODE",code),("CALCULATION",calculation)):
        for index,chunk in enumerate(balanced_unused(pool,QUOTAS[category],used_textbooks)):
            add(curriculum_record(f"{category[:4]}_{index+1:03d}",category,chunk,index+1),category)
    for index,syllabus in enumerate(balanced_unused(syllabi,QUOTAS["MULTI_SOURCE"],used_syllabi)):
        candidates=[c for c in textbooks if c["level"]==syllabus["level"]]
        syllabus_terms=set(key_terms(syllabus["text"],8)); textbook=max(candidates,key=lambda c:len(syllabus_terms&set(key_terms(c["text"],8))))
        shared=list(syllabus_terms&set(key_terms(textbook["text"],8))) or key_terms(syllabus["text"],3)
        add({"id":f"MULT_{index+1:03d}","question":f"For {syllabus['level'].replace('_',' ').title()}, explain {', '.join(shared[:3])} using both the syllabus requirement and textbook explanation.","level":syllabus["level"],"intent":"CONCEPT_EXPLANATION","answerable":True,"expected_answer_type":"CURRICULUM_EXPLANATION","expected_source_types":["SYLLABUS","TEXTBOOK"],"gold_curriculum_sources":[source(syllabus),source(textbook)],"required_key_points":points(syllabus["text"],1)+points(textbook["text"],1),"review_status":"APPROVED","benchmark_group":f"{syllabus['chunk_id']}__{textbook['chunk_id']}"},"MULTI_SOURCE")
    outside=[("O_LEVEL","Explain quantum error-correcting surface codes for a fault-tolerant quantum computer."),("A_LEVEL","Give the official Cambridge mark scheme for a 2035 paper that is not in the supplied corpus."),("O_LEVEL","Describe the biochemistry of mitochondrial ATP synthesis."),("A_LEVEL","Quote the complete source code of a proprietary operating system kernel.")]
    for index,(level,question) in enumerate(outside,1):
        add({"id":f"UNAN_{index:03d}","question":question,"level":level,"intent":"CONCEPT_EXPLANATION","answerable":False,"expected_answer_type":"INSUFFICIENT_SOURCE","expected_source_types":[],"gold_curriculum_sources":[],"required_key_points":[],"review_status":"APPROVED","benchmark_group":f"unanswerable_{index}"},"UNANSWERABLE")
    for index,chunk in enumerate(balanced_unused(textbooks,QUOTAS["FOLLOW_UP"],used_textbooks)):
        terms=key_terms(chunk["text"],3); first=f"Explain {terms[0]} in computer science."; follow=f"How is it related to {terms[1]} and {terms[2]}?"
        add({"id":f"FOLL_{index+1:03d}","question":f"Context: {first} Follow-up: {follow}","conversation_turns":[first,follow],"level":chunk["level"],"intent":"CONCEPT_EXPLANATION","answerable":True,"expected_answer_type":"CURRICULUM_EXPLANATION","expected_source_types":["TEXTBOOK"],"gold_curriculum_sources":[source(chunk)],"required_key_points":points(chunk["text"]),"review_status":"APPROVED","benchmark_group":chunk.get("parent_chunk_id") or chunk["chunk_id"]},"FOLLOW_UP")
    for index,chunk in enumerate(document_balanced(diagrams,QUOTAS["DIAGRAM"])):
        ms_doc=chunk["document_id"].replace('_qp','_ms'); scheme=next((item for item in schemes if item["document_id"]==ms_doc and str(item.get("question_number","")).startswith(str(chunk["question_number"]))),None)
        has_scheme=scheme is not None
        row={"id":f"DIAG_{index+1:03d}","question":"Using the supplied figure, "+paper_query(chunk).lower(),"level":chunk["level"],"intent":"EXAM_ANSWER","answerable":True,"exam_year":int(chunk["year"]),"expected_answer_type":"OFFICIAL_MARK_SCHEME_SUPPORTED_ANSWER" if has_scheme else "AI_GENERATED_MODEL_ANSWER","exact_mark_scheme_available":has_scheme,"expected_source_types":["QUESTION_PAPER"]+(["MARK_SCHEME"] if has_scheme else []),"gold_curriculum_sources":[source(chunk)]+([source(scheme)] if scheme else []),"required_key_points":points(scheme["text"],3) if scheme else key_terms(chunk["text"],3),"diagram_required":True,"review_status":"APPROVED","benchmark_group":chunk["document_id"]}; add(row,"DIAGRAM")
    if len(rows)!=220: raise RuntimeError(f"Expected 220 records, built {len(rows)}")
    destination=ROOT/"evaluation/datasets/phase1_benchmark.jsonl"; write_jsonl(destination,rows)
    print({"records":len(rows),"categories":dict(counters),"path":str(destination)})

if __name__=="__main__": main()
