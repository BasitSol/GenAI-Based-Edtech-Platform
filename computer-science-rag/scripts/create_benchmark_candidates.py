"""Create review-gated benchmark candidates from extracted question chunks."""
from pathlib import Path
import json, sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.core import ROOT, read_jsonl

if __name__=='__main__':
    chunks=read_jsonl(ROOT/'data_processed/chunks/all_chunks.jsonl')
    questions=[chunk for chunk in chunks if chunk['document_type']=='QUESTION_PAPER' and chunk.get('question_number')]
    records=[]
    for index,chunk in enumerate(questions,1):
        records.append({'id':f'CAND_{index:03d}','question':f"Answer {chunk['subject_code']}/{chunk['component']}/{chunk['session'][0]}/{chunk['session'][1]}/{str(chunk['year'])[-2:]} Question {chunk['question_number']}.",'level':chunk['level'],'intent':'EXAM_ANSWER','answerable':True,'exam_year':int(chunk['year']),'exact_mark_scheme_available':None,'expected_answer_type':None,'expected_source_types':['QUESTION_PAPER'],'gold_curriculum_sources':[{'document_id':chunk['document_id'],'page_start':chunk['page_start'],'page_end':chunk['page_end']}],'required_key_points':[],'review_status':'REQUIRES_REVIEW','review_notes':'Verify question text, source page, exact-scheme availability, expected status, and required key points before scoring.'})
    destination=ROOT/'evaluation/datasets/candidates_requires_review.jsonl'; destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in records),encoding='utf-8')
    print({'candidates':len(records),'path':str(destination)})
