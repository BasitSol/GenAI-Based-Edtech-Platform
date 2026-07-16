"""Create deterministic review packets for the three human-scored Phase 1 gates."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.core import ROOT

OUT=ROOT/'evaluation/manual_review'

def read_csv(path:Path)->list[dict]:
    with path.open(encoding='utf-8-sig',newline='') as handle:
        return list(csv.DictReader(handle))

def write_csv(name:str,rows:list[dict],fields:list[str])->None:
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)

def main()->None:
    manifests=read_csv(ROOT/'data_processed/manifests/documents_manifest.csv')
    by_document={row['document_id']:row for row in manifests}
    metadata=[]
    for row in manifests:
        metadata.append({**row,'review_correct':'','review_notes':'','reviewer':''})
    metadata_fields=list(manifests[0])+['review_correct','review_notes','reviewer']
    write_csv('metadata_review.csv',metadata,metadata_fields)

    boundaries=[]
    with (ROOT/'data_processed/chunks/all_chunks.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            item=json.loads(line)
            if item.get('document_type')!='QUESTION_PAPER': continue
            boundaries.append({
                'chunk_id':item['chunk_id'],'document_id':item['document_id'],
                'source_path':by_document.get(item['document_id'],{}).get('source_path',''),
                'page_start':item.get('page_start'),'page_end':item.get('page_end'),
                'question_number':item.get('question_number'),'marks':item.get('marks'),
                'text_preview':' '.join(item.get('text','').split())[:500],
                'boundary_correct':'','review_notes':'','reviewer':''})
    write_csv('question_boundary_review.csv',boundaries,list(boundaries[0]))

    ocr=[]
    for row in read_csv(ROOT/'data_processed/manifests/ocr_report.csv'):
        document=by_document.get(row['document_id'],{})
        page=int(row['page_number'])
        ocr.append({**row,'source_path':document.get('source_path',''),
                    'rendered_page':f"data_processed/figures/{row['document_id']}/page_{page:04d}.png",
                    'correct_words':'','reviewed_words':'','text_accuracy':'',
                    'review_notes':'','reviewer':''})
    ocr_fields=list(ocr[0])+['source_path','rendered_page','correct_words','reviewed_words','text_accuracy','review_notes','reviewer']
    # list(dict) already contains the appended keys, so de-duplicate while preserving order
    ocr_fields=list(dict.fromkeys(ocr_fields))
    write_csv('ocr_text_review.csv',ocr,ocr_fields)
    print({'metadata_records':len(metadata),'question_boundaries':len(boundaries),'ocr_pages':len(ocr),'path':str(OUT)})

if __name__=='__main__': main()
