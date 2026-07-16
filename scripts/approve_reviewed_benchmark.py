"""Apply a human review decision and fill deterministic benchmark metadata.

Use only after a reviewer has checked the records.  Exact-scheme availability
is derived from the indexed document IDs, never from an LLM inference.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.core import ROOT, read_jsonl, write_jsonl


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--finalize-drafts', action='store_true')
    args=parser.parse_args()
    dataset=args.dataset if args.dataset.is_absolute() else ROOT/args.dataset
    rows=read_jsonl(dataset)
    chunks=read_jsonl(ROOT/'data_processed/chunks/all_chunks.jsonl')
    mark_schemes={chunk['document_id'] for chunk in chunks if chunk['document_type']=='MARK_SCHEME'}
    by_document={}
    for chunk in chunks:
        by_document.setdefault(chunk['document_id'], []).append(chunk)
    official=0
    for row in rows:
        row['review_status']='APPROVED'
        if args.finalize_drafts:
            row['question']=str(row.get('question','')).replace('[REVIEW REQUIRED] ', '', 1)
        if row.get('intent') != 'EXAM_ANSWER':
            if not row.get('expected_answer_type'):
                row['expected_answer_type']='CURRICULUM_EXPLANATION'
            continue
        primary=(row.get('gold_curriculum_sources') or [{}])[0]
        question_document=primary.get('document_id','')
        scheme_document=question_document.replace('_qp', '_ms')
        if scheme_document in mark_schemes:
            row['exact_mark_scheme_available']=True
            row['expected_answer_type']='OFFICIAL_MARK_SCHEME_SUPPORTED_ANSWER'
            row['expected_source_types']=['QUESTION_PAPER','MARK_SCHEME']
            if not any(item.get('document_id') == scheme_document for item in row['gold_curriculum_sources']):
                question_number=re.search(r'Question\s+(\d+)', str(row.get('question','')), re.I)
                number=question_number.group(1) if question_number else None
                scheme_chunks=[chunk for chunk in by_document[scheme_document] if str(chunk.get('question_number','')).startswith(f'{number}(')] if number else []
                pages=sorted({chunk['page_start'] for chunk in scheme_chunks})
                row['gold_curriculum_sources'].append({
                    'document_id':scheme_document,
                    'page_start':pages[0] if pages else 1,
                    'page_end':pages[-1] if pages else 1,
                })
            official+=1
        else:
            row['exact_mark_scheme_available']=False
            row['expected_answer_type']='AI_GENERATED_MODEL_ANSWER'
            row['expected_source_types']=['QUESTION_PAPER']
    write_jsonl(dataset, rows)
    print({'updated':len(rows), 'approved':len(rows), 'official_scheme_records':official, 'path':str(dataset)})


if __name__=='__main__':
    main()
