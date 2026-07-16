"""Validated, review-gated benchmark records for Phase 1 evaluation."""
from __future__ import annotations
import json
from pathlib import Path

REQUIRED={'id','question','level','intent','answerable','review_status'}
VALID_STATUSES={'REQUIRES_REVIEW','APPROVED','REJECTED'}

REQUIRED_APPROVED_ANSWERABLE={
    'expected_answer_type', 'expected_source_types', 'gold_curriculum_sources'
}

def load_records(path: Path) -> list[dict]:
    rows=[]
    for number,line in enumerate(path.read_text(encoding='utf-8').splitlines(),1):
        if not line.strip(): continue
        row=json.loads(line); missing=REQUIRED-set(row)
        if missing: raise ValueError(f'{path}:{number} missing {sorted(missing)}')
        if row['review_status'] not in VALID_STATUSES: raise ValueError(f'{path}:{number} invalid review_status')
        if row['review_status']=='APPROVED' and row['answerable'] and not row.get('gold_curriculum_sources'):
            raise ValueError(f'{path}:{number} approved answerable record needs gold_curriculum_sources')
        rows.append(row)
    return rows

def approved_records(path: Path) -> list[dict]:
    return [row for row in load_records(path) if row['review_status']=='APPROVED']

def benchmark_status(path: Path) -> dict:
    records=load_records(path); counts={status:0 for status in VALID_STATUSES}
    for row in records: counts[row['review_status']]+=1
    return {'total':len(records),'by_review_status':counts,'measurable_count':counts['APPROVED']}


def audit_records(records: list[dict]) -> dict:
    """Report review completeness without changing any reviewer decision."""
    ids=set(); errors=[]; warnings=[]; categories={}; levels={}
    for index,row in enumerate(records, 1):
        identifier=row.get('id')
        if not identifier or identifier in ids:
            errors.append({'line':index, 'id':identifier, 'issue':'duplicate_or_missing_id'})
        ids.add(identifier)
        levels[row.get('level','UNKNOWN')]=levels.get(row.get('level','UNKNOWN'),0)+1
        categories[row.get('intent','UNKNOWN')]=categories.get(row.get('intent','UNKNOWN'),0)+1
        if row.get('review_status') != 'APPROVED':
            continue
        if row.get('answerable'):
            required=REQUIRED_APPROVED_ANSWERABLE-({'expected_answer_type'} if row.get('evaluation_scope')=='RETRIEVAL_ONLY' else set())
            for field in required:
                if not row.get(field):
                    warnings.append({'line':index, 'id':identifier, 'issue':f'missing_{field}'})
            if row.get('intent') == 'EXAM_ANSWER' and row.get('exact_mark_scheme_available') is None:
                warnings.append({'line':index, 'id':identifier, 'issue':'missing_exact_mark_scheme_available'})
            if row.get('evaluation_scope') != 'RETRIEVAL_ONLY' and not row.get('required_key_points'):
                warnings.append({'line':index, 'id':identifier, 'issue':'missing_required_key_points'})
        if not str(row.get('question','')).strip() or '[REVIEW REQUIRED]' in str(row.get('question','')):
            warnings.append({'line':index, 'id':identifier, 'issue':'question_needs_review'})
    return {
        'total':len(records), 'unique_ids':len(ids), 'levels':levels,
        'intents':categories, 'errors':errors, 'warnings':warnings,
        'freezable_count':sum(1 for row in records if row.get('review_status')=='APPROVED')-len({w['line'] for w in warnings})
    }
