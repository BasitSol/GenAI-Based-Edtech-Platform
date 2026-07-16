"""Phase 1 mandatory gate assessment from a completed evaluation report."""
from __future__ import annotations
import json
from src.core import ROOT

def _manual_gates()->dict:
    path=ROOT/'evaluation/manual_quality_gates.json'
    if not path.exists(): return {}
    try: return json.loads(path.read_text(encoding='utf-8')).get('gates',{})
    except (OSError,json.JSONDecodeError): return {}

def assess(report:dict)->dict:
    ingestion=report.get('ingestion') or {}; retrieval=report.get('retrieval') or {}; answers=report.get('answers') or {}; performance=report.get('performance') or {}
    specs={
        'page_extraction_coverage':(ingestion.get('coverage'),'>=',.99),
        'recall_at_5':(retrieval.get('recall_at_5'),'>=',.90),
        'recall_at_10':(retrieval.get('recall_at_10'),'>=',.95),
        'mrr':(retrieval.get('mrr'),'>=',.75),
        'source_routing_accuracy':(retrieval.get('source_routing_accuracy'),'>=',.95),
        'exact_scheme_retrieval_accuracy':(retrieval.get('exact_scheme_retrieval_accuracy'),'>=',.98),
        'faithfulness':(answers.get('faithfulness'),'>=',.92),
        'citation_precision':(answers.get('citation_precision'),'>=',.95),
        'citation_coverage':(answers.get('citation_coverage'),'>=',.90),
        'official_status_accuracy':(answers.get('official_status_accuracy'),'>=',1.0),
        'technical_failure_rate':(answers.get('technical_failure_rate'),'<=',.01),
        'median_latency_ms':(performance.get('median_latency_ms'),'<=',5000),
        'p95_latency_ms':(performance.get('p95_latency_ms'),'<=',10000),
    }
    gates={}
    for name,(value,operator,target) in specs.items():
        passed=None if value is None else (value>=target if operator=='>=' else value<=target)
        gates[name]={'value':value,'operator':operator,'target':target,'passed':passed}
    manual=_manual_gates()
    for name,target in (('metadata_accuracy',.98),('question_boundary_accuracy',.97),('ocr_text_accuracy',.95)):
        value=manual.get(name,{}).get('value')
        verified=manual.get(name,{}).get('verified_by') and manual.get(name,{}).get('reviewed_at')
        gates[name]={'value':value,'operator':'>=','target':target,'passed':(value>=target) if value is not None and verified else None,'requires_manual_review':True,'verified_by':manual.get(name,{}).get('verified_by'),'reviewed_at':manual.get(name,{}).get('reviewed_at')}
    measured=[gate['passed'] for gate in gates.values() if gate['passed'] is not None]
    return {'all_measured_gates_passed':bool(measured) and all(measured),'all_phase1_gates_verified':all(gate['passed'] is True for gate in gates.values()),'gates':gates}
