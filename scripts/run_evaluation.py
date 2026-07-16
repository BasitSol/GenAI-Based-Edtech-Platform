from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from evaluation.ingestion_eval import evaluate as ingestion
from evaluation.retrieval_eval import evaluate as retrieval
from evaluation.answer_eval import evaluate as answers
from evaluation.performance_eval import evaluate as performance
from evaluation.benchmark import approved_records, benchmark_status
from evaluation.quality_gates import assess
from src.core import ROOT
import json
import argparse
from datetime import datetime, timezone
from src.indexing.embedding_service import available_memory_mb, release_local_embedding_model
if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Run Phase 1 RAG evaluation on an approved benchmark split.')
    parser.add_argument('--dataset',default='evaluation/datasets/development.jsonl',help='Dataset path relative to project root or absolute path.')
    parser.add_argument('--retrieval-only',action='store_true',help='Skip answer generation and performance measurement.')
    parser.add_argument('--limit',type=int,help='Evaluate only the first N approved records (useful for low-cost pilot runs).')
    args=parser.parse_args(); dataset=Path(args.dataset); dataset=dataset if dataset.is_absolute() else ROOT/dataset
    records=approved_records(dataset); records=records[:args.limit] if args.limit else records
    available=available_memory_mb(); required=int(__import__('os').getenv('EMBEDDING_MIN_AVAILABLE_MB','2800'))
    if available is not None and available < required:
        raise SystemExit(f'Insufficient available RAM for full-precision BAAI/bge-m3: {available} MB available, {required} MB required. Close browsers, IDEs, Streamlit and other memory-heavy applications, then rerun. No model downgrade was applied.')
    retrieval_report=retrieval(dataset,args.limit)
    # Retrieval evaluation caches every benchmark/support query. Release the
    # full BGE-M3 weights before answer evaluation loads the full reranker.
    release_local_embedding_model()
    answer_report=None if args.retrieval_only else answers(dataset,args.limit)
    report={'generated_at':datetime.now(timezone.utc).isoformat(),'dataset':str(dataset),'evaluation_limit':args.limit,'benchmark':benchmark_status(dataset),'ingestion':ingestion(),'retrieval':retrieval_report,'answers':answer_report,'performance':None if args.retrieval_only else performance(answer_report['rows'])}
    report['quality_gates']=assess(report)
    output=ROOT/'evaluation/results'/f"{dataset.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"; output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({**report,'result_path':str(output)},indent=2))
