"""Bulk-update review status after a human reviewer has checked a JSONL set."""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from evaluation.benchmark import VALID_STATUSES, load_records
from src.core import ROOT

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--status',required=True,choices=sorted(VALID_STATUSES))
    args=parser.parse_args(); path=Path(args.dataset); path=path if path.is_absolute() else ROOT/path
    records=load_records(path)
    for record in records: record['review_status']=args.status
    path.write_text(''.join(json.dumps(record,ensure_ascii=False)+'\n' for record in records),encoding='utf-8')
    print({'updated':len(records),'status':args.status,'path':str(path)})
