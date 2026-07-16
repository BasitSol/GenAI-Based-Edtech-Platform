"""Freeze 120/50/50 benchmark splits from fully reviewed records only."""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.benchmark import audit_records, load_records
from src.core import ROOT, write_jsonl


def main() -> None:
    parser=argparse.ArgumentParser(description='Create deterministic benchmark splits from approved, complete records.')
    parser.add_argument('--input', type=Path, action='append', required=True, help='Repeat for each reviewed JSONL source.')
    parser.add_argument('--seed', type=int, default=20260716)
    parser.add_argument('--force', action='store_true', help='Replace existing frozen split files.')
    args=parser.parse_args()
    all_rows=[row for path in args.input for row in load_records(path)]
    audit=audit_records(all_rows)
    warning_ids={warning['id'] for warning in audit['warnings']}
    eligible=[row for row in all_rows if row.get('review_status')=='APPROVED' and row.get('id') not in warning_ids]
    if len(eligible) < 220:
        raise SystemExit(f'Need 220 fully reviewed records; found {len(eligible)} eligible. Audit warnings: {len(audit["warnings"])}')
    rng=random.Random(args.seed)
    # Build connected components so records sharing either an explicit paper/
    # parent group or any gold chunk can never leak across splits.
    parent=list(range(len(eligible)))
    def find(index):
        while parent[index]!=index:
            parent[index]=parent[parent[index]]; index=parent[index]
        return index
    def union(left,right):
        left,right=find(left),find(right)
        if left!=right: parent[right]=left
    owners={}
    for index,row in enumerate(eligible):
        keys=[f"group:{row.get('benchmark_group') or row['id']}"]
        keys += [f"chunk:{source.get('chunk_id')}" for source in row.get('gold_curriculum_sources',[]) if source.get('chunk_id')]
        for key in keys:
            if key in owners: union(index,owners[key])
            else: owners[key]=index
    groups=defaultdict(list)
    for index,row in enumerate(eligible): groups[find(index)].append(row)
    grouped=list(groups.items()); rng.shuffle(grouped); grouped.sort(key=lambda item:len(item[1]),reverse=True)
    targets={'development':120,'validation':50,'hidden_test':50}; splits={name:[] for name in targets}
    assigned=set()
    # Seed both evaluation splits with high-risk capabilities. Without this,
    # intact paper/report groups can all land in development despite an exact
    # 120/50/50 size split.
    for name in ('validation','hidden_test'):
        anchors=('EXACT_SCHEME','EXAMINER_FEEDBACK','UNANSWERABLE','FOLLOW_UP','MULTI_SOURCE')
        if name=='validation': anchors += ('DEFINITION','CONCEPT_EXPLANATION','PSEUDOCODE','SYLLABUS_QUERY')
        else: anchors += ('DIAGRAM','DEFINITION','CONCEPT_EXPLANATION')
        for category in anchors:
            if any(row.get('category')==category for row in splits[name]): continue
            candidates=[item for item in grouped if item[0] not in assigned and any(row.get('category')==category for row in item[1]) and len(splits[name])+len(item[1])<=targets[name]]
            if not candidates: continue
            if category=='EXACT_SCHEME':
                # The available exact-scheme paper groups are level-specific;
                # place one level in each held-out split before balancing the
                # remaining groups.
                preferred='O_LEVEL' if name=='validation' else 'A_LEVEL'
                group_id,rows=min(candidates,key=lambda item:(0 if Counter(row.get('level') for row in item[1])[preferred] else 1,len(item[1])))
            else:
                group_id,rows=min(candidates,key=lambda item:len(item[1]))
            splits[name].extend(rows); assigned.add(group_id)
    for group_id,rows in grouped:
        if group_id in assigned: continue
        choices=[name for name in targets if len(splits[name])+len(rows)<=targets[name]]
        if not choices: raise SystemExit(f'Group {group_id} ({len(rows)} records) cannot fit without source leakage.')
        total_categories=Counter(row.get('category') for row in eligible)
        group_levels=Counter(row.get('level') for row in rows); group_categories=Counter(row.get('category') for row in rows)
        def fit_score(name):
            current_levels=Counter(row.get('level') for row in splits[name]); current_categories=Counter(row.get('category') for row in splits[name])
            level_target=targets[name]/2
            level_gain=sum(min(count,max(0,level_target-current_levels[level]))/max(1,level_target) for level,count in group_levels.items())
            category_gain=0.0
            for category,count in group_categories.items():
                desired=total_categories[category]*targets[name]/len(eligible)
                category_gain += min(count,max(0,desired-current_categories[category]))/max(1,desired)
            remaining=(targets[name]-len(splits[name]))/targets[name]
            return 3*level_gain+category_gain+.01*remaining
        name=max(choices,key=fit_score)
        splits[name].extend(rows)
    if {name:len(rows) for name,rows in splits.items()}!=targets:
        raise SystemExit(f'Could not satisfy exact split sizes with intact groups: { {name:len(rows) for name,rows in splits.items()} }')
    for name, rows in splits.items():
        path=ROOT/'evaluation/datasets'/f'{name}.jsonl'
        if path.exists() and path.stat().st_size and not args.force:
            raise SystemExit(f'{path} already exists; use --force after reviewing the replacement.')
        write_jsonl(path, rows)
    print({'eligible':len(eligible), 'written':{name:len(rows) for name,rows in splits.items()}, 'seed':args.seed})


if __name__ == '__main__':
    main()
