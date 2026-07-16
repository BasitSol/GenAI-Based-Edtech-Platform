"""Create a traceable, review-gated supplement for the Phase 1 benchmark.

This script never approves a record.  It produces 69 source-linked review
items, bringing the existing 151 exact-paper candidates to the planned 220
records once the review queue is completed.
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import ROOT, read_jsonl, write_jsonl


BLUEPRINTS = [
    ('DEFINITION', 'TEXTBOOK', 'Define the key term or concept in the cited curriculum source.'),
    ('CONCEPT_EXPLANATION', 'TEXTBOOK', 'Explain the main concept described in the cited curriculum source.'),
    ('COMPARISON', 'TEXTBOOK', 'Compare the related concepts discussed in the cited curriculum source.'),
    ('SYLLABUS_SCOPE', 'SYLLABUS', 'What does the syllabus require a learner to understand in this cited objective?'),
    ('PSEUDOCODE', 'TEXTBOOK', 'Write or explain the pseudocode/programming technique demonstrated in the cited source.'),
    ('CALCULATION', 'TEXTBOOK', 'Solve the calculation or data-representation task supported by the cited source.'),
    ('EXAM_ANSWER', 'MARK_SCHEME', 'Answer the linked exam question using its exact mark scheme.'),
    ('EXAM_MODEL_ANSWER', 'QUESTION_PAPER', 'Answer the linked exam question using curriculum evidence; disclose when no exact scheme exists.'),
    ('MULTI_SOURCE', 'SYLLABUS', 'Explain the topic using both the syllabus objective and the supporting textbook source.'),
]


def usable(chunks: list[dict], document_type: str, level: str) -> list[dict]:
    return [
        chunk for chunk in chunks
        if chunk.get('document_type') == document_type
        and chunk.get('level') == level
        and chunk.get('content_type') != 'PARENT_CONTEXT'
        and len(chunk.get('text', '').split()) >= 40
    ]


def source(chunk: dict) -> dict:
    return {
        'document_id': chunk['document_id'],
        'page_start': chunk['page_start'],
        'page_end': chunk['page_end'],
        'chunk_id': chunk['chunk_id'],
    }


def excerpt(chunk: dict, words: int = 80) -> str:
    return ' '.join(chunk.get('text', '').split()[:words])


def main() -> None:
    parser = argparse.ArgumentParser(description='Create review-only mixed-source benchmark candidates.')
    parser.add_argument('--limit', type=int, default=69)
    parser.add_argument('--output', type=Path, default=ROOT / 'evaluation/datasets/mixed_candidates_requires_review.jsonl')
    args = parser.parse_args()
    chunks = read_jsonl(ROOT / 'data_processed/chunks/all_chunks.jsonl')
    pools = {(kind, level): usable(chunks, kind, level) for kind in {item[1] for item in BLUEPRINTS} for level in ('O_LEVEL', 'A_LEVEL')}
    cursors = {key: itertools.cycle(value) for key, value in pools.items() if value}
    records=[]
    for index in range(args.limit):
        intent, kind, prompt = BLUEPRINTS[index % len(BLUEPRINTS)]
        level = 'O_LEVEL' if index % 2 == 0 else 'A_LEVEL'
        iterator = cursors.get((kind, level))
        if iterator is None:
            raise RuntimeError(f'No usable {kind} chunks for {level}')
        primary = next(iterator)
        record={
            'id': f'MIXED_{index + 1:03d}', 'question': f'[REVIEW REQUIRED] {prompt}',
            'level': level, 'intent': intent, 'answerable': intent != 'UNANSWERABLE',
            'expected_answer_type': None, 'expected_source_types': [kind],
            'gold_curriculum_sources': [source(primary)], 'required_key_points': [],
            'review_status': 'REQUIRES_REVIEW',
            'review_notes': 'Replace the draft question, verify the page/chunk, add key points and expected answer status before approval.',
            'source_excerpt_for_reviewer': excerpt(primary),
        }
        if intent == 'EXAM_ANSWER':
            record['exact_mark_scheme_available'] = True
            record['expected_answer_type'] = 'OFFICIAL_MARK_SCHEME_SUPPORTED_ANSWER'
            record['review_notes'] += ' Add the matching QUESTION_PAPER source; do not approve unless the pair is exact.'
        elif intent == 'EXAM_MODEL_ANSWER':
            record['exact_mark_scheme_available'] = False
            record['expected_answer_type'] = 'AI_GENERATED_MODEL_ANSWER'
        elif intent == 'MULTI_SOURCE':
            secondary = next(cursors[('TEXTBOOK', level)])
            record['expected_source_types'] = ['SYLLABUS', 'TEXTBOOK']
            record['gold_curriculum_sources'].append(source(secondary))
        else:
            record['expected_answer_type'] = 'CURRICULUM_EXPLANATION'
        records.append(record)
    write_jsonl(args.output, records)
    print({'candidates': len(records), 'status': 'REQUIRES_REVIEW', 'path': str(args.output)})


if __name__ == '__main__':
    main()
