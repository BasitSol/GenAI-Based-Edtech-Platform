from __future__ import annotations
from .paper_reference_parser import parse_reference
INTENTS={'SYLLABUS_QUERY':['syllabus','learning objective'],'MARK_SCHEME_EXPLANATION':['mark scheme','marking scheme'],'EXAMINER_FEEDBACK':['examiner report','common mistake'],'PAST_PAPER_SEARCH':['find question','past paper'],'PSEUDOCODE':['pseudocode','declare','algorithm'],'CALCULATION':['calculate','convert','binary','denary'],'COMPARISON':['compare','difference between'],'DEFINITION':['define','what is'],'EXAM_ANSWER':['answer question','exam answer','model answer']}
def classify(query,level=None,exam_year=None):
    low=query.lower(); intent=next((name for name,terms in INTENTS.items() if any(t in low for t in terms)),'CONCEPT_EXPLANATION')
    metadata=parse_reference(query); inferred=level or ('A_LEVEL' if metadata.get('subject_code')=='9618' else 'O_LEVEL' if metadata.get('subject_code')=='2210' else None)
    if metadata.get('question_number') and metadata.get('component'): intent='EXAM_ANSWER'
    return {'intent':intent,'level':inferred,'exam_year':exam_year or metadata.get('year'),**metadata}
