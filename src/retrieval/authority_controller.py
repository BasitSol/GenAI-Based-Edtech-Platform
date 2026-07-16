from __future__ import annotations
PRIORITY={'SYLLABUS':5,'MARK_SCHEME':5,'TEXTBOOK':4,'EXAMINER_REPORT':4,'QUESTION_PAPER':4,'MARKING_PATTERN':2}
def filter_and_sort(chunks,intent,exact_scheme=False):
    if intent in {'DEFINITION','CONCEPT_EXPLANATION','COMPARISON'}: chunks=[x for x in chunks if x['document_type'] in {'TEXTBOOK','SYLLABUS'}]
    if intent=='SYLLABUS_QUERY': chunks=[x for x in chunks if x['document_type']=='SYLLABUS']
    if intent=='EXAMINER_FEEDBACK': chunks=[x for x in chunks if x['document_type']=='EXAMINER_REPORT']
    if intent in {'PSEUDOCODE','CALCULATION'}: chunks=[x for x in chunks if x['document_type'] in {'TEXTBOOK','SYLLABUS','QUESTION_PAPER'}]
    if intent=='EXAM_ANSWER' and not exact_scheme: chunks=[x for x in chunks if x['document_type']!='MARK_SCHEME' or x.get('relationship')=='EXACT_MATCH']
    if intent=='EXAM_ANSWER' and exact_scheme: chunks=[x for x in chunks if x['document_type']!='MARKING_PATTERN']
    # Authority controls which source types are allowed; it must not erase the
    # semantic/RRF ordering by forcing every syllabus chunk above textbooks.
    return chunks
