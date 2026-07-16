from src.chunking.pipeline import _question_chunks, _scheme_chunks

DOC={'document_id':'test','document_type':'QUESTION_PAPER','level':'A_LEVEL','subject_code':'9618','year':'2024','session':'MJ','component':'11','authority_level':'4'}
def test_question_parser_ignores_printed_page_number():
    pages=[{'page_number':1,'clean_text':'1\n(a) Question one text\n[1]\n[Page 2]\n2\n9618/11/M/J/24\n[Turn over]\n2\n(a) Question two text'}]
    assert [x['question_number'] for x in _question_chunks(DOC,pages)] == ['1(a)','2(a)']
def test_question_parser_keeps_questions_without_subparts_and_inline_numbers():
    pages=[{'page_number':1,'clean_text':'1 Tick one box to identify the answer.\n[1]\n2\nState two examples.\n[2]\n3 A final question.\n(a) Explain the result.\n[2]'}]
    assert [x['question_number'] for x in _question_chunks(DOC,pages)] == ['1','2','3(a)']
def test_mark_scheme_parser_ignores_mark_counts():
    pages=[{'page_number':1,'clean_text':'Question\n1(a)\n1 mark for answer A\n1(b)\n1 mark for answer B\n2(a)\n2 marks for answer C'}]
    assert [x['question_number'] for x in _scheme_chunks({**DOC,'document_type':'MARK_SCHEME'},pages)] == ['1(a)','1(b)','2(a)']
