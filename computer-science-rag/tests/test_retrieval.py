from src.retrieval.paper_reference_parser import parse_reference
from src.generation.answer_generator import answer_question
def test_reference_parser():
 value=parse_reference('Answer 9618/22/M/J/24 Question 3(b)')
 assert value['subject_code']=='9618' and value['component']=='22' and value['year']==2024 and value['question_number']=='3(b)'
def test_exact_scheme_requires_full_paper_identity():
 assert not answer_question('Define a compiler.', 'O_LEVEL')['exact_mark_scheme_available']
 assert answer_question('Answer 9618/11/M/J/24 Question 1.', 'A_LEVEL')['exact_mark_scheme_available']
def test_exact_paper_does_not_mix_other_components():
 from src.retrieval.hybrid_retriever import HybridRetriever
 chunks=HybridRetriever().retrieve('9618/11/M/J/24 Question 1','A_LEVEL')['chunks']
 assert all(chunk['component']=='11' for chunk in chunks)
def test_authority_filter_preserves_fused_semantic_order():
 from src.retrieval.authority_controller import filter_and_sort
 chunks=[{'chunk_id':'book','document_type':'TEXTBOOK'},{'chunk_id':'syllabus','document_type':'SYLLABUS'}]
 assert [item['chunk_id'] for item in filter_and_sort(chunks,'DEFINITION')]==['book','syllabus']
def test_no_scheme_exam_answer_adds_curriculum_support():
 from src.retrieval.hybrid_retriever import HybridRetriever
 result=HybridRetriever().retrieve('Answer 2210/22/M/J/23 Question 1.','O_LEVEL')
 types={item['document_type'] for item in result['chunks']}
 assert 'QUESTION_PAPER' in types
 assert types & {'TEXTBOOK','SYLLABUS'}
 assert 'MARKING_PATTERN' in types
def test_oversized_chunk_is_truncated_instead_of_erasing_context():
 from src.retrieval.context_builder import build_context
 result=build_context([{'chunk_id':'large','text':'x'*20}],maximum_chunks=1,maximum_chars=5)
 assert result and result[0]['text']=='xxxxx'
def test_diagram_reference_preserves_exact_paper_and_curriculum_context():
 from src.workflows.rag_graph import retrieve
 result=retrieve('Using the supplied figure, answer 2210/22/m/j/23 question 1.','O_LEVEL',2023,{})
 types={item['document_type'] for item in result['chunks']}
 assert result['retrieval_debug']['context_sufficient']
 assert 'QUESTION_PAPER' in types
 assert types & {'TEXTBOOK','SYLLABUS'}
 assert 'MARKING_PATTERN' in types
def test_no_scheme_support_uses_core_question_terms():
 from src.workflows.rag_graph import retrieve
 result=retrieve('Answer 9618/13/O/N/23 Question 1(a).','A_LEVEL',2023,{})
 curriculum=[item for item in result['chunks'] if item['document_type'] in {'TEXTBOOK','SYLLABUS'}]
 assert curriculum and any('analogue' in item['text'].lower() for item in curriculum)
 assert 'analogue' in result['retrieval_debug']['semantic_query']
