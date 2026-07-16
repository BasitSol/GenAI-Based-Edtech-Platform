from src.generation.citation_validator import validate
from evaluation.citation_eval import evaluate_answer
from src.generation.grounding_checker import enforce_grounding, claim_citation_coverage
def test_citations_are_linked_to_chunks():
 c={'chunk_id':'a','document_id':'d','page_start':2}; assert validate([{'chunk_id':'a','document_id':'d','page':2}],[c])
def test_gold_chunk_citation_scores_as_precise():
 answer={'answer':'Supported statement [d p.2].','citations':[{'chunk_id':'a','document_id':'d','page':2}],'retrieved_chunks':[{'chunk_id':'a','document_id':'d','page_start':2,'text':'This is a supported statement.'}]}
 record={'gold_curriculum_sources':[{'chunk_id':'a','document_id':'d','page_start':2,'page_end':2}]}
 assert evaluate_answer(answer,record)['citation_precision']==1.0
def test_citation_precision_measures_claim_support_not_gold_identity():
 answer={'answer':'A router forwards packets [book p.7].','citations':[{'chunk_id':'book-7','document_id':'book','page':7}],'retrieved_chunks':[{'chunk_id':'book-7','document_id':'book','page_start':7,'text':'Routers forward data packets to their destination.'}]}
 record={'gold_curriculum_sources':[{'chunk_id':'syllabus-2','document_id':'syllabus','page_start':2,'page_end':2}]}
 assert evaluate_answer(answer,record)['citation_precision']==1.0
def test_grounding_guard_cites_supported_and_removes_unsupported_claims():
 chunks=[{'chunk_id':'r1','document_id':'book','page_start':7,'text':'A router forwards data packets to a destination.'}]
 answer=enforce_grounding('A router forwards packets.\nThe moon is made of cheese.',chunks)
 assert answer=='A router forwards packets [book p.7].'
 assert claim_citation_coverage(answer)==1.0
def test_grounding_guard_respects_mark_based_claim_limit():
 chunks=[{'chunk_id':'q1','document_id':'qp','page_start':2,'text':'The first stage is Analysis. Flowcharts are a design method.'}]
 answer=enforce_grounding('The first stage is Analysis.\nOther visible question:\n- Flowcharts',chunks,maximum_claims=1)
 assert answer=='The first stage is Analysis [qp p.2].'
def test_grounding_guard_never_uses_pattern_as_factual_evidence():
 chunks=[
  {'chunk_id':'pattern','document_id':'old_ms','page_start':4,'document_type':'MARKING_PATTERN','text':'Quantum surface codes provide fault tolerance.'},
  {'chunk_id':'book','document_id':'book','page_start':2,'document_type':'TEXTBOOK','text':'Classical parity checks detect transmission errors.'},
 ]
 assert enforce_grounding('Quantum surface codes provide fault tolerance.',chunks)==''
