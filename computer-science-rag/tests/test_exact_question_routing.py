from src.retrieval.hybrid_retriever import HybridRetriever

def test_a_level_exact_question_is_limited_to_requested_paper_and_question():
    result=HybridRetriever().retrieve('9618/11/M/J/24 Question 1','A_LEVEL')
    chunks=result['chunks']
    assert result['exact_mark_scheme_available']
    assert chunks
    assert all(chunk['subject_code']=='9618' and chunk['year']=='2024' and chunk['session']=='MJ' and chunk['component']=='11' for chunk in chunks)
    assert all(chunk.get('question_number')=='1' or str(chunk.get('question_number','')).startswith('1(') for chunk in chunks)
    assert not any('video doorbell' in chunk['text'].lower() for chunk in chunks)

def test_missing_exact_scheme_is_not_mislabeled_as_official():
    result=HybridRetriever().retrieve('9618/11/M/J/25 Question 1','A_LEVEL')
    assert not result['exact_mark_scheme_available']
    assert not any(chunk['document_type']=='MARK_SCHEME' for chunk in result['chunks'])
