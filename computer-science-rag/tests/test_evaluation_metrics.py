from evaluation.retrieval_eval import _dcg
def test_dcg_is_normalized_by_matching_ideal_count():
    relevances=[1,1,1]
    assert _dcg(relevances)/_dcg([1]*sum(relevances)) == 1.0
