from src.generation.answer_generator import answer_question
def test_end_to_end_answer_has_contract_fields():
    answer=answer_question('Define virtual memory.', 'A_LEVEL')
    assert {'answer','answer_type','citations','conversation_id','citation_valid','generation_provider'} <= answer.keys()
    assert answer['answer_type']=='CURRICULUM_EXPLANATION'
    assert answer['generation_provider'] in {'retrieval_only','openai'}

def test_no_key_returns_grounded_retrieval_only_answer(monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY',raising=False)
    answer=answer_question('Define a compiler.', 'O_LEVEL')
    assert answer['generation_provider']=='retrieval_only'
    assert answer['citation_valid']

def test_empty_grounded_generation_becomes_insufficient_source(monkeypatch):
    import src.generation.answer_generator as module
    monkeypatch.setenv('OPENAI_API_KEY','test-key')
    monkeypatch.setattr(module,'_generate_with_openai',lambda *args: ('Unsupported lunar cheese claim.',{'input_tokens':1,'output_tokens':1}))
    answer=module.answer_question('Define a compiler.','O_LEVEL')
    assert answer['answer_type']=='INSUFFICIENT_SOURCE'
    assert answer['answer'].startswith('I could not find enough information')

def test_explicit_model_abstention_has_insufficient_status(monkeypatch):
    import src.generation.answer_generator as module
    monkeypatch.setenv('OPENAI_API_KEY','test-key')
    monkeypatch.setattr(module,'_generate_with_openai',lambda *args: ('There is insufficient evidence in the supplied sources [o_level_o_levels_book_pdf p.76].',{'input_tokens':1,'output_tokens':1}))
    answer=module.answer_question('Define a compiler.','O_LEVEL')
    assert answer['answer_type']=='INSUFFICIENT_SOURCE'
