from src.memory.followup_rewriter import rewrite_followup
def test_followup_rewriter_uses_relevant_previous_turn():
    state={'recent_messages':[{'role':'user','content':"Explain two's complement."}]}
    assert "two's complement" in rewrite_followup('Convert 11101010 using it.',state)
