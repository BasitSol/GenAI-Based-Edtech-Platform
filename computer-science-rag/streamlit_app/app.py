import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import streamlit as st
from src.generation.answer_generator import answer_question
st.set_page_config(page_title='Computer Science RAG'); st.title('Computer Science RAG Assistant')
level=st.selectbox('Level',['O_LEVEL','A_LEVEL']); year=st.number_input('Exam year',min_value=2020,max_value=2035,value=2025); query=st.text_area('Question')
difficulty=st.selectbox('Explanation level',['Beginner','Intermediate','Advanced'])
if st.button('Ask') and query:
 result=answer_question(query,level,int(year)); st.caption(f"{result['answer_type']} · confidence {result['confidence']} · {difficulty}"); st.write(result['answer'])
 st.write(f"Generation provider: {result['generation_provider']}" + (f" ({result['generator_model']})" if result['generator_model'] else ""))
 st.write(f"Mark scheme status: {'Exact matching scheme available' if result['exact_mark_scheme_available'] else 'Exact matching scheme not available'}")
 if result['disclosure']: st.warning(result['disclosure'])
 with st.expander('Sources'): st.json(result['citations'])
 with st.expander('Developer debug'): st.json(result['retrieval_debug'])
