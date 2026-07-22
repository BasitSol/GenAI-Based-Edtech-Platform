"""Student chat and local operational dashboard."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from src.generation.answer_generator import answer_question
from src.observability.telemetry import TelemetryStore
from src.observability.tracing import langsmith_status
from evaluation.live_excel import append_live_answer


st.set_page_config(page_title="Computer Science RAG", layout="wide")
st.title("Enterprise Computer Science RAG Assistant")
chat_tab, monitoring_tab = st.tabs(["Student chat", "Production monitoring"])

with chat_tab:
    level_choice = st.selectbox("Level", ["AUTO", "O_LEVEL", "A_LEVEL"])
    level = None if level_choice == "AUTO" else level_choice
    year = st.number_input("Exam year", min_value=2020, max_value=2035, value=2025)
    difficulty = st.selectbox("Explanation level", ["Beginner", "Intermediate", "Advanced"])
    query = st.text_area("Question")
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = None
    if st.session_state.get("active_level_choice") != level_choice:
        st.session_state.conversation_id = None
        st.session_state.active_level_choice = level_choice
    if st.button("Ask") and query:
        result = answer_question(query, level, int(year), st.session_state.conversation_id, difficulty)
        live_path = append_live_answer({**result, "question": query})
        st.session_state.conversation_id = result["conversation_id"]
        profile = result["question_understanding"]
        st.caption(f"{result['answer_type']} · {profile['category']} · {difficulty} · {result['execution_status']}")
        st.markdown(result["answer"])
        st.write(f"Generation provider: {result['generation_provider']}" + (f" ({result['generator_model']})" if result["generator_model"] else ""))
        st.write(f"Mark scheme: {'exact match available' if result['exact_mark_scheme_available'] else 'exact match not available'}")
        st.success(f"Saved to `{live_path.relative_to(live_path.parents[2])}`")
        st.download_button("Download live evaluation workbook", live_path.read_bytes(), file_name="live_answers.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        with st.expander("Sources"):
            if result["citations"]:
                for number, reference in enumerate(result["citations"], 1):
                    st.markdown(f"**[{number}]** `{reference['document_id']}`, page {reference['page']} — `{reference['chunk_id']}`")
            else:
                st.caption("No source was cited in this response.")
            st.dataframe(result["source_details"], use_container_width=True, hide_index=True)
        with st.expander("Question understanding"):
            st.json(profile)
        with st.expander("Verification and developer trace"):
            st.json({"trace_id": result["trace_id"], "verification": result["verification"],
                     "retrieval": result["retrieval_debug"], "estimated_cost": result["estimated_cost"],
                     "generation_error": result["generation_error"]})

with monitoring_tab:
    days = st.slider("Monitoring window (days)", 1, 90, 30)
    try:
        summary = TelemetryStore().summary(days)
        columns = st.columns(4)
        columns[0].metric("Requests", summary["total_requests"])
        columns[1].metric("Active sessions", summary["active_sessions"])
        columns[2].metric("Average latency", f"{summary['average_latency_ms'] or 0:.0f} ms")
        columns[3].metric("Total cost", f"${summary['total_cost']:.4f}")
        st.subheader("Question categories")
        st.bar_chart(summary["question_categories"])
        st.subheader("Failure categories")
        st.bar_chart(summary["failure_categories"])
        st.json({"local_telemetry": summary, "langsmith": langsmith_status()})
    except Exception as exc:
        st.info(f"Monitoring data is not available yet: {exc}")
