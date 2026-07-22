"""Bounded LangGraph retrieval workflow with one corrective retry."""
from __future__ import annotations

import re
import time
from typing import TypedDict

from src.core import tokens
from src.memory.followup_rewriter import rewrite_followup
from src.retrieval.context_builder import build_context
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import rerank_with_debug


STOPWORDS = {"what", "when", "where", "which", "that", "this", "with", "from", "into", "about", "explain", "define", "answer", "question", "using", "write", "show"}


class RetrievalState(TypedDict, total=False):
    query: str
    standalone_query: str
    level: str | None
    exam_year: int | None
    conversation_state: dict
    raw: dict
    ranked: list[dict]
    sufficient: bool
    retry_count: int
    accumulated_embedding_cost: float
    accumulated_embedding_tokens: int
    result: dict


def _coverage(query: str, chunks: list[dict]) -> float:
    terms = {term for term in tokens(query) if len(term) > 2 and term not in STOPWORDS}
    if not terms:
        return 0.0
    factual = [item for item in chunks if item.get("document_type") in {"TEXTBOOK", "SYLLABUS", "EXAMINER_REPORT"}]
    return max((len(terms & set(tokens(item.get("text", "")))) / len(terms) for item in factual), default=0.0)


def _corrective_query(query: str, category: str) -> str:
    cleaned = re.sub(r"\b(?:please|could you|explain|describe|write|answer|question)\b", " ", query, flags=re.I)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .:-")
    suffix = {
        "PROGRAMMING": "algorithm prerequisite steps pseudocode",
        "SQL": "SQL syntax SELECT FROM WHERE example",
        "COMPARISON": "definition differences advantages disadvantages",
        "CALCULATION": "method formula worked example",
    }.get(category, "definition explanation")
    return f"{cleaned} {suffix}".strip()


def _graph(retriever: HybridRetriever, maximum_chunks: int | None):
    from langgraph.graph import END, START, StateGraph

    def understand(state: RetrievalState) -> dict:
        return {"standalone_query": rewrite_followup(state["query"], state.get("conversation_state", {})),
                "retry_count": 0, "accumulated_embedding_cost": 0.0, "accumulated_embedding_tokens": 0}

    def retrieve_candidates(state: RetrievalState) -> dict:
        query = state["standalone_query"]
        raw = retriever.retrieve(query, state.get("level"), state.get("exam_year"), result_limit=24)
        debug = raw.get("retrieval_debug", {})
        return {"raw": raw,
                "accumulated_embedding_cost": state.get("accumulated_embedding_cost", 0.0) + float(debug.get("embedding_cost", 0.0)),
                "accumulated_embedding_tokens": state.get("accumulated_embedding_tokens", 0) + int(debug.get("embedding_input_tokens", 0))}

    def rerank(state: RetrievalState) -> dict:
        raw = state["raw"]
        plan = raw["retrieval_debug"]["retrieval_plan"]
        top_k = maximum_chunks or int(plan["context_chunks"])
        exact = [item for item in raw["chunks"] if item.get("retrieval_route") == "exact_metadata"]
        semantic = [item for item in raw["chunks"] if item.get("retrieval_route") != "exact_metadata"]
        # Paper identifiers are poor semantic queries.  When an exact question
        # is resolved, use its wording to rerank supporting curriculum while
        # preserving every deterministic exact hit ahead of semantic results.
        question = next((item.get("text", "") for item in exact if item.get("document_type") == "QUESTION_PAPER"), "")
        ranked_semantic, debug = rerank_with_debug(question or state["standalone_query"], semantic, max(top_k, 8))
        ranked = exact + [item for item in ranked_semantic if item.get("chunk_id") not in {value.get("chunk_id") for value in exact}]
        raw["retrieval_debug"]["reranker"] = debug
        return {"ranked": ranked}

    def check(state: RetrievalState) -> dict:
        raw = state["raw"]
        overlap = _coverage(state["standalone_query"], state["ranked"])
        sufficient = bool(state["ranked"]) and (raw["exact_mark_scheme_available"] or overlap >= 0.20)
        raw["retrieval_debug"].update({"evidence_coverage": round(overlap, 3), "context_sufficient": sufficient, "retry_count": state.get("retry_count", 0)})
        return {"sufficient": sufficient}

    def route_after_check(state: RetrievalState) -> str:
        return "assemble" if state["sufficient"] or state.get("retry_count", 0) >= 1 else "correct"

    def correct(state: RetrievalState) -> dict:
        category = state["raw"]["route"]["category"]
        return {"standalone_query": _corrective_query(state["standalone_query"], category), "retry_count": 1}

    def assemble(state: RetrievalState) -> dict:
        raw = state["raw"]
        raw["retrieval_debug"]["embedding_cost"] = round(state.get("accumulated_embedding_cost", 0.0), 8)
        raw["retrieval_debug"]["embedding_input_tokens_total"] = state.get("accumulated_embedding_tokens", 0)
        if raw["exact_mark_scheme_available"]:
            # Official evidence must remain byte-for-byte complete.  It is not
            # compressed or truncated to the generative context budget.
            raw["chunks"] = [item for item in state["ranked"] if item.get("retrieval_route") == "exact_metadata"]
            raw["standalone_query"] = state["standalone_query"]
            raw["retrieval_debug"]["context_chars"] = sum(len(item.get("text", "")) for item in raw["chunks"])
            return {"result": raw}
        plan = raw["retrieval_debug"]["retrieval_plan"]
        limit = maximum_chunks or int(plan["context_chunks"])
        expanded: list[dict] = []
        for item in state["ranked"]:
            expanded.append(item)
            parent = retriever.metadata.parent(item.get("parent_chunk_id"))
            if parent and all(existing["chunk_id"] != parent["chunk_id"] for existing in expanded):
                parent["relationship"] = "PARENT_CONTEXT"
                expanded.append(parent)
        context = build_context(expanded, state["standalone_query"], maximum_chunks=limit, maximum_chars=int(plan["context_chars"])) if state["sufficient"] else []
        raw["chunks"] = context
        raw["standalone_query"] = state["standalone_query"]
        raw["retrieval_debug"]["context_chars"] = sum(len(item.get("text", "")) for item in context)
        return {"result": raw}

    graph = StateGraph(RetrievalState)
    graph.add_node("understand", understand)
    graph.add_node("retrieve", retrieve_candidates)
    graph.add_node("rerank", rerank)
    graph.add_node("check", check)
    graph.add_node("correct", correct)
    graph.add_node("assemble", assemble)
    graph.add_edge(START, "understand")
    graph.add_edge("understand", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "check")
    graph.add_conditional_edges("check", route_after_check, {"assemble": "assemble", "correct": "correct"})
    graph.add_edge("correct", "retrieve")
    graph.add_edge("assemble", END)
    return graph.compile()


def retrieve(query: str, level: str | None = None, exam_year: int | None = None, conversation_state: dict | None = None, maximum_chunks: int | None = None, retriever: HybridRetriever | None = None) -> dict:
    started = time.perf_counter()
    system = retriever or HybridRetriever()
    result = _graph(system, maximum_chunks).invoke({
        "query": query, "level": level, "exam_year": exam_year,
        "conversation_state": conversation_state or {},
    })["result"]
    result["retrieval_debug"]["workflow_latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
    return result
