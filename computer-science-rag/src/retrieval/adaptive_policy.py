"""Question-category-specific retrieval planning and low-cost query expansion."""
from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RetrievalPlan:
    dense_k: int = 80
    sparse_k: int = 80
    fusion_k: int = 24
    rerank_k: int = 6
    context_chunks: int = 6
    context_chars: int = 12000
    preferred_types: tuple[str, ...] = ("TEXTBOOK", "SYLLABUS")
    multi_query: bool = False
    strategy: str = "hybrid_balanced"

    def to_dict(self) -> dict:
        data = asdict(self)
        data["preferred_types"] = list(self.preferred_types)
        return data


POLICIES = {
    "THEORY": RetrievalPlan(strategy="hybrid_conceptual"),
    "DEFINITION": RetrievalPlan(context_chunks=5, context_chars=9000, strategy="concise_conceptual"),
    "COMPARISON": RetrievalPlan(context_chunks=8, context_chars=15000, multi_query=True, strategy="multi_aspect_comparison"),
    "PROGRAMMING": RetrievalPlan(context_chunks=8, context_chars=16000, preferred_types=("TEXTBOOK", "SYLLABUS", "QUESTION_PAPER"), multi_query=True, strategy="code_aware"),
    "SQL": RetrievalPlan(context_chunks=8, context_chars=16000, preferred_types=("TEXTBOOK", "SYLLABUS", "QUESTION_PAPER"), multi_query=True, strategy="sql_aware"),
    "DEBUGGING": RetrievalPlan(context_chunks=8, context_chars=16000, preferred_types=("TEXTBOOK", "QUESTION_PAPER", "SYLLABUS"), multi_query=True, strategy="debugging"),
    "TRACE_TABLE": RetrievalPlan(context_chunks=8, context_chars=16000, preferred_types=("QUESTION_PAPER", "TEXTBOOK", "MARK_SCHEME"), multi_query=True, strategy="trace_execution"),
    "CALCULATION": RetrievalPlan(context_chunks=7, context_chars=13000, preferred_types=("TEXTBOOK", "QUESTION_PAPER", "MARK_SCHEME"), strategy="worked_calculation"),
    "MCQ": RetrievalPlan(context_chunks=5, context_chars=9000, preferred_types=("QUESTION_PAPER", "MARK_SCHEME", "TEXTBOOK", "SYLLABUS"), multi_query=True, strategy="mcq_evidence"),
    "FILL_IN_THE_BLANK": RetrievalPlan(context_chunks=5, context_chars=9000, preferred_types=("QUESTION_PAPER", "MARK_SCHEME", "TEXTBOOK"), strategy="concise_completion"),
    "EXAM_QUESTION": RetrievalPlan(context_chunks=8, context_chars=15000, preferred_types=("QUESTION_PAPER", "MARK_SCHEME", "TEXTBOOK", "SYLLABUS", "MARKING_PATTERN"), strategy="exam_authority"),
}


def plan_for(route: dict, requested_chunks: int | None = None) -> RetrievalPlan:
    category = route.get("category", "THEORY")
    plan = POLICIES.get(category, POLICIES["THEORY"])
    if route.get("intent") == "EXAM_ANSWER":
        plan = POLICIES["EXAM_QUESTION"]
    if requested_chunks:
        plan = RetrievalPlan(**{**asdict(plan), "rerank_k": requested_chunks, "context_chunks": requested_chunks})
    return plan


def query_variants(query: str, route: dict, plan: RetrievalPlan) -> list[str]:
    variants = [query.strip()]
    enabled = os.getenv("ADAPTIVE_MULTI_QUERY_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
    if not enabled or not plan.multi_query:
        return variants
    cleaned = re.sub(r"\b(?:answer|question|marks?|explain|state|give|write|show your working)\b", " ", query, flags=re.I)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .:-")
    if cleaned and cleaned.lower() != query.strip().lower():
        variants.append(cleaned)
    category = route.get("category")
    suffix = {
        "SQL": "SQL syntax example database",
        "PROGRAMMING": "algorithm pseudocode worked example",
        "DEBUGGING": "program error correction explanation",
        "TRACE_TABLE": "algorithm trace variable values",
        "MCQ": "definition concept evidence",
        "COMPARISON": "similarities differences advantages disadvantages",
    }.get(category)
    if suffix and len(variants) < 2:
        variants.append(f"{cleaned or query} {suffix}")
    return list(dict.fromkeys(item for item in variants if item))[:2]


def type_priority(chunk: dict, plan: RetrievalPlan) -> int:
    try:
        return len(plan.preferred_types) - plan.preferred_types.index(chunk.get("document_type"))
    except ValueError:
        return 0
