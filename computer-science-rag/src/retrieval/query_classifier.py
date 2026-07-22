"""Layered question understanding with optional schema-constrained LLM review."""
from __future__ import annotations

import json
import os
import re
from typing import Any

from .paper_reference_parser import parse_reference


CATEGORY_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("MCQ", (r"(?:^|\n)\s*(?:[A-D][.)]|\([A-D]\))\s+",)),
    ("FILL_IN_THE_BLANK", (r"_{3,}|\bfill (?:in )?the blank\b",)),
    ("SQL", (r"\b(?:SQL|SELECT|INSERT|UPDATE|DELETE FROM|CREATE TABLE|JOIN|primary key|foreign key)\b",)),
    ("DEBUGGING", (r"\b(?:debug|fix|correct).*\b(?:code|program|algorithm)\b|\b(?:syntax|logic|runtime) error\b",)),
    ("TRACE_TABLE", (r"\b(?:trace table|dry[ -]?run|trace the algorithm|variable values)\b",)),
    ("PROGRAMMING", (r"\b(?:pseudocode|algorithm|procedure|function|recursion|binary search|linear search|bubble sort|insertion sort|write (?:a )?program|Python)\b",)),
    ("CALCULATION", (r"\b(?:calculate|convert|show your working|binary addition|hexadecimal conversion|floating[ -]?point)\b",)),
    ("COMPARISON", (r"\b(?:compare|contrast|difference between|similarities and differences)\b",)),
    ("DEFINITION", (r"^\s*(?:define|what is|what does .+ mean)\b",)),
    ("EXAMINER_FEEDBACK", (r"\b(?:examiner report|candidate mistake|common mistake)\b",)),
    ("SYLLABUS", (r"\b(?:syllabus|learning objective|assessment objective)\b",)),
]


def _deterministic_category(query: str, metadata: dict) -> tuple[str, float, list[str]]:
    evidence: list[str] = []
    matches: list[tuple[int, int, str]] = []
    for priority, (category, patterns) in enumerate(CATEGORY_RULES):
        count = sum(bool(re.search(pattern, query, re.I)) for pattern in patterns)
        if count:
            matches.append((count, -priority, category))
            evidence.append(f"rule:{category.lower()}")
    if metadata.get("question_number") and metadata.get("component"):
        return "EXAM_QUESTION", 0.99, ["structured_paper_reference"]
    if not matches:
        return "THEORY", 0.64, ["default_conceptual"]
    matches.sort(reverse=True)
    category = matches[0][2]
    return category, min(0.96, 0.74 + 0.08 * matches[0][0]), evidence


def _intent(category: str, metadata: dict, query: str) -> str:
    if metadata.get("question_number") and metadata.get("component"):
        return "EXAM_ANSWER"
    return {
        "SYLLABUS": "SYLLABUS_QUERY",
        "EXAMINER_FEEDBACK": "EXAMINER_FEEDBACK",
        "COMPARISON": "CONCEPT_EXPLANATION",
        "DEFINITION": "CONCEPT_EXPLANATION",
        "THEORY": "CONCEPT_EXPLANATION",
        "PROGRAMMING": "CODE_EXPLANATION",
        "SQL": "CODE_GENERATION",
        "DEBUGGING": "CODE_DEBUGGING",
        "TRACE_TABLE": "EXECUTION_TRACE",
        "CALCULATION": "WORKED_CALCULATION",
        "MCQ": "OPTION_SELECTION",
        "FILL_IN_THE_BLANK": "CONCISE_COMPLETION",
    }.get(category, "CONCEPT_EXPLANATION")


def _difficulty(query: str) -> str:
    if re.search(r"\b(?:evaluate|justify|analyse|complexity|recursive|normal form)\b", query, re.I):
        return "ADVANCED"
    if re.search(r"^\s*(?:state|define|identify|name|what is)\b", query, re.I):
        return "BEGINNER"
    return "INTERMEDIATE"


def _llm_review(query: str, baseline: dict[str, Any]) -> dict[str, Any] | None:
    """Review only genuinely ambiguous profiles; never make classification a mandatory paid call."""
    if os.getenv("INTENT_CLASSIFIER_MODE", "adaptive").lower() != "adaptive":
        return None
    if baseline["classification_confidence"] >= float(os.getenv("INTENT_LLM_THRESHOLD", "0.68")):
        return None
    if not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        from openai import OpenAI
        from src.generation.prompts import prompt_text
        schema = {
            "type": "object",
            "properties": {
                "category": {"type": "string", "enum": [item[0] for item in CATEGORY_RULES] + ["THEORY"]},
                "difficulty": {"type": "string", "enum": ["BEGINNER", "INTERMEDIATE", "ADVANCED"]},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["category", "difficulty", "confidence"],
            "additionalProperties": False,
        }
        response = OpenAI(api_key=os.environ["OPENAI_API_KEY"]).chat.completions.create(
            model=os.getenv("CLASSIFIER_MODEL", os.getenv("GENERATOR_MODEL", "gpt-4.1-mini")),
            temperature=0,
            max_tokens=180,
            response_format={"type": "json_schema", "json_schema": {"name": "question_profile", "strict": True, "schema": schema}},
            messages=[
                {"role": "system", "content": prompt_text("query_understanding")},
                {"role": "user", "content": json.dumps({"query": query, "baseline": baseline})},
            ],
        )
        result = json.loads(response.choices[0].message.content)
        return {"category": result["category"], "difficulty": result["difficulty"], "classification_confidence": result["confidence"], "classifier_provider": "openai_structured"}
    except Exception as exc:
        baseline["classifier_error"] = f"{type(exc).__name__}: {str(exc)[:180]}"
        return None


def classify(query: str, level: str | None = None, exam_year: int | None = None) -> dict[str, Any]:
    metadata = parse_reference(query)
    inferred_level = level or ("A_LEVEL" if metadata.get("subject_code") == "9618" else "O_LEVEL" if metadata.get("subject_code") == "2210" else None)
    category, confidence, evidence = _deterministic_category(query, metadata)
    profile: dict[str, Any] = {
        "category": category,
        "intent": _intent(category, metadata, query),
        "difficulty": _difficulty(query),
        "educational_objective": "ANALYSE" if category in {"COMPARISON", "DEBUGGING", "TRACE_TABLE"} else "APPLY" if category in {"PROGRAMMING", "SQL", "CALCULATION"} else "UNDERSTAND",
        "answer_style": category.lower(),
        "needs_reasoning": category in {"COMPARISON", "PROGRAMMING", "SQL", "DEBUGGING", "TRACE_TABLE", "CALCULATION", "MCQ"},
        "needs_citations": True,
        "needs_code": category in {"PROGRAMMING", "SQL", "DEBUGGING"},
        "classification_confidence": confidence,
        "classification_evidence": evidence,
        "classifier_provider": "layered_rules",
        "level": inferred_level,
        "exam_year": exam_year,
        **metadata,
    }
    reviewed = _llm_review(query, profile)
    if reviewed:
        profile.update(reviewed)
        profile["intent"] = _intent(profile["category"], metadata, query)
    return profile
