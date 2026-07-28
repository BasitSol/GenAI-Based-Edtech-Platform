"""Bounded LangGraph grading workflow for typed/clearly printed submissions.

Handwriting recognition is intentionally outside this MVP.  Each score is a
draft until a teacher reviews it; the workflow never overwrites human marks.
"""
from __future__ import annotations

import json
import os
import re
from typing import TypedDict

from backend.module1_rag.chat.workflow import retrieve
from backend.module1_rag.retrieval.hybrid_retriever import HybridRetriever


class GradingState(TypedDict, total=False):
    answer_text: str
    assessment: dict
    extracted: dict[int, str]
    aligned: list[dict]
    retrieval_evidence: list[list[dict]]
    evaluation: dict
    result: dict


def extract_answers(answer_text: str, question_numbers: list[int]) -> dict[int, str]:
    """Segment typed answers labelled Q1/Question 1/Answer 1, with safe fallback."""
    pattern = re.compile(r"(?:^|\n)\s*(?:q(?:uestion)?|answer)\s*(\d+)\s*[:.)-]?", re.I)
    matches = list(pattern.finditer(answer_text))
    if not matches:
        return {question_numbers[0]: answer_text.strip()} if question_numbers else {}
    extracted: dict[int, str] = {}
    for index, match in enumerate(matches):
        number = int(match.group(1))
        if number not in question_numbers:
            continue
        end = matches[index + 1].start() if index + 1 < len(matches) else len(answer_text)
        extracted[number] = answer_text[match.end():end].strip()
    return extracted


def _align(assessment: dict, extracted: dict[int, str]) -> list[dict]:
    """Pair student answer segments with the generated question rubric/evidence."""
    aligned = []
    for index, question in enumerate(assessment["content"].get("questions", []), 1):
        number = int(question.get("number", index))
        aligned.append({"number": number, "student_answer": extracted.get(number, ""), "question": question.get("question", ""),
                        "model_answer": question.get("model_answer", ""), "rubric": question.get("rubric", []),
                        "marks": question.get("marks", 0), "citations": question.get("citations", [])})
    return aligned


def retrieve_grading_evidence(aligned: list[dict], level: str | None = None) -> list[list[dict]]:
    """Retrieve supplementary official/examiner evidence for each question.

    A generated assessment has no exact historical mark scheme by definition.
    Its stored model answer/rubric remains the direct grading authority, while
    this retrieval adds matching official mark-scheme or examiner-report
    evidence whenever the topic/question wording resolves to it. Retrieval is
    skipped without a configured provider so a missing key never becomes a
    fake grade or a failed typed-submission record.
    """
    if not os.getenv("OPENAI_API_KEY"):
        return [[] for _ in aligned]
    evidence: list[list[dict]] = []
    retriever = HybridRetriever()
    try:
        for item in aligned:
            query = f"{item['question']} mark scheme examiner report marking guidance"
            try:
                result = retrieve(query, level=level, maximum_chunks=6, retriever=retriever)
                evidence.append([chunk for chunk in result.get("chunks", [])
                                 if chunk.get("document_type") in {"MARK_SCHEME", "EXAMINER_REPORT", "QUESTION_PAPER"}])
            except Exception:
                # The persisted rubric is sufficient for a teacher-reviewable
                # generated assessment; report the absence in the audit payload.
                evidence.append([])
    finally:
        retriever.close()
    return evidence


def _evaluate(aligned: list[dict], retrieval_evidence: list[list[dict]]) -> dict:
    """Ask the configured model for strict JSON, or return an honest no-call state."""
    if not aligned:
        return {"status": "INVALID_SUBMISSION", "reason": "Assessment has no questions."}
    if not os.getenv("OPENAI_API_KEY"):
        return {"status": "GENERATION_NOT_RUN", "reason": "OPENAI_API_KEY is not configured."}
    from openai import OpenAI
    schema = {"type": "object", "additionalProperties": False, "required": ["items", "total_score", "max_score", "comments", "confidence_score"],
              "properties": {"items": {"type": "array", "minItems": len(aligned), "maxItems": len(aligned),
                                         "items": {"type": "object", "additionalProperties": False,
                                                   "required": ["number", "score", "strengths", "weaknesses", "comments", "confidence_score"],
                                                   "properties": {"number": {"type": "integer"}, "score": {"type": "number", "minimum": 0},
                                                                  "strengths": {"type": "array", "items": {"type": "string"}},
                                                                  "weaknesses": {"type": "array", "items": {"type": "string"}},
                                                                  "comments": {"type": "string"}, "confidence_score": {"type": "number", "minimum": 0, "maximum": 1}}}},
                             "total_score": {"type": "number", "minimum": 0}, "max_score": {"type": "number", "minimum": 0},
                             "comments": {"type": "string"}, "confidence_score": {"type": "number", "minimum": 0, "maximum": 1}}}
    response = OpenAI(api_key=os.environ["OPENAI_API_KEY"]).chat.completions.create(
        model=os.getenv("GRADING_MODEL", os.getenv("GENERATOR_MODEL", "gpt-4.1-mini")), temperature=0,
        max_tokens=int(os.getenv("GRADING_MAX_TOKENS", "1800")),
        response_format={"type": "json_schema", "json_schema": {"name": "assessment_grade", "strict": True, "schema": schema}},
        messages=[{"role": "system", "content": "Grade only against the supplied rubric and model answer. Be concise, fair, and explain deductions. Never award more than the question marks. Return JSON only."},
                  {"role": "user", "content": json.dumps({"questions": aligned, "supplementary_retrieved_evidence": retrieval_evidence}, ensure_ascii=False)}],
    )
    result = json.loads(response.choices[0].message.content or "{}")
    total_marks = sum(float(item["marks"]) for item in aligned)
    valid = all(0 <= float(item["score"]) <= float(next(question["marks"] for question in aligned if int(question["number"]) == int(item["number"]))) for item in result["items"])
    if not valid or not 0 <= float(result["total_score"]) <= total_marks:
        return {"status": "INVALID_GENERATION", "reason": "Generated score lies outside the rubric bounds."}
    result.update({"status": "PENDING_TEACHER_REVIEW", "max_score": total_marks})
    return result


def _graph():
    from langgraph.graph import END, START, StateGraph
    def extract_node(state: GradingState) -> dict:
        questions = state["assessment"]["content"].get("questions", [])
        return {"extracted": extract_answers(state["answer_text"], [int(item.get("number", index + 1)) for index, item in enumerate(questions)])}
    def align_node(state: GradingState) -> dict:
        return {"aligned": _align(state["assessment"], state["extracted"])}
    def retrieve_node(state: GradingState) -> dict:
        return {"retrieval_evidence": retrieve_grading_evidence(state["aligned"], state["assessment"].get("level"))}
    def evaluate_node(state: GradingState) -> dict:
        evaluation = _evaluate(state["aligned"], state["retrieval_evidence"])
        return {"evaluation": evaluation, "result": {"evaluation": evaluation, "aligned_questions": state["aligned"],
                                                         "retrieval_evidence": state["retrieval_evidence"]}}
    graph = StateGraph(GradingState)
    graph.add_node("extract", extract_node); graph.add_node("align", align_node); graph.add_node("retrieve", retrieve_node); graph.add_node("evaluate", evaluate_node)
    graph.add_edge(START, "extract"); graph.add_edge("extract", "align"); graph.add_edge("align", "retrieve"); graph.add_edge("retrieve", "evaluate"); graph.add_edge("evaluate", END)
    return graph.compile()


def grade_typed_submission(answer_text: str, assessment: dict) -> dict:
    """Return a machine-grade draft for teacher review; no handwriting OCR involved."""
    return _graph().invoke({"answer_text": answer_text, "assessment": assessment})["result"]
