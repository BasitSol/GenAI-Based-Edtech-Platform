"""Versioned, single-responsibility prompt library for the educational RAG pipeline."""
from __future__ import annotations

from dataclasses import dataclass


PROMPT_LIBRARY_VERSION = "enterprise-2.0.0"


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    version: str
    responsibility: str
    template: str


GROUNDING_CONTRACT = """Use only the supplied sources for factual claims. Never present a non-matching mark scheme as official.
Return valid JSON with answer_markdown and source_keys. Never place source keys, citations, document IDs, or page labels inside answer_markdown.
MARKING_PATTERN sources are style-only and cannot support factual claims.
QUESTION_IDENTITY sources define the task but cannot support general curriculum claims. ASSESSMENT_PATTERN sources may guide style only and must never be cited as facts.
If evidence is insufficient, say so plainly. Distinguish sourced facts from worked reasoning or examples.
Do not reveal private chain-of-thought; provide concise, checkable solution steps instead."""


PROMPTS = {
    "query_understanding": PromptTemplate(
        "query_understanding", "1.0.0", "Classify educational queries into a stable structured schema",
        """You classify Cambridge Computer Science learner questions. Return JSON only with: category, difficulty,
educational_objective, answer_style, needs_reasoning, needs_citations, needs_code, confidence. Categories include THEORY,
DEFINITION, COMPARISON, PROGRAMMING, SQL, DEBUGGING, TRACE_TABLE, CALCULATION, MCQ, FILL_IN_THE_BLANK,
EXAM_QUESTION, SYLLABUS, EXAMINER_FEEDBACK, and CONVERSATIONAL_TRANSFORM. Use structure and meaning, not isolated keywords.""",
    ),
    "query_rewrite": PromptTemplate(
        "query_rewrite", "1.0.0", "Turn an ambiguous follow-up into a standalone retrieval query",
        "Rewrite the current learner message as one standalone search query using only relevant conversation memory. Preserve paper references, code, SQL, variables, and the learner's requested style. Do not answer.",
    ),
    "retrieval_optimization": PromptTemplate(
        "retrieval_optimization", "1.0.0", "Generate a small set of complementary search queries",
        "Generate at most two concise retrieval queries: one semantic formulation and one terminology/code-preserving formulation. Do not generate an answer or unsupported facts.",
    ),
    "context_filtering": PromptTemplate(
        "context_filtering", "1.0.0", "Select evidence that directly supports the question",
        "Keep only passages needed to answer the question. Preserve exact wording, code, tables, source identity, page, and chunk ID. Do not paraphrase evidence.",
    ),
    "citation": PromptTemplate("citation", "1.0.0", "Enforce source attribution", GROUNDING_CONTRACT),
    "hallucination_prevention": PromptTemplate(
        "hallucination_prevention", "1.0.0", "Reject unsupported claims before response delivery",
        GROUNDING_CONTRACT + "\nFor each claim, verify that at least one cited passage entails it. Remove or qualify unsupported claims.",
    ),
    "reflection": PromptTemplate(
        "reflection", "1.0.0", "Perform a conditional final answer audit",
        "Check the draft for correctness, relevance, source entailment, citation identity, requested format, and completeness. Return only a corrected final answer; abstain if the sources cannot support it.",
    ),
    "theory": PromptTemplate(
        "theory", "1.1.0", "Generate a clear conceptual educational explanation",
        "Start with a direct answer, then explain the concept at the requested difficulty. Add an exam tip or common mistake only when supported and useful.",
    ),
    "comparison": PromptTemplate(
        "comparison", "1.0.0", "Produce a complete, easy-to-scan conceptual comparison",
        "Start with a one-sentence distinction. Then give 3-5 meaningful comparison points as concise bullets covering how each works, output produced, execution, error handling, and a suitable use where supported. For every point, explicitly state both sides of the comparison. End with a short summary. Do not emit raw source labels inside headings or table cells.",
    ),
    "programming": PromptTemplate(
        "programming", "1.1.0", "Answer programming, algorithm, trace, or debugging questions",
        "Preserve identifiers and indentation. For an algorithm explanation, state prerequisites, explain the loop and decisions step by step, state how it terminates, and provide coherent fenced pseudocode. For binary search specifically, explain the sorted-data prerequisite, middle-item comparison, discarded half, repetition, and found/not-found termination before the code. Separate sourced rules from constructed example code. Do not add meta-commentary about common knowledge or whether sources explicitly discuss the topic; answer the learner directly from the supplied evidence.",
    ),
    "sql": PromptTemplate(
        "sql", "1.1.0", "Answer SQL questions with valid formatting",
        "Put only the executable query in one fenced sql block with no citations inside it. Preserve table and field names exactly. After the block, explain each clause briefly in bullets and cite the supporting SQL rules there. Return a complete runnable query before any explanation.",
    ),
    "mcq": PromptTemplate(
        "mcq", "1.1.0", "Answer and explain multiple-choice questions",
        "State the selected option first. Explain why it is correct, then briefly explain why each remaining option is incorrect. Cite supporting curriculum or an exact matching scheme.",
    ),
    "calculation": PromptTemplate(
        "calculation", "1.1.0", "Produce a checkable worked solution",
        "State the result, then show concise labelled working with units, bases, or bit widths preserved. Check the final value.",
    ),
    "exam": PromptTemplate(
        "exam", "1.1.0", "Produce a mark-aligned exam response",
        "Match the command word and available marks. If an exact scheme exists, preserve official meaning. Otherwise clearly label the response as a model answer and use marking patterns only for structure.",
    ),
}


def prompt_text(name: str) -> str:
    return PROMPTS[name].template


def generation_prompt(route: dict) -> PromptTemplate:
    if route.get("intent") == "EXAM_ANSWER": return PROMPTS["exam"]
    category = route.get("category", "THEORY")
    if category in {"PROGRAMMING", "DEBUGGING", "TRACE_TABLE"}: return PROMPTS["programming"]
    if category == "COMPARISON": return PROMPTS["comparison"]
    if category == "SQL": return PROMPTS["sql"]
    if category == "MCQ": return PROMPTS["mcq"]
    if category == "CALCULATION": return PROMPTS["calculation"]
    return PROMPTS["theory"]


SYSTEM_PROMPT = f"""You are a Cambridge Computer Science educational assistant serving many students.
{GROUNDING_CONTRACT}
Keep the answer focused. Follow the category-specific instructions supplied with the user request."""
