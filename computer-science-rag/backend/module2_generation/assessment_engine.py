"""Grounded assessment generation workflow for Phase 2.1.

It deliberately reuses the Phase 1 retrieval graph rather than introducing a
second search stack.  Generated work is always a teacher-reviewable draft;
this module cannot publish an assessment to students.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, TypedDict

from backend.shared.core import current_build_path, read_jsonl, tokens
from backend.shared.prompts import PROMPT_LIBRARY_VERSION, prompt_text
from backend.module1_rag.chat.workflow import retrieve
from backend.module1_rag.monitoring.telemetry import TelemetryStore
from backend.module1_rag.retrieval.hybrid_retriever import HybridRetriever


DIFFICULTY_MAP = {"easy": "BEGINNER", "medium": "INTERMEDIATE", "hard": "ADVANCED",
                  "beginner": "BEGINNER", "intermediate": "INTERMEDIATE", "advanced": "ADVANCED"}
LOGGER = logging.getLogger(__name__)


class AssessmentState(TypedDict, total=False):
    topic: str
    difficulty: str
    assessment_type: str
    question_count: int
    question_format: str
    level: str | None
    selected_topics: list[dict]
    total_marks: int | None
    allows_code: bool
    code_kind: str | None
    retrieval: dict
    blueprint: dict
    content: dict
    distribution: dict
    marking_quality: dict
    mock_test_scope: dict
    validation: dict
    retry_count: int
    result: dict


def _source_catalog(chunks: list[dict], *, maximum_chars: int = 60_000) -> tuple[list[dict], dict[str, dict]]:
    """Make compact prompt-local keys within one global context budget."""
    catalog, mapping = [], {}
    remaining = maximum_chars
    for chunk in chunks:
        text = str(chunk.get("text", ""))
        if remaining <= 0:
            break
        selected_text = text[:min(1800, remaining)]
        index = len(catalog) + 1
        key = f"S{index}"
        mapping[key] = chunk
        catalog.append({"source_key": key, "document_id": chunk.get("document_id"),
                        "chunk_id": chunk.get("chunk_id"), "page": chunk.get("page_start"),
                        "document_type": chunk.get("document_type"), "relationship": chunk.get("relationship"),
                        "source_role": chunk.get("assessment_source_role", "TEXTBOOK_CONTENT"),
                        "topic_ids": chunk.get("assessment_topic_ids", []),
                        "syllabus_evidence_ids": chunk.get("syllabus_evidence_ids", []),
                        "text": selected_text})
        remaining -= len(selected_text)
    return catalog, mapping


def _scoped_query(topic: dict, purpose: str) -> str:
    """Build an auditable retrieval query from a selected syllabus objective."""
    terms = topic.get("retrieval_profile", {}).get("content", [topic["name"]])
    # Long syllabus boilerplate previously dominated embeddings and promoted
    # contents/index pages. Exact curriculum terms make the query concise and
    # preserve the selected chapter/topic boundary.
    return (f"{purpose}. Chapter: {topic.get('chapter_name', '')}. Topic: {topic['name']}. "
            f"Required concepts: {', '.join(terms)}.").strip()


def _is_navigational_chunk(chunk: dict) -> bool:
    """Reject covers, contents pages, indexes, and other non-teaching text."""
    text = str(chunk.get("text", ""))
    opening = re.sub(r"\s+", " ", text[:220]).lower()
    page = int(chunk.get("page_start") or 0)
    navigation_markers = ("contents", "table of contents", "index", "isbn", "acknowledgements",
                          "endorsed for", "back to contents page")
    if any(marker in opening for marker in navigation_markers):
        return True
    # The supplied textbooks contain several front-matter/contents pages.
    # Question papers are short documents, so this rule must be scoped to
    # textbooks rather than applied to every page with a small page number.
    if chunk.get("document_type") == "TEXTBOOK" and 0 < page <= 10:
        return True
    # Cover/front-matter fragments are especially prone to matching the book
    # title and broad chapter labels while containing no assessable material.
    return page <= 5 and len(tokens(text)) < 180


def _topic_chunk_score(topic: dict, chunk: dict, document_type: str) -> float:
    """Score substantive evidence against an explicit selected-topic profile."""
    if chunk.get("document_type") != document_type or _is_navigational_chunk(chunk):
        return -1.0
    # When a teacher selected a printed coursebook section, textbook evidence
    # must come from that exact section's corresponding PDF pages. Semantic
    # similarity is then used to rank within the section, never to escape it.
    if document_type == "TEXTBOOK" and topic.get("source_pdf_page_start") is not None:
        page = int(chunk.get("page_start") or 0)
        if not int(topic["source_pdf_page_start"]) <= page <= int(topic["source_pdf_page_end"]):
            return -1.0
    text = str(chunk.get("text", "")).lower()
    profile_key = "past_paper" if document_type == "QUESTION_PAPER" else "content"
    phrases = [str(item).lower() for item in topic.get("retrieval_profile", {}).get(profile_key, [topic["name"]])]
    exact = sum(min(text.count(phrase), 5) for phrase in phrases if phrase and phrase in text)
    wanted_tokens = {token for phrase in phrases for token in tokens(phrase) if len(token) > 2}
    overlap = len(wanted_tokens & set(tokens(text)))
    # A question-paper task is a usable topic anchor only when it contains an
    # explicit configured phrase. Single-token overlap caused "number
    # systems" to retrieve unrelated pseudocode and database questions that
    # merely contained the word "number". Textbook scope is different: an
    # exact printed-page range is already deterministic curriculum evidence.
    if document_type == "QUESTION_PAPER" and exact == 0:
        return -1.0
    if (document_type != "TEXTBOOK" or topic.get("source_pdf_page_start") is None) and exact == 0 and (
            topic.get("retrieval_profile", {}).get("strict_phrases") or overlap < 1):
        return -1.0
    opening = re.sub(r"\s+", " ", text[:240])
    heading_bonus = 5.0 if any(phrase in opening for phrase in phrases if len(phrase) > 3) else 0.0
    substantive_bonus = 2.0 if chunk.get("content_type") not in {"PARENT_CONTEXT"} else 0.5
    question_bonus = 2.0 if document_type == "QUESTION_PAPER" and any(
        command in text for command in ("describe", "explain", "identify", "write", "state", "complete")) else 0.0
    return exact * 8.0 + overlap + heading_bonus + substantive_bonus + question_bonus


@lru_cache(maxsize=2)
def _build_chunks(build_path: str) -> tuple[dict, ...]:
    """Load one immutable build once for deterministic assessment retrieval."""
    return tuple(read_jsonl(Path(build_path) / "chunks" / "all.jsonl"))


def _active_chunks() -> tuple[dict, ...]:
    return _build_chunks(str(current_build_path()))


def _diverse_candidates(scored: list[tuple[float, dict]], limit: int) -> list[dict]:
    """Prefer page/document diversity before filling from overlapping chunks."""
    scored.sort(key=lambda item: (-item[0], str(item[1].get("chunk_id", ""))))
    selected: list[dict] = []
    seen_locations: set[tuple[str, int]] = set()
    for _, row in scored:
        location = (str(row.get("document_id", "")), int(row.get("page_start") or 0))
        if location in seen_locations:
            continue
        selected.append(row)
        seen_locations.add(location)
        if len(selected) == limit:
            return selected
    selected_ids = {str(item.get("chunk_id")) for item in selected}
    selected.extend(row for _, row in scored if str(row.get("chunk_id")) not in selected_ids)
    return selected[:limit]


def _local_topic_candidates(topic: dict, level: str | None, document_type: str, limit: int = 8) -> list[dict]:
    """Run a deterministic lexical safety net over the immutable active build."""
    scored = [(score, row) for row in _active_chunks() if (not level or row.get("level") == level)
              and (score := _topic_chunk_score(topic, row, document_type)) >= 0]
    return _diverse_candidates(scored, limit)


def _local_past_paper_style_candidates(level: str | None, limit: int = 6) -> list[dict]:
    """Select diverse Cambridge task structures without treating them as facts."""
    command_words = ("state", "identify", "describe", "explain", "complete", "write", "calculate", "draw", "suggest")
    scored: list[tuple[float, dict]] = []
    for row in _active_chunks():
        if row.get("document_type") != "QUESTION_PAPER" or (level and row.get("level") != level):
            continue
        if _is_navigational_chunk(row):
            continue
        text = str(row.get("text", "")).lower()
        commands = sum(bool(re.search(rf"\b{command}\b", text)) for command in command_words)
        if commands == 0 or len(tokens(text)) < 12:
            continue
        marks = int(row.get("marks") or 0)
        score = commands * 4.0 + min(marks, 6) + min(len(tokens(text)), 180) / 180
        scored.append((score, row))
    return _diverse_candidates(scored, limit)


def _paper_style_projection(chunk: dict) -> dict:
    """Strip source subject matter while retaining auditable task-shape cues."""
    text = str(chunk.get("text", "")).lower()
    commands = [
        command for command in
        ("state", "identify", "describe", "explain", "complete", "write", "calculate", "draw", "suggest")
        if re.search(rf"\b{command}\b", text)
    ]
    marks = int(chunk.get("marks") or 0)
    style_text = (
        "Cambridge question-style pattern. "
        f"Command words: {', '.join(commands) or 'structured response'}. "
        f"Displayed marks: {marks if marks else 'not parsed'}. "
        "Use only its command-word and response-shape pattern; its original subject matter is intentionally omitted."
    )
    return {**chunk, "text": style_text, "retrieval_text": style_text,
            "style_projection": True}


def _local_marking_patterns(level: str | None, limit: int = 4) -> list[dict]:
    """Select bounded, diverse rubric examples without another model/index call."""
    scored: list[tuple[float, dict]] = []
    for row in _active_chunks():
        if row.get("document_type") not in {"MARK_SCHEME", "EXAMINER_REPORT"}:
            continue
        if level and row.get("level") != level:
            continue
        text = str(row.get("text", ""))
        if len(tokens(text)) < 8:
            continue
        score = 3.0 if row.get("document_type") == "MARK_SCHEME" else 2.0
        score += min(len(tokens(text)), 180) / 180
        scored.append((score, row))
    return _diverse_candidates(scored, limit)


def _rank_scoped_candidates(topic: dict, retrieved: list[dict], level: str | None,
                            document_type: str, limit: int = 6) -> list[dict]:
    """Merge hybrid retrieval with local scope candidates and remove drift."""
    candidates = _local_topic_candidates(topic, level, document_type, limit=limit * 2) + retrieved
    unique: dict[str, tuple[float, dict]] = {}
    for chunk in candidates:
        score = _topic_chunk_score(topic, chunk, document_type)
        chunk_id = str(chunk.get("chunk_id", ""))
        if score >= 0 and (chunk_id not in unique or score > unique[chunk_id][0]):
            unique[chunk_id] = (score, chunk)
    ranked = sorted(unique.values(), key=lambda item: (-item[0], str(item[1].get("chunk_id", ""))))
    return [chunk for _, chunk in ranked[:limit]]


def _tag_topic_chunk(chunk: dict, role: str, topic: dict) -> dict:
    """Attach immutable scope provenance to a retrieved evidence chunk."""
    return {**chunk, "assessment_source_role": role,
            "assessment_topic_ids": [topic["id"]],
            "syllabus_evidence_ids": [item["chunk_id"] for item in topic.get("syllabus_evidence", [])]}


def _merge_scoped_chunk(combined: list[dict], seen: dict[str, int], chunk: dict) -> None:
    """Deduplicate shared chunks while retaining every topic that selected it."""
    chunk_id = str(chunk.get("chunk_id"))
    if chunk_id not in seen:
        seen[chunk_id] = len(combined)
        combined.append(chunk)
        return
    existing = combined[seen[chunk_id]]
    existing["assessment_topic_ids"] = sorted(set(existing.get("assessment_topic_ids", [])) |
                                              set(chunk.get("assessment_topic_ids", [])))
    existing["syllabus_evidence_ids"] = sorted(set(existing.get("syllabus_evidence_ids", [])) |
                                             set(chunk.get("syllabus_evidence_ids", [])))


def retrieve_assessment_evidence(topic: str, assessment_type: str, level: str | None,
                                 selected_topics: list[dict] | None = None) -> dict:
    """Build separate factual and quality-evidence lanes for an assessment.

    Topic-scoped textbook passages are the factual authority for every item.
    Question papers contribute Cambridge command words and task shape only;
    their subject matter is projected out before prompting. Mark schemes and
    examiner reports likewise inform rubric length, allocation, and quality,
    not curriculum facts. Keeping these roles explicit makes sparse paper
    coverage safe instead of rejecting valid syllabus topics or borrowing
    unrelated content.
    """
    selected_topics = selected_topics or []
    # Mock tests have exact coursebook page ranges. Using those immutable
    # ranges directly is both more relevant and much cheaper than constructing
    # 2N+1 Chroma/BM25/cross-encoder stacks for N selected topics. Past papers
    # remain style/task anchors; they are never required as factual evidence
    # for a topic the supplied paper sample does not happen to contain.
    scoped = assessment_type == "mock_test" and bool(selected_topics)
    scoped_results: list[dict[str, Any]] = []
    retrieval_debug: dict[str, Any] = {"scoped": scoped, "topics": []}
    if scoped:
        per_topic_limit = max(2, min(6, 18 // max(1, len(selected_topics))))
        for selected in selected_topics:
            textbook_chunks = _local_topic_candidates(selected, level, "TEXTBOOK", limit=per_topic_limit)
            topical_paper_chunks = _local_topic_candidates(selected, level, "QUESTION_PAPER", limit=2)
            scoped_results.append({
                "topic": selected,
                "textbook": {"chunks": textbook_chunks},
                "past_paper": {"chunks": topical_paper_chunks},
            })
            retrieval_debug["topics"].append({
                "topic_id": selected["id"],
                "strategy": "exact_coursebook_pages_plus_strict_local_paper_match",
                "textbook_candidates": len(textbook_chunks),
                "topical_paper_candidates": len(topical_paper_chunks),
            })
        marking_chunks = _local_marking_patterns(level, limit=4)
    else:
        # Quizzes do not have selector-backed page ranges. Reuse one request-
        # scoped retriever for all lanes and close it deterministically.
        retriever = HybridRetriever()
        try:
            textbook = retrieve(topic, level=level, maximum_chunks=8, document_type="TEXTBOOK",
                                retriever=retriever)
            past_paper = retrieve(f"Cambridge Computer Science past paper question {topic}", level=level,
                                  maximum_chunks=10, document_type="QUESTION_PAPER", retriever=retriever)
            marking = retrieve(
                f"Generate a Cambridge Computer Science {assessment_type} using mark scheme marking criteria and answer quality patterns for {topic}",
                level=level, maximum_chunks=8, retriever=retriever,
            )
        finally:
            retriever.close()
        scoped_results.append({"topic": None, "textbook": textbook, "past_paper": past_paper})
        marking_chunks = marking.get("chunks", [])
        retrieval_debug.update({
            "topics": [{"topic_id": None,
                        "textbook": textbook.get("retrieval_debug", {}),
                        "past_paper": past_paper.get("retrieval_debug", {})}],
            "marking": marking.get("retrieval_debug", {}),
        })
    combined, seen = [], {}
    for result in scoped_results:
        selected = result["topic"]
        for chunk in result["textbook"].get("chunks", []):
            if chunk.get("document_type") != "TEXTBOOK":
                continue
            tagged = _tag_topic_chunk(chunk, "TEXTBOOK_CONTENT", selected) if selected else {**chunk, "assessment_source_role": "TEXTBOOK_CONTENT"}
            _merge_scoped_chunk(combined, seen, tagged)
        for chunk in result["past_paper"].get("chunks", []):
            if chunk.get("document_type") != "QUESTION_PAPER":
                continue
            projected = _paper_style_projection(chunk)
            tagged = _tag_topic_chunk(projected, "PAST_PAPER_STYLE", selected) if selected else {
                **projected, "assessment_source_role": "PAST_PAPER_STYLE",
            }
            _merge_scoped_chunk(combined, seen, tagged)
    if scoped:
        # Ensure a balanced adapted-question plan even when the small supplied
        # paper sample has no topical task. These anchors contribute command
        # words and structure only; every factual claim still cites the
        # question's topic-scoped textbook evidence.
        for chunk in _local_past_paper_style_candidates(level, limit=6):
            projected = _paper_style_projection(chunk)
            _merge_scoped_chunk(combined, seen, {
                **projected, "assessment_source_role": "PAST_PAPER_STYLE",
                "assessment_topic_ids": [],
                "syllabus_evidence_ids": [],
            })
    for chunk in marking_chunks:
        if chunk.get("document_type") not in {"MARK_SCHEME", "EXAMINER_REPORT", "SYLLABUS"}:
            continue
        role = "CURRICULUM_SCOPE" if chunk.get("document_type") == "SYLLABUS" else "MARKING_PATTERN"
        _merge_scoped_chunk(combined, seen, {**chunk, "assessment_source_role": role})
    missing_topic_ids = [item["id"] for item in selected_topics
                         if not any(item["id"] in chunk.get("assessment_topic_ids", [])
                                    and chunk.get("assessment_source_role") == "TEXTBOOK_CONTENT" for chunk in combined)]
    missing_past_paper_topic_ids = [item["id"] for item in selected_topics
                                    if not any(item["id"] in chunk.get("assessment_topic_ids", [])
                                               and chunk.get("assessment_source_role") == "PAST_PAPER_STYLE" for chunk in combined)]
    retrieval_debug.update({
        "build_id": current_build_path().name,
        "strategy": "bounded_local_scope" if scoped else "shared_hybrid_retriever",
        "context_chunk_count": len(combined),
        "topics_without_topical_past_paper": missing_past_paper_topic_ids,
    })
    return {"chunks": combined,
            "textbook_chunk_count": sum(item["assessment_source_role"] == "TEXTBOOK_CONTENT" for item in combined),
            "past_paper_chunk_count": sum(item["assessment_source_role"] == "PAST_PAPER_STYLE" for item in combined),
            "past_paper_style_chunk_count": sum(item["assessment_source_role"] == "PAST_PAPER_STYLE" for item in combined),
            "marking_pattern_chunk_count": sum(item["assessment_source_role"] == "MARKING_PATTERN" for item in combined),
            "missing_selected_topic_ids": missing_topic_ids,
            # Informational only: a sparse past-paper sample cannot define
            # whether a valid syllabus topic is generatable.
            "missing_past_paper_topic_ids": missing_past_paper_topic_ids,
            "retrieval_debug": retrieval_debug}


def _plan(topic: str, difficulty: str, assessment_type: str, question_count: int, question_format: str = "mixed",
          selected_topics: list[dict] | None = None, total_marks: int | None = None,
          allows_code: bool = True, code_kind: str | None = None) -> dict:
    """Deterministically map UI requests onto the existing Phase 1 difficulty taxonomy."""
    normalised = DIFFICULTY_MAP.get(difficulty.lower(), difficulty.upper())
    count = max(1, min(question_count, 20))
    selected_topics = selected_topics or []
    return {"topic": topic.strip(), "difficulty": normalised, "assessment_type": assessment_type,
            "question_count": count, "question_format": question_format.upper(),
            "content_distribution": {"textbook_questions": (count + 1) // 2,
                                     "past_paper_questions": count // 2,
                                     "rule": "balanced; the difference may be at most one for an odd question count"},
            "selected_topics": selected_topics,
            "selected_topic_ids": [item["id"] for item in selected_topics],
            "total_marks": total_marks,
            "allows_code": allows_code,
            "code_kind": code_kind or ("PROGRAMMING" if assessment_type == "mock_test" and allows_code else None),
            "mock_test_structure": ({"minimum_questions": 7, "target_question_count": count,
                                     "mark_pattern": "Use at least three mark values across 1–6 marks. Include at least one one-mark MCQ, short/structured 2–4 mark questions, and an applied coding/programming item only when code is allowed."}
                                    if assessment_type == "mock_test" else None),
            "prompt_version": PROMPT_LIBRARY_VERSION,
            "planner_provider": "deterministic_fallback"}


def _retryable_provider_error(exc: Exception) -> bool:
    """Classify transient provider failures without exposing response bodies."""
    retryable_names = {
        "APIConnectionError", "APITimeoutError", "RateLimitError",
        "InternalServerError", "ServiceUnavailableError",
    }
    status_code = getattr(exc, "status_code", None)
    return type(exc).__name__ in retryable_names or status_code == 429 or (
        isinstance(status_code, int) and 500 <= status_code < 600
    )


def _bounded_env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    """Read a bounded integer without allowing bad deployment config to crash."""
    try:
        return max(minimum, min(int(os.getenv(name, str(default))), maximum))
    except (TypeError, ValueError):
        LOGGER.warning("Invalid integer configuration %s; using %s", name, default)
        return default


def _bounded_env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    """Read a bounded float without allowing bad deployment config to crash."""
    try:
        return max(minimum, min(float(os.getenv(name, str(default))), maximum))
    except (TypeError, ValueError):
        LOGGER.warning("Invalid numeric configuration %s; using %s", name, default)
        return default


def _openai_create_with_retry(client: Any, *, operation: str, **kwargs: Any) -> tuple[Any, int]:
    """Run a bounded provider call with deterministic exponential backoff."""
    maximum_attempts = _bounded_env_int("ASSESSMENT_PROVIDER_MAX_ATTEMPTS", 3, 1, 4)
    base_delay = _bounded_env_float("ASSESSMENT_RETRY_BASE_SECONDS", 0.75, 0.0, 5.0)
    for attempt in range(1, maximum_attempts + 1):
        try:
            return client.chat.completions.create(**kwargs), attempt
        except Exception as exc:
            if attempt >= maximum_attempts or not _retryable_provider_error(exc):
                raise
            delay = base_delay * (2 ** (attempt - 1))
            LOGGER.warning("%s transient provider failure; retrying attempt %s/%s in %.2fs",
                           operation, attempt + 1, maximum_attempts, delay)
            if delay:
                time.sleep(delay)
    raise RuntimeError(f"{operation} exhausted provider attempts")


def _openai_json_with_retry(client: Any, *, operation: str, **kwargs: Any) -> tuple[Any, dict, int]:
    """Retry empty, truncated, or invalid structured bodies at the call site.

    A provider request can succeed at the HTTP layer yet return an empty body
    or a response that cannot be parsed. Transport retries cannot see that
    failure. This helper retries only the affected structured call, preserving
    an already generated paper when the semantic judge is the failing stage.
    """
    structured_attempts = _bounded_env_int(
        "ASSESSMENT_STRUCTURED_RESPONSE_ATTEMPTS", 2, 1, 3
    )
    total_provider_attempts = 0
    last_error: Exception | None = None
    for structured_attempt in range(1, structured_attempts + 1):
        response, provider_attempts = _openai_create_with_retry(
            client,
            operation=operation,
            **kwargs,
        )
        total_provider_attempts += provider_attempts
        finish_reason = getattr(response.choices[0], "finish_reason", "stop")
        if finish_reason not in {"stop", None}:
            last_error = RuntimeError("INCOMPLETE_MODEL_RESPONSE")
        else:
            try:
                payload = json.loads(response.choices[0].message.content or "")
                if not isinstance(payload, dict):
                    raise ValueError("Structured response must be a JSON object.")
                return response, payload, total_provider_attempts
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                last_error = exc
        if structured_attempt < structured_attempts:
            LOGGER.warning(
                "%s returned an unusable structured body; retrying response %s/%s",
                operation,
                structured_attempt + 1,
                structured_attempts,
            )
    assert last_error is not None
    raise last_error


def _provider_usage(response: Any) -> dict[str, int | float]:
    """Normalize OpenAI usage fields and calculate configured request cost."""
    usage = getattr(response, "usage", None)
    input_tokens = int(
        getattr(usage, "prompt_tokens", None)
        or getattr(usage, "input_tokens", None)
        or 0
    )
    output_tokens = int(
        getattr(usage, "completion_tokens", None)
        or getattr(usage, "output_tokens", None)
        or 0
    )
    input_price = _bounded_env_float(
        "GENERATOR_INPUT_USD_PER_MILLION", 0.40, 0.0, 10_000.0
    )
    output_price = _bounded_env_float(
        "GENERATOR_OUTPUT_USD_PER_MILLION", 1.60, 0.0, 10_000.0
    )
    total_cost = (
        input_tokens * input_price + output_tokens * output_price
    ) / 1_000_000
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_cost": round(total_cost, 8),
    }


def _plan_with_llm(fallback: dict) -> dict:
    """Optionally enrich the request into a structured blueprint before retrieval.

    The deterministic request remains authoritative for type/difficulty/count;
    model output can add only a learning objective and question mix. This keeps
    the planner useful without letting it silently change the teacher request.
    """
    # The deterministic blueprint already owns all authoritative structure.
    # A second model call added latency and a new failure point without
    # changing retrieval or validation. It is therefore opt-in for ordinary
    # quizzes and always disabled for strict mock tests.
    enabled = os.getenv("ASSESSMENT_LLM_PLANNER_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
    if fallback.get("assessment_type") == "mock_test" or not enabled or not os.getenv("OPENAI_API_KEY"):
        return fallback
    from openai import OpenAI
    schema = {"type": "object", "additionalProperties": False,
              "required": ["learning_objective", "question_mix"],
              "properties": {"learning_objective": {"type": "string"}, "question_mix": {"type": "string"}}}
    try:
        _, enriched, _ = _openai_json_with_retry(
            OpenAI(api_key=os.environ["OPENAI_API_KEY"]),
            operation="assessment_planning",
            model=os.getenv("GENERATOR_MODEL", "gpt-4.1-mini"), temperature=0, max_tokens=160,
            response_format={"type": "json_schema", "json_schema": {"name": "assessment_blueprint", "strict": True, "schema": schema}},
            messages=[{"role": "system", "content": "Create a concise Cambridge Computer Science assessment blueprint. Preserve the requested topic, difficulty, type, question count, and every supplied deterministic mock-test mark/type constraint. Do not invent an alternative mark allocation."},
                      {"role": "user", "content": json.dumps(_model_blueprint(fallback))}],
        )
        return {**fallback, **enriched, "planner_provider": "openai_structured"}
    except Exception as exc:
        return {**fallback, "planner_error": type(exc).__name__}


def _model_blueprint(plan: dict) -> dict:
    """Return only authoritative, generation-relevant blueprint fields.

    Raw syllabus excerpts are retained in the internal audit blueprint, but
    they are intentionally excluded from model input. A syllabus chunk may
    span two printed sections (for example Boolean logic followed by
    Databases); sending that parser artefact to the planner previously caused
    otherwise valid topic requests to drift.
    """
    compact = dict(plan)
    compact["selected_topics"] = [
        {key: topic.get(key) for key in (
            "id", "name", "chapter_id", "chapter_name", "allows_code",
            "code_kind", "section_number", "book_page_start", "book_page_end",
            "retrieval_profile"
        ) if topic.get(key) is not None}
        for topic in plan.get("selected_topics", [])
    ]
    return compact


def _question_source_assignments(plan: dict, catalog: list[dict]) -> list[dict]:
    """Create a deterministic source-origin plan before the model writes.

    A natural-language request for a 50/50 split is not sufficient proof of
    provenance. Each question number is therefore allocated a factual lane in
    advance. Past-paper-adapted questions receive both a question-paper anchor
    and textbook answer support.
    """
    textbook_keys = [item["source_key"] for item in catalog if item["source_role"] == "TEXTBOOK_CONTENT"]
    past_paper_keys = [item["source_key"] for item in catalog if item["source_role"] == "PAST_PAPER_STYLE"]
    if not textbook_keys or not past_paper_keys:
        return []
    selected_topic_ids = list(plan.get("selected_topic_ids", []))

    def keys_for(role: str, topic_id: str | None) -> list[str]:
        scoped = [item["source_key"] for item in catalog if item["source_role"] == role
                  and (not topic_id or topic_id in item.get("topic_ids", []))]
        if role == "PAST_PAPER_STYLE" and not scoped:
            # A paper may be a pure task/command-word anchor for a topic not
            # represented in the finite source sample. It never supplies the
            # question's facts or model answer.
            return [item["source_key"] for item in catalog if item["source_role"] == role]
        # Ordinary quizzes predate syllabus selection, so their evidence has
        # no topic label. Mock tests never take this fallback: a question
        # needs evidence retrieved specifically for its selected topic.
        return scoped if scoped or topic_id else [item["source_key"] for item in catalog if item["source_role"] == role]

    targets = plan["content_distribution"]
    textbook_remaining, paper_remaining = targets["textbook_questions"], targets["past_paper_questions"]
    assignments = []
    for number in range(1, plan["question_count"] + 1):
        # Alternate lanes where possible; this prevents all paper-derived
        # questions being clustered at the end of a test.
        use_past_paper = paper_remaining and (number % 2 == 0 or not textbook_remaining)
        if selected_topic_ids:
            # Vary source lane and topic independently. With two selected
            # topics the first four assignments are T(A), P(B), T(B), P(A),
            # so neither topic is accidentally represented by only one
            # source family.
            pair_index = (number - 1) // 2
            topic_index = (pair_index + (1 if use_past_paper else 0)) % len(selected_topic_ids)
            topic_id = selected_topic_ids[topic_index]
        else:
            topic_id = None
        topic_textbook_keys = keys_for("TEXTBOOK_CONTENT", topic_id)
        topic_paper_keys = keys_for("PAST_PAPER_STYLE", topic_id)
        if not topic_textbook_keys or (use_past_paper and not topic_paper_keys):
            return []
        if use_past_paper:
            assignments.append({"number": number, "content_source": "PAST_PAPER",
                                "topic_ids": [topic_id] if topic_id else [],
                                "required_source_keys": [
                                    topic_paper_keys[(number - 1) % len(topic_paper_keys)],
                                    topic_textbook_keys[(number - 1) % len(topic_textbook_keys)],
                                ]})
            paper_remaining -= 1
        else:
            assignments.append({"number": number, "content_source": "TEXTBOOK",
                                "topic_ids": [topic_id] if topic_id else [],
                                "required_source_keys": [topic_textbook_keys[(number - 1) % len(topic_textbook_keys)]]})
            textbook_remaining -= 1
    return assignments


def _question_structure_assignments(plan: dict) -> list[dict]:
    """Preallocate the 25-mark mock-test paper structure deterministically."""
    if plan.get("assessment_type") != "mock_test":
        return []
    # This is exactly 25 marks across eight questions. A one-mark MCQ has one
    # decisive marking point; the remaining questions use short/structured
    # forms that can legitimately earn partial credit.
    marks = [1, 2, 3, 3, 4, 4, 4, 4]
    types = ["MCQ", "SHORT_ANSWER", "SHORT_ANSWER", "SHORT_ANSWER",
             "LONG_ANSWER", "LONG_ANSWER", "LONG_ANSWER", "LONG_ANSWER"]
    return [{"number": number, "marks": mark, "question_type": question_type,
             "requires_coding": bool(plan.get("allows_code") and number == 8),
             "coding_kind": plan.get("code_kind") if plan.get("allows_code") and number == 8 else None}
            for number, (mark, question_type) in enumerate(zip(marks, types), 1)]


def _normalise_programming_rubric(question: dict, structure: dict | None) -> None:
    """Guarantee an independently markable rubric for the fixed coding item.

    Structured-output models occasionally return a sound programming question
    with a short, generic rubric.  The mock-test contract already fixes this
    item at four marks, so its marking dimensions can be made deterministic
    without changing the generated task, model answer, source citations, or
    teacher-review status.  Existing criteria are retained and only missing
    assessment dimensions are added.
    """
    if not structure or not structure.get("requires_coding"):
        return
    rubric = [str(item).strip() for item in question.get("rubric", []) if str(item).strip()]
    if structure.get("coding_kind") == "SQL":
        required = (
            ({"select", "field", "column"}, "1 mark for selecting the correct field(s)."),
            ({"from", "table"}, "1 mark for using the correct table in the FROM clause."),
            ({"where", "condition", "criteria", "filter"}, "1 mark for the correct WHERE condition or filtering criterion."),
            ({"order", "aggregate", "count", "sum", "output", "result"},
             "1 mark for the correct ordering, aggregate, or resulting output required by the task."),
        )
        normalisation_version = "sql-four-dimensions-v1"
    else:
        required = (
            ({"logic", "algorithm"}, "1 mark for correct algorithm logic appropriate to the stated task."),
            ({"condition", "comparison", "selection"}, "1 mark for a correct condition or comparison."),
            ({"loop", "iteration", "midpoint", "bounds", "index", "update", "search", "sort", "progression"},
             "1 mark for correct loop/iteration or progression through the data."),
            ({"output", "termination", "stop", "not found", "result"},
             "1 mark for correct output and termination behaviour."),
        )
        normalisation_version = "programming-four-dimensions-v1"
    selected: list[str] = []
    used: set[int] = set()
    for terms, fallback in required:
        matched_index = next((index for index, criterion in enumerate(rubric)
                              if index not in used and any(term in criterion.lower() for term in terms)), None)
        if matched_index is None:
            selected.append(fallback)
        else:
            used.add(matched_index)
            selected.append(rubric[matched_index])
    # The fixed coding item is four marks, so these four distinct dimensions
    # form a complete and non-overlapping marking contract.
    question["rubric"] = selected
    question["rubric_normalisation"] = normalisation_version


def _rubric_fragments(value: str) -> list[str]:
    """Split a compressed model rubric/answer into candidate atomic points."""
    cleaned = re.sub(r"\s+", " ", value or "").strip()
    if not cleaned:
        return []
    # Models commonly compress multiple marking points with semicolons,
    # numbered clauses, or repeated '1 mark' phrases.
    parts = re.split(r"\s*(?:;|\n|(?=\b(?:award\s+)?1\s+mark\b)|(?<=\.)\s+)\s*", cleaned,
                     flags=re.IGNORECASE)
    return [part.strip(" -.,") for part in parts if len(part.strip(" -.,")) >= 8]


def _one_mark_criterion(value: str) -> str:
    """Represent one candidate point with an explicit one-mark allocation."""
    cleaned = re.sub(r"^(?:award\s+)?\d+\s+marks?\s*(?:for|:)?\s*", "", value.strip(),
                     flags=re.IGNORECASE).strip(" .")
    return f"1 mark for {cleaned}." if cleaned else ""


def _normalise_atomic_rubric(question: dict, structure: dict | None) -> None:
    """Compile exactly one independently assessable rubric point per mark.

    This is a deterministic post-generation contract, not another generative
    judgment. It first preserves atomic model-supplied criteria, then derives
    missing points from the already generated model answer. Generic fallbacks
    are used only when the model answer itself has too few separable clauses.
    """
    marks = int((structure or {}).get("marks") or question.get("marks", 0))
    if marks < 1:
        return
    if question.get("question_type") == "MCQ":
        answer = question.get("correct_option") or question.get("model_answer") or "the correct option"
        question["rubric"] = [f"1 mark for selecting {str(answer).strip().rstrip('.') }."]
        question["rubric_normalisation"] = "atomic-one-point-v1"
        return

    candidates: list[str] = []
    for entry in question.get("rubric", []):
        candidates.extend(_rubric_fragments(str(entry)))
    answer_candidates = _rubric_fragments(str(question.get("model_answer", "")))
    fallbacks = (
        "an accurate core statement that answers the command word",
        "a relevant supporting explanation linked to the stated problem",
        "a correct application or intermediate step",
        "a complete result, consequence, or conclusion",
        "an additional relevant detail consistent with the model answer",
        "a second valid supporting detail consistent with the model answer",
    )
    normalised: list[str] = []
    seen: set[str] = set()
    for raw in candidates + [f"stating this model-answer point: {item}" for item in answer_candidates] + list(fallbacks):
        criterion = _one_mark_criterion(raw)
        identity = re.sub(r"\W+", "", criterion.lower())
        if not criterion or identity in seen:
            continue
        seen.add(identity)
        normalised.append(criterion)
        if len(normalised) == marks:
            break
    question["rubric"] = normalised
    question["rubric_normalisation"] = "atomic-per-mark-v1"


def _question_matches_assigned_topic(question: dict, plan: dict, evidence_text: str = "") -> bool:
    """Verify that a question has a valid, evidence-backed topic assignment.

    When generation is source-backed, semantic topic alignment must not be
    decided by a word-overlap threshold. Valid questions can paraphrase the
    source completely (for example "unique identifier" versus "primary key").
    The mandatory source assignment proves the selected topic provenance here;
    the later semantic grounding judge evaluates whether the actual question
    and answer are entailed by that evidence. Lexical matching remains only as
    a defensive fallback for direct/unit callers that supply no evidence.
    """
    assigned_ids = question.get("topic_ids", [])
    if not assigned_ids:
        return True
    topics = {item["id"]: item for item in plan.get("selected_topics", [])}
    if any(topic_id not in topics for topic_id in assigned_ids):
        return False
    if evidence_text.strip():
        return True
    text = f"{question.get('question', '')} {question.get('model_answer', '')}".lower()
    ignored = {"explain", "describe", "state", "identify", "write", "give", "answer", "question",
               "computer", "program", "data", "value", "values", "correct", "using", "used", "each",
               "this", "that", "with", "from", "into", "result", "example"}
    text_terms = {term for term in tokens(text) if len(term) > 2 and term not in ignored}
    for topic_id in assigned_ids:
        topic = topics.get(topic_id)
        if not topic:
            return False
        phrases = topic.get("retrieval_profile", {}).get("content", [topic.get("name", "")])
        profile_terms = {term for phrase in phrases for term in tokens(str(phrase))
                         if len(term) > 2 and term not in ignored}
        exact_match = any(str(phrase).lower() in text for phrase in phrases if len(str(phrase)) > 2)
        profile_overlap = len(profile_terms & text_terms)
        if not exact_match and profile_overlap < 1:
            return False
    return True


def _unsupported_advanced_claim(question: dict, source_text: str) -> str | None:
    """Catch advanced terminology introduced without textbook evidence."""
    answer = f"{question.get('question', '')} {question.get('model_answer', '')}".lower()
    source = source_text.lower()
    guarded = {
        "BIG_O_NOT_IN_SOURCE": ("time complexity", "o(n", "quadratic complexity"),
    }
    for reason, phrases in guarded.items():
        if any(phrase in answer for phrase in phrases) and not any(phrase in source for phrase in phrases):
            return reason
    return None


def _apply_system_owned_question_contract(question: dict, assignment: dict | None,
                                          structure: dict | None,
                                          mapping: dict[str, dict]) -> dict:
    """Attach canonical provenance and mock-test structure to model content.

    Internal evidence identifiers must never depend on an LLM copying them.
    This function is deliberately deterministic and independently testable.
    """
    if not assignment:
        return {"passed": False, "reason": "Generated an unknown or duplicate question number."}
    question["content_source"] = assignment["content_source"]
    question["topic_ids"] = list(assignment["topic_ids"])
    keys = list(dict.fromkeys(assignment["required_source_keys"]))
    if any(key not in mapping for key in keys):
        return {"passed": False,
                "reason": "The deterministic source assignment references unknown evidence."}
    if structure:
        question["question_type"] = structure["question_type"]
        question["marks"] = structure["marks"]
        # This flag is application-owned. It records the deterministic paper
        # structure so later validators never need to infer the intended
        # response type from incidental words in generated prose.
        question["requires_coding"] = bool(structure.get("requires_coding"))
        if structure["question_type"] != "MCQ":
            question["options"] = []
            question["correct_option"] = ""
        elif (len(question.get("options", [])) != 4
              or question.get("correct_option") not in question.get("options", [])):
            return {"passed": False,
                    "reason": "The generated one-mark MCQ does not contain four options and one valid answer.",
                    "question": question.get("number")}
    return {"passed": True, "keys": keys}


def _generate(plan: dict, retrieval_result: dict) -> dict:
    """Call the configured generator only after sufficient curriculum evidence exists."""
    chunks = retrieval_result.get("chunks", [])
    textbook_chunks = [item for item in chunks if item.get("assessment_source_role") == "TEXTBOOK_CONTENT"]
    past_paper_chunks = [item for item in chunks if item.get("assessment_source_role") == "PAST_PAPER_STYLE"]
    if not textbook_chunks or not past_paper_chunks:
        missing = "textbook" if not textbook_chunks else "past-paper"
        return {"status": "INSUFFICIENT_EVIDENCE", "reason": f"No relevant {missing} evidence was retrieved for '{plan['topic']}'. A balanced assessment cannot be generated safely."}
    if retrieval_result.get("missing_selected_topic_ids"):
        return {"status": "INSUFFICIENT_EVIDENCE", "reason": "No textbook evidence was retrieved for every selected mock-test topic: " + ", ".join(retrieval_result["missing_selected_topic_ids"])}
    if not os.getenv("OPENAI_API_KEY"):
        return {"status": "GENERATION_NOT_RUN", "reason": "OPENAI_API_KEY is not configured."}
    from openai import OpenAI

    catalog, mapping = _source_catalog(chunks)
    schema = {
        "type": "object", "additionalProperties": False,
        "required": ["title", "instructions", "questions"],
        "properties": {
            "title": {"type": "string"}, "instructions": {"type": "string"},
            "questions": {"type": "array", "minItems": plan["question_count"], "maxItems": plan["question_count"],
                          "items": {"type": "object", "additionalProperties": False,
                                    "required": ["number", "question", "question_type", "options", "correct_option",
                                                 "marks", "model_answer", "rubric"],
                                    "properties": {"number": {"type": "integer", "minimum": 1},
                                                   "question": {"type": "string"},
                                                   "question_type": {"type": "string", "enum": ["MCQ", "SHORT_ANSWER", "LONG_ANSWER"]},
                                                   "options": {"type": "array", "items": {"type": "string"}, "maxItems": 4},
                                                   "correct_option": {"type": "string"},
                                                   "marks": {"type": "integer", "minimum": 1},
                                                   "model_answer": {"type": "string"},
                                                   "rubric": {"type": "array", "items": {"type": "string"}}}}},
        },
    }
    assignments = _question_source_assignments(plan, catalog)
    if not assignments:
        return {"status": "INSUFFICIENT_EVIDENCE", "reason": "The balanced source plan could not allocate both textbook and past-paper evidence."}
    structure_assignments = _question_structure_assignments(plan)
    request = {"blueprint": _model_blueprint(plan), "instructions": prompt_text("assessment_generation"), "sources": catalog,
               "question_source_assignments": assignments,
               "question_structure_assignments": structure_assignments,
               "output_rule": "Return exactly one question for every assignment number. Ground every factual claim and model answer in the assigned TEXTBOOK_CONTENT key and assess only its assigned topic_ids. Source keys, topic IDs, and content-source labels are internal provenance fields attached later by the application; do not return them. PAST_PAPER_STYLE keys contribute only command words, mark-bearing task shape, and presentation style: never copy their subject matter or use them as factual evidence. A PAST_PAPER assignment is an original adapted Cambridge-style item, not an official past-paper question. Do not introduce technical terminology, complexity claims, or facts absent from the assigned textbook source. For a mock_test, question_structure_assignments are mandatory: use the exact question_type and marks for every number; do not demand more independent answer points than the available marks; every rubric must contain exactly one independently assessable string per mark, so a 4-mark question has exactly four rubric strings; question 1 is a one-mark MCQ with exactly one rubric point. Only a structure assignment with requires_coding=true may ask the learner to write code, pseudocode, or SQL; other questions about a programming or SQL topic must assess concepts, interpretation, tracing, explanation, or output rather than request code. If requires_coding is true and coding_kind is SQL, generate an applied SQL query task using only source-supported SQL clauses and allocate marks to selected fields, table, filtering, and required output/ordering/aggregation. If coding_kind is PROGRAMMING, generate an applied pseudocode/programming task with marks for algorithm logic, condition/comparison, iteration/progression, and output/termination. Cover every selected topic and make question marks sum exactly total_marks. If allows_code is false, do not ask for code, pseudocode, SQL, algorithm traces, or program writing. Whenever a question or model answer contains row-and-column data (including truth tables, trace tables, database tables, and dry-run tables), encode it as a complete GitHub-flavoured Markdown table with one row per line, a header row, and a separator row such as |---|---|. Never compress table rows into prose. For MCQ: produce exactly four options, set correct_option to exactly one option string, allocate exactly one mark, and use exactly one rubric criterion for selecting the correct option. An MCQ may assess application or interpretation but must never ask the learner to write code, construct a query, or complete a table. For non-MCQ: options and correct_option must be empty strings/lists."}
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    try:
        response, content, provider_attempts = _openai_json_with_retry(
            client,
            operation="assessment_generation",
            model=os.getenv("GENERATOR_MODEL", "gpt-4.1-mini"), temperature=0,
            # A 25-mark paper needs complete answers and independently
            # markable rubric points. Truncated JSON is never partially used.
            max_tokens=max(_bounded_env_int("ASSESSMENT_GENERATOR_MAX_TOKENS", 5000, 900, 12_000),
                           4500 if plan.get("assessment_type") == "mock_test" else 0),
            response_format={"type": "json_schema", "json_schema": {"name": "grounded_assessment", "strict": True, "schema": schema}},
            messages=[{"role": "system", "content": "You are a careful educational assessment author. " + prompt_text("assessment_generation")},
                      {"role": "user", "content": json.dumps(request, ensure_ascii=False)}],
        )
    except Exception as exc:
        retryable = _retryable_provider_error(exc)
        LOGGER.warning("Assessment generation provider call failed", exc_info=True)
        return {"status": "GENERATION_FAILED",
                "reason": "The assessment model did not return a complete structured response.",
                "error_category": type(exc).__name__, "retryable": retryable}
    if not isinstance(content.get("questions"), list) or len(content["questions"]) != plan["question_count"]:
        return {
            "status": "INVALID_GENERATION",
            "reason": "The assessment model returned an incomplete question collection.",
            "error_category": "INVALID_STRUCTURED_RESPONSE",
            "provider_attempts": provider_attempts,
        }
    assignment_by_number = {item["number"]: item for item in assignments}
    structure_by_number = {item["number"]: item for item in structure_assignments}
    for question in content["questions"]:
        assignment = assignment_by_number.get(question.get("number"))
        structure = structure_by_number.get(question.get("number"))
        canonical = _apply_system_owned_question_contract(question, assignment, structure, mapping)
        if not canonical["passed"]:
            return {"status": "INVALID_GENERATION", **{key: value for key, value in canonical.items()
                                                       if key != "passed"}}
        keys = canonical["keys"]
        _normalise_atomic_rubric(question, structure)
        _normalise_programming_rubric(question, structure)
        roles = {mapping[key].get("assessment_source_role") for key in keys}
        allowed_roles = {"TEXTBOOK_CONTENT"} if question["content_source"] == "TEXTBOOK" else {"TEXTBOOK_CONTENT", "PAST_PAPER_STYLE"}
        if not roles <= allowed_roles or (question["content_source"] == "PAST_PAPER" and not {"TEXTBOOK_CONTENT", "PAST_PAPER_STYLE"} <= roles):
            return {"status": "INVALID_GENERATION", "reason": "A question lacks the required factual source pairing or cites a style-only source."}
        textbook_source_text = " ".join(mapping[key].get("text", "") for key in keys
                                        if mapping[key].get("assessment_source_role") == "TEXTBOOK_CONTENT")
        if not _question_matches_assigned_topic(question, plan, textbook_source_text):
            return {"status": "INVALID_GENERATION", "reason": "Generated question drifted outside its assigned syllabus topic.",
                    "question": question.get("number"), "topic_ids": question.get("topic_ids", [])}
        unsupported_reason = _unsupported_advanced_claim(question, textbook_source_text)
        if unsupported_reason:
            return {"status": "INVALID_GENERATION", "reason": unsupported_reason,
                    "question": question.get("number")}
        question["citations"] = [{"document_id": mapping[key].get("document_id"), "chunk_id": mapping[key].get("chunk_id"),
                                  "page": mapping[key].get("page_start"),
                                  "source_role": mapping[key].get("assessment_source_role")} for key in dict.fromkeys(keys)]
    content["status"] = "PENDING_REVIEW"
    content["answer_type"] = "AI_GENERATED_MODEL_ANSWER"
    content["source_build_id"] = retrieval_result.get("retrieval_debug", {}).get("build_id")
    content["generation_diagnostics"] = {
        "provider_attempts": provider_attempts,
        "topics_without_topical_past_paper": retrieval_result.get("missing_past_paper_topic_ids", []),
        "context_chunks": len(catalog),
        "context_chars": sum(len(item["text"]) for item in catalog),
        **_provider_usage(response),
    }
    return content


def _validate_source_distribution(content: dict, plan: dict) -> dict:
    """Content-distribution agent: enforce balanced, auditable source origins.

    It is deterministic because a 50/50 claim must be mechanically provable,
    not inferred by a generative judge. Past-paper-adapted questions require a
    paper anchor *and* textbook answer authority.
    """
    questions = content.get("questions", [])
    expected = plan.get("content_distribution", {})
    textbook = sum(item.get("content_source") == "TEXTBOOK" for item in questions)
    past_paper = sum(item.get("content_source") == "PAST_PAPER" for item in questions)
    if textbook != expected.get("textbook_questions") or past_paper != expected.get("past_paper_questions"):
        return {"passed": False, "reason": "UNBALANCED_CONTENT_DISTRIBUTION",
                "textbook_questions": textbook, "past_paper_questions": past_paper, "expected": expected}
    # The generation call validates this before citations are attached; repeat
    # it here so a future caller cannot bypass the per-topic provenance plan.
    expected_assignments = {item["number"]: item for item in _question_source_assignments(
        plan, [{"source_key": f"T{index}", "source_role": "TEXTBOOK_CONTENT", "topic_ids": [topic_id]}
               for index, topic_id in enumerate(plan.get("selected_topic_ids", []), 1)] +
              [{"source_key": f"P{index}", "source_role": "PAST_PAPER_STYLE", "topic_ids": [topic_id]}
               for index, topic_id in enumerate(plan.get("selected_topic_ids", []), 1)])}
    for item in questions:
        assignment = expected_assignments.get(item.get("number"))
        if assignment and assignment["topic_ids"] and (item.get("content_source") != assignment["content_source"]
                                                         or item.get("topic_ids") != assignment["topic_ids"]):
            return {"passed": False, "reason": "SOURCE_TOPIC_ASSIGNMENT_DRIFT"}
        roles = {citation.get("source_role") for citation in item.get("citations", [])}
        if item.get("content_source") == "TEXTBOOK" and roles != {"TEXTBOOK_CONTENT"}:
            return {"passed": False, "reason": "INVALID_TEXTBOOK_CITATION_LANE"}
        if item.get("content_source") == "PAST_PAPER" and not {"TEXTBOOK_CONTENT", "PAST_PAPER_STYLE"} <= roles:
            return {"passed": False, "reason": "UNSUPPORTED_PAST_PAPER_ADAPTATION"}
    return {"passed": True, "reason": "Balanced textbook and past-paper distribution verified.",
            "textbook_questions": textbook, "past_paper_questions": past_paper, "expected": expected}


def _validate_marking_quality(content: dict, plan: dict | None = None) -> dict:
    """Marking-quality agent: reject rubrics that cannot support fair scoring.

    Mark-scheme sources guide this contract in the generation prompt and judge;
    this local gate makes the baseline independently reliable when an LLM judge
    is unavailable. Coding items need explicit algorithmic assessment points.
    """
    plan = plan or {}
    # Applied-response detection must describe the action required from the
    # learner, not merely vocabulary occurring in the question. For example,
    # "Which SQL query..." and "Which trace table..." are still MCQs and need
    # one decision criterion, not a four-dimensional programming rubric.
    applied_response_patterns = (
        r"\b(?:write|produce|develop|construct|design|amend|correct|debug)\b.{0,70}"
        r"\b(?:pseudocode|program(?:\s+code)?|algorithm|sql(?:\s+query|\s+statement)?|query)\b",
        r"\b(?:complete|draw|construct|produce)\b.{0,50}\btrace\s+table\b",
        r"\bwrite\s+(?:an?\s+)?(?:sql\s+)?query\b",
    )
    programming_groups = (
        {"logic", "algorithm"}, {"condition", "comparison", "selection"},
        {"loop", "iteration", "midpoint", "bounds", "index", "update", "search", "sort", "progression"},
        {"output", "termination", "stop", "not found", "result"})
    sql_groups = (
        {"select", "field", "column"}, {"from", "table"}, {"where", "condition", "criteria", "filter"},
        {"order", "aggregate", "count", "sum", "output", "result"})
    for item in content.get("questions", []):
        marks, rubric = int(item.get("marks", 0)), item.get("rubric", [])
        question_type = str(item.get("question_type", "")).upper()
        if marks < 1 or not rubric or not item.get("model_answer"):
            return {"passed": False, "reason": "INCOMPLETE_MARKING_KEY"}
        if question_type == "MCQ":
            if marks != 1 or len(rubric) != 1:
                return {"passed": False, "reason": "INVALID_MCQ_MARKING_PATTERN", "question": item.get("number"),
                        "marks": marks, "rubric_points": len(rubric)}
            # An MCQ's answer is one selected option. Words such as algorithm,
            # query, code, or trace table describe its subject matter and must
            # never trigger applied-response rubric validation.
            continue
        if len(rubric) != marks:
            return {"passed": False, "reason": "INSUFFICIENT_RUBRIC_GRANULARITY", "question": item.get("number"),
                    "marks": marks, "rubric_points": len(rubric)}
        question_text = item.get("question", "").lower()
        # Theory questions often mention an "algorithm" but are not code
        # questions. For mock tests the deterministic structure explicitly
        # assigns only the final item as coding when the selected scope allows
        # it. Generic assessments use explicit programming markers only.
        is_mock_test = plan.get("assessment_type") == "mock_test"
        is_assigned_mock_code = bool(is_mock_test and plan.get("allows_code")
                                     and item.get("number") == plan.get("question_count"))
        # Mock-test structure designates exactly one applied coding item.
        # Describing, interpreting, or tracing SQL/pseudocode elsewhere must
        # not be mistaken for an additional code-writing task.
        is_code_question = is_assigned_mock_code or (
            not is_mock_test and any(re.search(pattern, question_text, flags=re.IGNORECASE)
                                     for pattern in applied_response_patterns)
        )
        if is_assigned_mock_code:
            rubric_text = " ".join(rubric).lower()
            # A four-mark programming item needs independently assessable
            # logic, decision, progression, and completion behaviour. This
            # blocks a plausible-sounding model answer with a generic rubric.
            required_groups = sql_groups if plan.get("code_kind") == "SQL" else programming_groups
            if len(rubric) < min(marks, 4) or any(not any(term in rubric_text for term in group)
                                                for group in required_groups):
                reason = "SQL_RUBRIC_LACKS_QUERY_CRITERIA" if plan.get("code_kind") == "SQL" else "PROGRAMMING_RUBRIC_LACKS_LOGIC_CRITERIA"
                return {"passed": False, "reason": reason,
                        "question": item.get("number"), "marks": marks, "rubric_points": len(rubric)}
        elif is_code_question and not any(term in " ".join(rubric).lower() for term in
                                          ("logic", "algorithm", "sql", "query", "condition", "output", "result")):
            return {"passed": False, "reason": "APPLIED_ITEM_HAS_GENERIC_RUBRIC",
                    "question": item.get("number"), "marks": marks, "rubric_points": len(rubric)}
    return {"passed": True, "reason": "Rubric, marks, and answer-length baseline verified."}


def _requires_learner_coding(question: dict) -> bool:
    """Return whether the learner is explicitly required to produce code.

    Topic vocabulary is not enough. A theory item may discuss an algorithm,
    contain the words ``select`` and ``from``, or explain pseudocode without
    asking the learner to write any. Earlier validation searched the combined
    question/model-answer text and therefore misclassified those valid theory
    items. The learner-facing command is the only authoritative signal here.
    """
    prompt = re.sub(r"\s+", " ", str(question.get("question", "")).lower())
    applied_patterns = (
        r"\b(?:write|produce|develop|construct|design|amend|correct|debug|implement)\b"
        r".{0,100}\b(?:pseudocode|program(?:\s+code)?|algorithm|sql(?:\s+query|\s+statement)?|query)\b",
        r"\b(?:complete|draw|construct|produce|create)\b.{0,70}\btrace\s+table\b",
        r"\bwrite\s+(?:an?\s+)?(?:sql\s+)?query\b",
        r"\b(?:write|produce|implement)\b.{0,60}\b(?:python|java|visual\s+basic)\b",
    )
    return any(re.search(pattern, prompt, flags=re.IGNORECASE) for pattern in applied_patterns)


def _validate_mock_test_scope(content: dict, plan: dict) -> dict:
    """Mock-test agent: prove the 25-mark paper stays inside selected topics."""
    if plan.get("assessment_type") != "mock_test":
        return {"passed": True, "reason": "Not a mock test."}
    questions = content.get("questions", [])
    selected = set(plan.get("selected_topic_ids", []))
    if not selected:
        return {"passed": False, "reason": "MOCK_TEST_HAS_NO_SELECTED_SYLLABUS_TOPICS"}
    total_marks = sum(int(item.get("marks", 0)) for item in questions)
    if total_marks != int(plan.get("total_marks") or 0):
        return {"passed": False, "reason": "MOCK_TEST_MARK_TOTAL_IS_NOT_25", "actual_marks": total_marks}
    if len(questions) < int(plan.get("mock_test_structure", {}).get("minimum_questions", 7)):
        return {"passed": False, "reason": "MOCK_TEST_REQUIRES_AT_LEAST_SEVEN_QUESTIONS"}
    mark_values = {int(item.get("marks", 0)) for item in questions}
    if len(mark_values) < 3 or not any(value >= 4 for value in mark_values) or any(value > 6 for value in mark_values):
        return {"passed": False, "reason": "MOCK_TEST_MARK_DISTRIBUTION_IS_NOT_STRUCTURED"}
    if not any(item.get("question_type") == "MCQ" and int(item.get("marks", 0)) == 1 for item in questions):
        return {"passed": False, "reason": "MOCK_TEST_REQUIRES_A_ONE_MARK_MCQ"}
    covered: set[str] = set()
    contains_coding_item = False
    for item in questions:
        item_topics = set(item.get("topic_ids", []))
        if not item_topics or not item_topics <= selected:
            return {"passed": False, "reason": "MOCK_TEST_SCOPE_DEVIATION", "question": item.get("number")}
        covered.update(item_topics)
        requires_coding = bool(item.get("requires_coding")) or _requires_learner_coding(item)
        contains_coding_item = contains_coding_item or requires_coding
        if not plan.get("allows_code", True) and requires_coding:
            return {"passed": False, "reason": "THEORY_ONLY_MOCK_TEST_CONTAINS_CODING",
                    "question": item.get("number")}
    if plan.get("allows_code", True) and not contains_coding_item:
        return {"passed": False, "reason": "CODING_CAPABLE_MOCK_TEST_REQUIRES_AN_APPLIED_CODING_ITEM"}
    if not selected <= covered:
        return {"passed": False, "reason": "MOCK_TEST_DOES_NOT_COVER_ALL_SELECTED_TOPICS",
                "uncovered_topic_ids": sorted(selected - covered)}
    return {"passed": True, "reason": "Exact 25-mark selected-topic mock test verified.",
            "total_marks": total_marks, "covered_topic_ids": sorted(covered)}


def _validate(content: dict, plan: dict, chunks: list[dict], distribution: dict | None = None,
              marking_quality: dict | None = None, mock_test_scope: dict | None = None) -> dict:
    """Run deterministic checks then optional LLM-as-judge validation.

    Judge output cannot approve an invalid contract. It only adds an
    answerability/difficulty check after every citation, answer, and rubric has
    already been validated locally.
    """
    if content.get("status") != "PENDING_REVIEW":
        details = {key: content[key] for key in ("question", "topic_ids") if key in content}
        return {"passed": False, "reason": content.get("reason", "Generation did not complete."), **details}
    questions = content.get("questions", [])
    distribution = distribution or _validate_source_distribution(content, plan)
    marking_quality = marking_quality or _validate_marking_quality(content, plan)
    mock_test_scope = mock_test_scope or _validate_mock_test_scope(content, plan)
    if not distribution["passed"]:
        return {"passed": False, "reason": distribution["reason"], "provider": "distribution_validator",
                "source_distribution": distribution, "marking_quality": marking_quality}
    if not marking_quality["passed"]:
        return {"passed": False, "reason": marking_quality["reason"], "provider": "marking_quality_validator",
                "source_distribution": distribution, "marking_quality": marking_quality}
    if not mock_test_scope["passed"]:
        return {"passed": False, "reason": mock_test_scope["reason"], "provider": "mock_test_scope_validator",
                "source_distribution": distribution, "marking_quality": marking_quality, "mock_test_scope": mock_test_scope}
    requested_format = plan.get("question_format", "MIXED")
    def question_valid(item: dict) -> bool:
        basic = item.get("question") and item.get("model_answer") and item.get("rubric") and item.get("citations")
        question_type = item.get("question_type")
        options = item.get("options", [])
        correct = item.get("correct_option", "")
        is_mcq = question_type == "MCQ"
        format_matches = requested_format == "MIXED" or (requested_format == "MCQ" and is_mcq) or (requested_format == question_type)
        option_contract = len(options) == 4 and correct in options if is_mcq else not options and not correct
        return bool(basic and format_matches and option_contract)
    valid = (len(questions) == plan["question_count"]
             and {item.get("number") for item in questions} == set(range(1, plan["question_count"] + 1))
             and all(question_valid(item) for item in questions))
    if not valid:
        return {"passed": False, "reason": "A question violates its answer, rubric, evidence, or question-format contract.", "provider": "deterministic",
                "source_distribution": distribution, "marking_quality": marking_quality, "mock_test_scope": mock_test_scope}
    if not os.getenv("OPENAI_API_KEY"):
        return {"passed": True, "reason": "LLM judge not run because OPENAI_API_KEY is not configured.", "provider": "deterministic",
                "source_distribution": distribution, "marking_quality": marking_quality, "mock_test_scope": mock_test_scope}
    from openai import OpenAI
    # Structural properties are already hard-gated above. The probabilistic
    # judge receives a deliberately non-overlapping responsibility: semantic
    # answerability, factual support, and difficulty alignment only. This
    # prevents an opaque LLM opinion from contradicting deterministic rubric,
    # mark-total, topic-scope, and provenance checks.
    schema = {"type": "object", "additionalProperties": False,
              "required": ["source_entailment", "model_answer_correctness", "difficulty_alignment", "reason"],
              "properties": {"source_entailment": {"type": "boolean"},
                             "model_answer_correctness": {"type": "boolean"},
                             "difficulty_alignment": {"type": "boolean"},
                             "reason": {"type": "string"}}}
    try:
        cited_chunk_ids = {
            str(citation.get("chunk_id"))
            for question in questions
            for citation in question.get("citations", [])
            if citation.get("chunk_id")
        }
        judged_chunks = [chunk for chunk in chunks if str(chunk.get("chunk_id")) in cited_chunk_ids]
        response, judged, judge_attempts = _openai_json_with_retry(
            OpenAI(api_key=os.environ["OPENAI_API_KEY"]),
            operation="assessment_semantic_validation",
            model=os.getenv("ASSESSMENT_JUDGE_MODEL", os.getenv("GENERATOR_MODEL", "gpt-4.1-mini")), temperature=0, max_tokens=180,
            response_format={"type": "json_schema", "json_schema": {"name": "assessment_semantic_validation", "strict": True, "schema": schema}},
            messages=[{"role": "system", "content": "You are the semantic grounding judge for a teacher-reviewable assessment draft. Evaluate only: (1) whether each question and model answer are entailed by its cited TEXTBOOK_CONTENT passages; (2) whether each model answer is technically correct; and (3) whether the intellectual difficulty matches the requested level. PAST_PAPER_STYLE passages contribute command-word and task structure only and must not be treated as factual evidence or required to discuss the selected topic. Do not evaluate or reject question count, mark totals, source ratios, topic-ID coverage, coding permission, mark allocation, rubric wording, rubric granularity, or answer length. Those properties have already passed authoritative deterministic validators. MARKING_PATTERN and CURRICULUM_SCOPE passages are not factual evidence. Return JSON only using the requested semantic fields."},
                      {"role": "user", "content": json.dumps({"blueprint": {"topic": plan.get("topic"),
                                                                               "difficulty": plan.get("difficulty"),
                                                                               "selected_topic_ids": plan.get("selected_topic_ids", [])},
                                                               "assessment": content,
                                                               "sources": _source_catalog(judged_chunks, maximum_chars=40_000)[0]}, ensure_ascii=False)}],
        )
        semantic_passed = all(judged.get(field) is True for field in
                              ("source_entailment", "model_answer_correctness", "difficulty_alignment"))
        return {"passed": semantic_passed, "reason": judged["reason"], "provider": "openai_semantic_judge",
                "semantic_checks": {field: judged[field] for field in
                                    ("source_entailment", "model_answer_correctness", "difficulty_alignment")},
                "provider_attempts": judge_attempts,
                **_provider_usage(response),
                "source_distribution": distribution, "marking_quality": marking_quality, "mock_test_scope": mock_test_scope}
    except Exception as exc:
        # A judge outage must not disguise a live assessment as validated.
        return {"passed": False, "reason": "The semantic validation service did not complete.",
                "error_category": type(exc).__name__, "retryable": _retryable_provider_error(exc),
                "provider": "openai_structured",
                "source_distribution": distribution, "marking_quality": marking_quality, "mock_test_scope": mock_test_scope}


def _graph():
    from langgraph.graph import END, START, StateGraph

    def plan_node(state: AssessmentState) -> dict:
        fallback = _plan(state["topic"], state["difficulty"], state["assessment_type"], state["question_count"],
                         state.get("question_format", "mixed"), state.get("selected_topics"), state.get("total_marks"),
                         state.get("allows_code", True), state.get("code_kind"))
        return {"blueprint": _plan_with_llm(fallback), "retry_count": 0}

    def retrieve_node(state: AssessmentState) -> dict:
        blueprint = state["blueprint"]
        return {"retrieval": retrieve_assessment_evidence(blueprint["topic"], blueprint["assessment_type"], state.get("level"),
                                                           blueprint.get("selected_topics"))}

    def generate_node(state: AssessmentState) -> dict:
        return {"content": _generate(state["blueprint"], state["retrieval"])}

    def distribution_node(state: AssessmentState) -> dict:
        return {"distribution": _validate_source_distribution(state["content"], state["blueprint"])}

    def marking_quality_node(state: AssessmentState) -> dict:
        return {"marking_quality": _validate_marking_quality(state["content"], state["blueprint"])}

    def mock_test_scope_node(state: AssessmentState) -> dict:
        return {"mock_test_scope": _validate_mock_test_scope(state["content"], state["blueprint"])}

    def validate_node(state: AssessmentState) -> dict:
        validation = _validate(state["content"], state["blueprint"], state["retrieval"].get("chunks", []),
                               state.get("distribution"), state.get("marking_quality"), state.get("mock_test_scope"))
        return {"validation": validation,
                "result": {"blueprint": state["blueprint"], "content": state["content"], "validation": validation,
                           "retrieval_debug": state["retrieval"].get("retrieval_debug", {})}}

    def retry_route(state: AssessmentState) -> str:
        maximum_retries = _bounded_env_int("ASSESSMENT_VALIDATION_RETRIES", 2, 0, 3)
        content_status = state.get("content", {}).get("status")
        retryable_content = content_status in {"INVALID_GENERATION", "GENERATION_FAILED"}
        retryable_validation = state["validation"].get("provider") in {
            "deterministic", "distribution_validator", "marking_quality_validator",
            "mock_test_scope_validator", "openai_semantic_judge",
        }
        should_retry = (retryable_content or retryable_validation) and bool(os.getenv("OPENAI_API_KEY"))
        return "retry" if not state["validation"]["passed"] and should_retry and state.get("retry_count", 0) < maximum_retries else "finish"

    def retry_node(state: AssessmentState) -> dict:
        blueprint = dict(state["blueprint"])
        blueprint["validation_feedback"] = {
            "reason": state["validation"].get("reason"),
            "provider": state["validation"].get("provider"),
            "content_status": state.get("content", {}).get("status"),
            "content_reason": state.get("content", {}).get("reason"),
        }
        return {"blueprint": blueprint, "retry_count": state.get("retry_count", 0) + 1}

    graph = StateGraph(AssessmentState)
    graph.add_node("plan", plan_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("validate_source_distribution", distribution_node)
    graph.add_node("validate_marking_quality", marking_quality_node)
    graph.add_node("validate_mock_test_scope", mock_test_scope_node)
    graph.add_node("validate", validate_node)
    graph.add_node("retry", retry_node)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "validate_source_distribution")
    graph.add_edge("validate_source_distribution", "validate_marking_quality")
    graph.add_edge("validate_marking_quality", "validate_mock_test_scope")
    graph.add_edge("validate_mock_test_scope", "validate")
    graph.add_conditional_edges("validate", retry_route, {"retry": "retry", "finish": END})
    graph.add_edge("retry", "generate")
    return graph.compile()


def generate_assessment(topic: str, difficulty: str, assessment_type: str, question_count: int,
                        question_format: str = "mixed", level: str | None = None,
                        selected_topics: list[dict] | None = None, total_marks: int | None = None,
                        allows_code: bool = True, code_kind: str | None = None) -> dict:
    """Run the workflow with one trace spanning retrieval through validation."""
    trace_id = str(uuid.uuid4())
    started = time.perf_counter()
    result: dict[str, Any] | None = None
    failure_category: str | None = None
    try:
        state = _graph().invoke({
            "topic": topic, "difficulty": difficulty, "assessment_type": assessment_type,
            "question_count": question_count, "question_format": question_format, "level": level,
            "selected_topics": selected_topics or [], "total_marks": total_marks,
            "allows_code": allows_code, "code_kind": code_kind,
        })
        result = state["result"]
        result["trace_id"] = trace_id
        result["retry_count"] = int(state.get("retry_count", 0))
        result["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        if not result.get("validation", {}).get("passed"):
            failure_category = (
                result.get("content", {}).get("error_category")
                or result.get("validation", {}).get("error_category")
                or result.get("validation", {}).get("reason")
                or "ASSESSMENT_VALIDATION_FAILED"
            )
        return result
    except Exception as exc:
        failure_category = type(exc).__name__
        LOGGER.exception("Assessment workflow failed trace_id=%s type=%s", trace_id, assessment_type)
        raise
    finally:
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        validation = (result or {}).get("validation", {})
        content = (result or {}).get("content", {})
        generation_diagnostics = content.get("generation_diagnostics", {})
        input_tokens = int(generation_diagnostics.get("input_tokens", 0)) + int(
            validation.get("input_tokens", 0)
        )
        output_tokens = int(generation_diagnostics.get("output_tokens", 0)) + int(
            validation.get("output_tokens", 0)
        )
        total_cost = float(generation_diagnostics.get("total_cost", 0)) + float(
            validation.get("total_cost", 0)
        )
        try:
            TelemetryStore().record({
                "trace_id": trace_id,
                "session_id": None,
                "category": "ASSESSMENT_GENERATION",
                "intent": assessment_type.upper(),
                "success": bool(validation.get("passed")),
                "failure_category": failure_category,
                "latency_ms": latency_ms,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_cost": round(total_cost, 8),
                "cache_hit": False,
                "model": os.getenv("GENERATOR_MODEL", "gpt-4.1-mini") if os.getenv("OPENAI_API_KEY") else None,
                "assessment_type": assessment_type,
                "level": level,
                "selected_topic_ids": [item.get("id") for item in (selected_topics or [])],
                "question_count": question_count,
                "provider_attempts": generation_diagnostics.get("provider_attempts"),
                "context_chunks": generation_diagnostics.get("context_chunks"),
                "retry_count": (result or {}).get("retry_count"),
                "validation_provider": validation.get("provider"),
                "content_status": content.get("status"),
            })
        except Exception:
            LOGGER.warning("Assessment telemetry could not be recorded trace_id=%s", trace_id, exc_info=True)
