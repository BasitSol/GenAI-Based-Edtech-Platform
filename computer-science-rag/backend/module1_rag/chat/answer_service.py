"""Grounded answer orchestration with a machine-validated output contract.

The generator never asks a model to format canonical citations inside prose.
Instead, the model returns source keys for each answer section and this module
maps those keys back to immutable chunk metadata.  This prevents citations
from corrupting SQL, code, tables, and ordinary sentences in the student UI.
"""
from __future__ import annotations

import base64
import json
import os
import time
import uuid
from typing import Any

from backend.shared.core import ROOT
from backend.module1_rag.chat.memory.conversation_store import ConversationStore
from backend.module1_rag.monitoring.telemetry import TelemetryStore
from backend.module1_rag.monitoring.tracing import traced
from backend.module1_rag.chat.workflow import retrieve
from backend.shared.prompts import PROMPT_LIBRARY_VERSION, SYSTEM_PROMPT, generation_prompt
from .verification import verify_response


def _source_catalog(chunks: list[dict]) -> tuple[list[dict], dict[str, dict]]:
    """Create short, prompt-local source keys while preserving canonical IDs."""
    selected, mapping = [], {}
    for chunk in chunks:
        if chunk.get("document_type") == "MARKING_PATTERN" or chunk.get("relationship") == "ASSESSMENT_PATTERN":
            continue
        key = f"S{len(selected) + 1}"
        mapping[key] = chunk
        selected.append({
            "source_key": key,
            "document_id": chunk.get("document_id"),
            "document_type": chunk.get("document_type"),
            "source_role": chunk.get("relationship", "CURRICULUM_EVIDENCE"),
            "page": chunk.get("page_start"),
            "chunk_id": chunk.get("chunk_id"),
            "chapter": chunk.get("chapter") or chunk.get("section_title"),
            "topic": chunk.get("topic") or chunk.get("section_title"),
            "text": chunk.get("text", "")[:2200],
        })
    return selected, mapping


def _canonical_citations(source_keys: list[str], mapping: dict[str, dict]) -> list[dict]:
    citations, seen = [], set()
    for key in source_keys:
        chunk = mapping.get(key)
        if not chunk:
            raise ValueError(f"The model cited unknown source key {key!r}")
        identity = (chunk.get("document_id"), chunk.get("page_start"), chunk.get("chunk_id"))
        if identity in seen:
            continue
        seen.add(identity)
        citations.append({
            "source_key": key,
            "document_id": identity[0],
            "page": identity[1],
            "chunk_id": identity[2],
            "document_type": chunk.get("document_type"),
            "chapter": chunk.get("chapter") or chunk.get("section_title"),
            "topic": chunk.get("topic") or chunk.get("section_title"),
            "relationship": chunk.get("relationship", "EVIDENCE"),
        })
    return citations


def _source_details(chunks: list[dict]) -> list[dict]:
    details, seen = [], set()
    for chunk in chunks:
        identity = (chunk.get("document_id"), chunk.get("page_start"), chunk.get("chunk_id"))
        if identity in seen:
            continue
        seen.add(identity)
        details.append({
            "book_name": chunk.get("book_name") or chunk.get("document_title") or chunk.get("document_id"),
            "chapter": chunk.get("chapter") or chunk.get("section_title"),
            "topic": chunk.get("topic") or chunk.get("section_title") or chunk.get("content_type"),
            "page": chunk.get("page_start"),
            "source_document": chunk.get("document_id"),
            "chunk_id": chunk.get("chunk_id"),
            "document_type": chunk.get("document_type"),
        })
    return details


def _official_scheme_answer(chunks: list[dict]) -> tuple[str, list[dict]]:
    """Return exact mark-scheme text without inventing explanatory prose."""
    schemes = [item for item in chunks if item.get("document_type") == "MARK_SCHEME"]
    answer = "\n\n".join(item.get("text", "").strip() for item in schemes if item.get("text", "").strip())
    _, mapping = _source_catalog(schemes)
    return answer, _canonical_citations(list(mapping), mapping)


def _parse_generation(payload: str, mapping: dict[str, dict]) -> tuple[str, list[dict]]:
    """Validate the LLM JSON before any content reaches the user."""
    data = json.loads(payload)
    if not isinstance(data, dict) or not isinstance(data.get("answer_markdown"), str):
        raise ValueError("Generation did not satisfy the answer JSON contract")
    answer = data["answer_markdown"].strip()
    source_keys = data.get("source_keys")
    if not answer or not isinstance(source_keys, list):
        raise ValueError("Generation returned an empty answer or invalid source_keys")
    # Citation markers inside prose/code are forbidden by design.
    if any(f"[{key}]" in answer for key in mapping):
        raise ValueError("Generation placed source labels inside answer text")
    return answer, _canonical_citations([str(key) for key in source_keys], mapping)


def _generate_with_openai(query: str, chunks: list[dict], route: dict, answer_type: str) -> tuple[str, list[dict], dict]:
    """Generate one schema-constrained answer; environment loading occurs upstream."""
    from openai import OpenAI

    catalog, mapping = _source_catalog(chunks)
    prompt = generation_prompt(route)
    request = {
        "question": query,
        "answer_type": answer_type,
        "question_profile": route,
        "category_instructions": prompt.template,
        "sources": catalog,
        "output_contract": {
            "answer_markdown": "Complete student-facing answer. Never include source keys or citations in this text.",
            "source_keys": "Distinct S-number keys that directly support the answer.",
        },
    }
    content: list[dict[str, Any]] = [{"type": "text", "text": json.dumps(request, ensure_ascii=False)}]
    visual_terms = ("diagram", "figure", "logic circuit", "flowchart", "shown below")
    if any(term in query.lower() for term in visual_terms):
        figure = next((item for item in chunks if item.get("figure_path")), None)
        if figure:
            path = ROOT / figure["figure_path"]
            if path.exists():
                content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii"), "detail": "high"}})
    response_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["answer_markdown", "source_keys"],
        "properties": {
            "answer_markdown": {"type": "string", "minLength": 1},
            "source_keys": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        },
    }
    response = OpenAI(api_key=os.environ["OPENAI_API_KEY"]).chat.completions.create(
        model=os.getenv("GENERATOR_MODEL", "gpt-4.1-mini"),
        temperature=0,
        max_tokens=int(os.getenv("GENERATOR_MAX_TOKENS", "900")),
        response_format={"type": "json_schema", "json_schema": {"name": "grounded_answer", "strict": True, "schema": response_schema}},
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}],
    )
    answer, citations = _parse_generation(response.choices[0].message.content or "", mapping)
    usage = response.usage
    return answer, citations, {
        "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
    }


def _estimated_cost(model: str, usage: dict) -> float:
    # Pricing is configuration, not a hidden claim that remains correct forever.
    input_price = float(os.getenv("GENERATOR_INPUT_USD_PER_MILLION", "0.40"))
    output_price = float(os.getenv("GENERATOR_OUTPUT_USD_PER_MILLION", "1.60"))
    return round((usage["input_tokens"] * input_price + usage["output_tokens"] * output_price) / 1_000_000, 8)


@traced("educational_rag_answer", run_type="chain")
def answer_question(query: str, level: str | None = None, exam_year: int | None = None,
                    conversation_id: str | None = None, difficulty: str | None = None) -> dict:
    """Run retrieval, generation, validation, persistence, and local telemetry."""
    trace_id, started = str(uuid.uuid4()), time.perf_counter()
    memory = ConversationStore()
    state = memory.get(conversation_id)
    retrieval = retrieve(query, level or state.get("selected_level"), exam_year or state.get("selected_exam_year"), state)
    chunks, route = retrieval["chunks"], retrieval["route"]
    if difficulty:
        route["difficulty"] = difficulty.upper()
    exact = bool(retrieval["exact_mark_scheme_available"])
    context_sufficient = bool(retrieval.get("retrieval_debug", {}).get("context_sufficient") and chunks)
    usage = {"input_tokens": 0, "output_tokens": 0}
    generation_error = None

    if not context_sufficient:
        answer, citations = "I could not find sufficient curriculum evidence to answer this reliably.", []
        answer_type, provider, execution_status = "INSUFFICIENT_SOURCE", "none", "COMPLETED_ABSTENTION"
    elif exact:
        answer, citations = _official_scheme_answer(chunks)
        answer_type, provider, execution_status = "OFFICIAL_MARK_SCHEME_SUPPORTED_ANSWER", "deterministic_mark_scheme", "COMPLETED"
    elif not os.getenv("OPENAI_API_KEY"):
        answer, citations = "Generation was not run because OPENAI_API_KEY is not configured.", []
        answer_type, provider, execution_status = "GENERATION_NOT_RUN", "none", "SKIPPED_MISSING_API_KEY"
    else:
        answer_type = "AI_GENERATED_MODEL_ANSWER" if route.get("intent") == "EXAM_ANSWER" else "CURRICULUM_EXPLANATION"
        try:
            answer, citations, usage = _generate_with_openai(query, chunks, route, answer_type)
            provider, execution_status = "openai", "COMPLETED"
        except Exception as exc:
            answer, citations = "Answer generation failed. Review the execution diagnostics and retry.", []
            provider, execution_status = "openai", "FAILED_GENERATION"
            generation_error = f"{type(exc).__name__}: {str(exc)[:400]}"

    verification = verify_response(answer, chunks, route, context_sufficient, citations=citations,
                                   generation_completed=execution_status == "COMPLETED")
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    model = os.getenv("GENERATOR_MODEL", "gpt-4.1-mini")
    generation_cost = _estimated_cost(model, usage) if provider == "openai" else 0.0
    embedding_cost = float(retrieval.get("retrieval_debug", {}).get("embedding_cost", 0.0) or 0.0)
    cost = round(generation_cost + embedding_cost, 8)
    conversation_id = memory.record(state, query, answer, route, chunks)
    response = {
        "answer": answer, "answer_type": answer_type, "execution_status": execution_status,
        "exact_mark_scheme_available": exact, "generation_provider": provider,
        "generator_model": model if provider == "openai" else None,
        "question_understanding": route, "prompt_version": PROMPT_LIBRARY_VERSION,
        "generation_prompt": generation_prompt(route).name,
        "citations": citations, "source_details": _source_details(chunks), "retrieved_chunks": chunks,
        "conversation_id": conversation_id, "trace_id": trace_id, "latency_ms": latency_ms,
        "input_tokens": usage["input_tokens"], "output_tokens": usage["output_tokens"],
        "estimated_cost": cost, "cost_breakdown": {"embedding": embedding_cost, "generation": generation_cost, "total": cost},
        "technical_failure": execution_status.startswith("FAILED"),
        "generation_error": generation_error, "citation_valid": not verification["invalid_citations"],
        "verification": verification, "retrieval_debug": retrieval["retrieval_debug"],
    }
    try:
        TelemetryStore().record({
            "trace_id": trace_id, "session_id": conversation_id, "query": query,
            "category": route.get("category"), "intent": route.get("intent"),
            "success": execution_status in {"COMPLETED", "COMPLETED_ABSTENTION"},
            "failure_category": verification.get("failure_category"), "latency_ms": latency_ms,
            "input_tokens": usage["input_tokens"], "output_tokens": usage["output_tokens"],
            "total_cost": cost, "retrieved_documents": [item.get("document_id") for item in chunks],
            "retrieved_chunks": [item.get("chunk_id") for item in chunks], "prompt_version": PROMPT_LIBRARY_VERSION,
            "model": response["generator_model"], "execution_status": execution_status,
        })
    except Exception as exc:  # Telemetry must never break a student answer.
        response["telemetry_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    return response
