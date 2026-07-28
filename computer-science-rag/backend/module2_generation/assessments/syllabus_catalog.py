"""Syllabus-backed chapter and topic catalogue for Phase 2 mock tests.

The UI must not invent curriculum labels.  This module exposes a small,
versioned projection of the supplied Cambridge syllabus documents and first
verifies that the active immutable RAG build contains the relevant syllabus.
The projection keeps UI labels stable while retrieval continues to use the
original syllabus/textbook/paper chunks as evidence.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from backend.shared.core import current_build_path
from backend.module2_generation.assessments.book_catalog import public_book_catalog


# These chapter/topic identifiers follow the supplied 2210 and 9618 syllabus
# content-overview/subject-content structure. `allows_code` is intentionally
# conservative: a theory-only selection can never silently introduce code.
CATALOG: dict[str, tuple[dict[str, Any], ...]] = {
    "O_LEVEL": (
        {"id": "ol_data_representation", "name": "Data representation", "topics": (
            ("ol_number_systems", "Number systems", False),
            ("ol_text_sound_images", "Text, sound and images", False),
            ("ol_data_storage", "Data storage and compression", False),
        )},
        {"id": "ol_data_transmission", "name": "Data transmission", "topics": (
            ("ol_data_transmission_methods", "Types and methods of data transmission", False),
            ("ol_error_detection", "Methods of error detection", False),
            ("ol_encryption", "Encryption", False),
        )},
        {"id": "ol_hardware", "name": "Hardware", "topics": (
            ("ol_computer_architecture", "Computer architecture", False),
            ("ol_input_output", "Input and output devices", False),
            ("ol_data_storage_hardware", "Data storage", False),
            ("ol_network_hardware", "Network hardware", False),
        )},
        {"id": "ol_software", "name": "Software", "topics": (
            ("ol_system_software", "Types of software and interrupts", False),
            ("ol_languages_translators",
             "Types of programming language, translators and integrated development environments (IDEs)", False),
        )},
        {"id": "ol_internet", "name": "The internet and its uses", "topics": (
            ("ol_internet_networks", "The internet and the world wide web", False),
            ("ol_digital_currency", "Digital currency", False),
            ("ol_cyber_security", "Cyber security", False),
        )},
        {"id": "ol_automated_tech", "name": "Automated and emerging technologies", "topics": (
            ("ol_automated_systems", "Automated systems", False),
            ("ol_robotics", "Robotics", False),
            ("ol_artificial_intelligence", "Artificial intelligence", False),
        )},
        {"id": "ol_algorithms", "name": "Algorithm design and problem-solving", "topics": (
            ("ol_program_development", "Program development life cycle", True),
            ("ol_decomposition_design", "Decomposition and solution design", True),
            ("ol_algorithm_purpose", "Purpose of algorithms", True),
            ("ol_searching_sorting", "Standard methods of solution", True),
            ("ol_programming_validation", "Validation and verification", True),
            ("ol_test_data", "Test data", True),
            ("ol_trace_tables", "Trace tables", True),
            ("ol_algorithm_errors", "Identifying errors in algorithms", True),
            ("ol_algorithm_design", "Writing and amending algorithms", True),
        )},
        {"id": "ol_programming", "name": "Programming", "topics": (
            ("ol_programming_constructs", "Programming concepts", True),
            ("ol_arrays", "Arrays", True),
            ("ol_file_handling", "File handling", True),
        )},
        {"id": "ol_databases", "name": "Databases", "topics": (
            ("ol_database_concepts", "Single-table database concepts", False),
            ("ol_sql", "Structured Query Language (SQL)", True),
        )},
        {"id": "ol_boolean_logic", "name": "Boolean logic", "topics": (
            ("ol_logic_gates", "Logic gates and Boolean expressions", False),
        )},
    ),
    "A_LEVEL": (
        {"id": "al_information_representation", "name": "Information representation", "topics": (
            ("al_data_representation_as", "Data Representation", False),
            ("al_multimedia", "Multimedia - Graphics, Sound", False),
            ("al_compression", "Compression", False),
        )},
        {"id": "al_communication", "name": "Communication", "topics": (
            ("al_networking", "Networks including the internet", False),
        )},
        {"id": "al_hardware", "name": "Hardware", "topics": (
            ("al_computer_components", "Computers and their components", False),
            ("al_logic_gates", "Logic Gates and Logic Circuits", False),
        )},
        {"id": "al_processor_fundamentals", "name": "Processor Fundamentals", "topics": (
            ("al_cpu_architecture", "Central Processing Unit (CPU) Architecture", False),
            ("al_assembly_language", "Assembly Language", True),
            ("al_bit_manipulation", "Bit manipulation", True),
        )},
        {"id": "al_system_software", "name": "System Software", "topics": (
            ("al_operating_systems", "Operating Systems", False),
            ("al_language_translators", "Language Translators", False),
        )},
        {"id": "al_security_privacy", "name": "Security, privacy and data integrity", "topics": (
            ("al_data_security", "Data Security", False),
            ("al_data_integrity", "Data Integrity", False),
        )},
        {"id": "al_ethics_ownership", "name": "Ethics and Ownership", "topics": (
            ("al_ethics_ownership", "Ethics and Ownership", False),
        )},
        {"id": "al_databases", "name": "Databases", "topics": (
            ("al_database_concepts", "Database Concepts", False),
            ("al_dbms", "Database Management Systems (DBMS)", False),
            ("al_sql", "Data Definition Language (DDL) and Data Manipulation Language (DML)", True),
        )},
        {"id": "al_algorithm_design", "name": "Algorithm Design and Problem-solving", "topics": (
            ("al_computational_thinking", "Computational Thinking Skills", True),
            ("al_algorithms", "Algorithms", True),
        )},
        {"id": "al_data_types_structures", "name": "Data Types and Structures", "topics": (
            ("al_data_types_records", "Data Types and Records", True),
            ("al_arrays", "Arrays", True),
            ("al_files", "Files", True),
            ("al_abstract_data_types", "Introduction to Abstract Data Types (ADT)", True),
        )},
        {"id": "al_programming", "name": "Programming", "topics": (
            ("al_programming_basics", "Programming Basics", True),
            ("al_constructs", "Constructs", True),
            ("al_structured_programming", "Structured Programming", True),
        )},
        {"id": "al_software_development", "name": "Software Development", "topics": (
            ("al_program_development_lifecycle", "Program Development Life cycle", True),
            ("al_program_design", "Program Design", True),
            ("al_program_testing", "Program Testing and Maintenance", True),
        )},
        {"id": "al_data_representation", "name": "Data Representation", "topics": (
            ("al_user_defined_types", "User-defined data types", True),
            ("al_file_organisation", "File organisation and access", True),
            ("al_floating_point", "Floating-point numbers, representation and manipulation", False),
        )},
        {"id": "al_communication_internet", "name": "Communication and internet technologies", "topics": (
            ("al_protocols", "Protocols", False),
            ("al_switching", "Circuit switching, packet switching", False),
        )},
        {"id": "al_hardware_virtual_machines", "name": "Hardware and Virtual Machines", "topics": (
            ("al_processors_parallel_virtual", "Processors, Parallel Processing and Virtual Machines", False),
            ("al_boolean_algebra", "Boolean Algebra and Logic Circuits", False),
        )},
        {"id": "al_advanced_system_software", "name": "System Software", "topics": (
            ("al_os_purposes", "Purposes of an Operating System (OS)", False),
            ("al_translation_software", "Translation Software", False),
        )},
        {"id": "al_security", "name": "Security", "topics": (
            ("al_encryption", "Encryption, Encryption Protocols and Digital certificates", False),
        )},
        {"id": "al_artificial_intelligence_chapter", "name": "Artificial Intelligence (AI)", "topics": (
            ("al_artificial_intelligence", "Artificial Intelligence", False),
        )},
        {"id": "al_computational_problem_solving", "name": "Computational thinking and Problem-solving", "topics": (
            ("al_advanced_algorithms", "Algorithms", True),
            ("al_recursion", "Recursion", True),
        )},
        {"id": "al_further_programming", "name": "Further Programming", "topics": (
            ("al_programming_paradigms", "Programming Paradigms", True),
            ("al_file_processing", "File Processing and Exception Handling", True),
        )},
    ),
}


# Stable retrieval vocabulary bridges concise syllabus/UI labels to wording
# used inside textbooks and question papers. Only genuine curriculum concepts
# belong here; it is not a source of facts or generated question content.
TOPIC_RETRIEVAL_PROFILES: dict[str, dict[str, tuple[str, ...]]] = {
    "ol_algorithm_design": {
        "content": ("algorithm", "algorithm design", "decomposition", "pseudocode", "flowchart",
                    "structure diagram", "program development life cycle"),
        "past_paper": ("algorithm", "pseudocode", "flowchart", "design of a solution", "decomposition"),
    },
    "ol_searching_sorting": {
        "content": ("linear search", "bubble sort", "totalling", "counting",
                    "maximum", "minimum", "average"),
        # Past papers may provide a closely related algorithm task/style anchor
        # even when the small supplied paper sample has no named search item.
        "past_paper": ("linear search", "bubble sort", "totalling", "counting",
                       "maximum", "minimum", "average", "algorithm", "pseudocode"),
    },
    "ol_programming_constructs": {
        "content": ("programming concepts", "variables", "constants", "data types",
                    "input", "output", "sequence", "selection", "iteration",
                    "string handling", "operators", "procedures", "functions"),
        "past_paper": ("pseudocode", "program code", "selection", "iteration",
                       "IF", "CASE", "FOR", "WHILE", "REPEAT", "procedure", "function"),
    },
    "ol_arrays": {
        "content": ("one-dimensional array", "two-dimensional array", "array index",
                    "read values from an array", "write values into an array", "nested iteration"),
        "past_paper": ("array", "one-dimensional", "two-dimensional", "index",
                       "nested iteration", "pseudocode"),
    },
    "ol_file_handling": {
        "content": ("file handling", "storing data in a file", "open a file", "close a file",
                    "file for reading", "file for writing", "read a line", "write a line"),
        "past_paper": ("OPENFILE", "CLOSEFILE", "READFILE", "WRITEFILE",
                       "read from a file", "write to a file", "pseudocode"),
    },
    "ol_database_concepts": {
        "content": ("database", "single-table database", "field", "record",
                    "primary key", "data type", "validation"),
        "past_paper": ("database", "table", "field", "record", "primary key", "data type"),
    },
    "ol_sql": {
        "content": ("SQL", "SELECT", "FROM", "WHERE", "ORDER BY", "SUM", "COUNT"),
        "past_paper": ("SQL", "SELECT", "FROM", "WHERE", "ORDER BY", "SUM", "COUNT"),
    },
    "al_sql": {
        "content": ("DDL", "DML", "SQL", "CREATE TABLE", "PRIMARY KEY", "FOREIGN KEY",
                    "SELECT", "FROM", "WHERE", "ORDER BY", "JOIN", "GROUP BY"),
        "past_paper": ("DDL", "DML", "SQL", "CREATE TABLE", "SELECT", "FROM",
                       "WHERE", "JOIN", "GROUP BY"),
    },
    "al_arrays": {
        "content": ("one-dimensional array", "two-dimensional array", "array", "index"),
        "past_paper": ("array", "one-dimensional", "two-dimensional", "index", "pseudocode"),
    },
    "al_files": {
        "content": ("file access", "text file", "open a file", "close a file",
                    "read from a file", "write to a file"),
        "past_paper": ("OPENFILE", "CLOSEFILE", "READFILE", "WRITEFILE", "file access"),
    },
    "al_file_organisation": {
        "content": ("file organisation", "serial access", "sequential access",
                    "random access", "direct access", "hashing"),
        "past_paper": ("file organisation", "serial file", "sequential file",
                       "random access", "direct access", "hashing"),
    },
    "al_file_processing": {
        "content": ("file processing", "exception handling", "read mode", "write mode",
                    "append mode", "try", "except", "exception"),
        "past_paper": ("file processing", "exception handling", "read mode",
                       "write mode", "append mode", "exception"),
    },
}


SQL_TOPIC_IDS = {"ol_sql", "al_sql"}
STRICT_PROFILE_TOPIC_IDS = {
    "ol_arrays", "ol_file_handling", "al_arrays", "al_files",
    "al_file_organisation", "al_file_processing",
}


def retrieval_profile(topic: dict[str, Any]) -> dict[str, Any]:
    """Return explicit content/style terms, with a conservative generic fallback."""
    configured = TOPIC_RETRIEVAL_PROFILES.get(str(topic["id"]), {})
    name = str(topic["name"])
    acronym = re.findall(r"\(([A-Z][A-Z0-9]{1,8})\)", name)
    # The complete official heading is the safe generic match. Individual
    # words such as "number", "language", "system", and "data" are too broad
    # for question-paper matching and caused cross-topic evidence leakage.
    generic = tuple(dict.fromkeys([name, *acronym]))
    content = list(configured.get("content", generic))
    past_paper = list(configured.get("past_paper", tuple(content) + (str(topic.get("chapter_name", "")),)))
    return {"content": [item for item in content if item],
            "past_paper": [item for item in past_paper if item],
            "strict_phrases": str(topic["id"]) in STRICT_PROFILE_TOPIC_IDS}


@lru_cache(maxsize=2)
def _catalog_chunks(build_path: str) -> tuple[dict[str, Any], ...]:
    all_chunks = Path(build_path) / "chunks" / "all.jsonl"
    with all_chunks.open(encoding="utf-8") as handle:
        return tuple(json.loads(line) for line in handle if line.strip())


def _active_catalog_chunks() -> tuple[dict[str, Any], ...]:
    build = current_build_path()
    return _catalog_chunks(str(build))


def _syllabus_documents(level: str) -> set[str]:
    """Read the active build, proving the selector is backed by ingested data."""
    return {
        str(row["document_id"])
        for row in _active_catalog_chunks()
        if row.get("document_type") == "SYLLABUS" and row.get("level") == level
    }


def _normalised_terms(value: str) -> set[str]:
    """Return meaningful lexical terms for a curriculum heading or objective."""
    ignored = {"and", "the", "for", "with", "its", "use", "uses", "level", "concept", "concepts"}
    return {term.lower() for term in re.findall(r"[A-Za-z]{3,}", value)
            if term.lower() not in ignored}


def _syllabus_evidence_for_topic(level: str, chapter: dict[str, Any], topic: dict[str, Any]) -> list[dict[str, Any]]:
    """Resolve a selected UI topic to its strongest ingested syllabus excerpts.

    The catalogue provides stable UI identifiers, but it must never become an
    unverified substitute for the ingested syllabus.  This lookup retains the
    actual syllabus chunk IDs and short excerpts that led to later textbook
    and past-paper retrieval.  A lexical score is intentional here: it makes
    the selection explainable and avoids an extra paid embedding request.
    """
    topic_terms = _normalised_terms(str(topic["name"]))
    chapter_terms = _normalised_terms(str(chapter["name"]))
    candidates: list[tuple[int, dict[str, Any]]] = []
    for row in _active_catalog_chunks():
        if row.get("document_type") != "SYLLABUS" or row.get("level") != level:
            continue
        text = str(row.get("text", ""))
        row_terms = _normalised_terms(text)
        topic_overlap = len(topic_terms & row_terms)
        chapter_overlap = len(chapter_terms & row_terms)
        if topic_overlap:
            candidates.append((topic_overlap * 5 + chapter_overlap, row))
    # A deterministic tie-breaker keeps the same build reproducible.
    candidates.sort(key=lambda item: (-item[0], str(item[1].get("chunk_id", ""))))
    evidence = []
    needles = sorted(topic_terms, key=len, reverse=True)
    for _, item in candidates[:3]:
        text = str(item.get("text", ""))
        lowered = text.lower()
        positions = [lowered.find(term) for term in needles if lowered.find(term) >= 0]
        position = min(positions) if positions else 0
        start = max(0, position - 140)
        evidence.append({"chunk_id": item["chunk_id"], "document_id": item["document_id"],
                         "page": item.get("page_start"), "excerpt": text[start:start + 700]})
    return evidence


def chapters_for_level(level: str) -> dict[str, Any]:
    """Return printed coursebook chapters only when their syllabus is ingested.

    The syllabus remains the authority for level/scope validation. The visible
    hierarchy and page ranges come from the supplied coursebook contents,
    because teachers navigate by printed book pages rather than PDF indexes.
    """
    normalised = level.upper()
    if normalised not in CATALOG:
        raise ValueError("Level must be O_LEVEL or A_LEVEL.")
    source_documents = sorted(_syllabus_documents(normalised))
    if not source_documents:
        raise RuntimeError(f"No {normalised} syllabus document is available in the active RAG build.")
    return {
        "level": normalised,
        "source_documents": source_documents,
        "page_numbering": "PRINTED_BOOK",
        "page_numbering_note": "Page ranges use the numbers printed inside the supplied coursebook, not PDF viewer indexes.",
        "chapters": public_book_catalog(normalised),
    }


def resolve_topics(level: str, chapter_ids: list[str], topic_ids: list[str]) -> dict[str, Any]:
    """Validate UI choices against the syllabus projection and build scope data."""
    catalog = chapters_for_level(level)
    chapters = {item["id"]: item for item in catalog["chapters"]}
    requested_chapters = list(dict.fromkeys(chapter_ids))
    if not requested_chapters or any(identifier not in chapters for identifier in requested_chapters):
        raise ValueError("Select one or more valid syllabus chapters.")
    allowed_topics = {topic["id"]: {**topic, "chapter_id": chapter["id"], "chapter_name": chapter["name"]}
                      for chapter in chapters.values() if chapter["id"] in requested_chapters for topic in chapter["topics"]}
    requested_topics = list(dict.fromkeys(topic_ids))
    if not requested_topics or any(identifier not in allowed_topics for identifier in requested_topics):
        raise ValueError("Select one or more topics belonging to the selected syllabus chapter(s).")
    selected = []
    # The supplied PDFs have different amounts of front matter. These offsets
    # are used only inside retrieval; the API/UI continues to expose printed
    # book page numbers exclusively.
    pdf_page_offset = 12 if catalog["level"] == "O_LEVEL" else 8
    for identifier in requested_topics:
        topic = allowed_topics[identifier]
        chapter = chapters[topic["chapter_id"]]
        # Retain the selected chapter/topic context and source excerpts.  The
        # assessment workflow uses this evidence to run a separate textbook
        # and past-paper retrieval for each selected topic.
        topic_name = str(topic["name"]).upper()
        code_kind = "SQL" if topic["id"] in SQL_TOPIC_IDS or "QUERY LANGUAGE (SQL)" in topic_name else (
            "PROGRAMMING" if topic["allows_code"] else None
        )
        enriched = {
                    **topic,
                    "code_kind": code_kind,
                    "source_pdf_page_start": int(topic["book_page_start"]) + pdf_page_offset,
                    "source_pdf_page_end": int(topic["book_page_end"]) + pdf_page_offset,
                    "syllabus_evidence": _syllabus_evidence_for_topic(catalog["level"], chapter, topic)}
        selected.append({**enriched, "retrieval_profile": retrieval_profile(enriched)})
    code_kinds = {item["code_kind"] for item in selected if item.get("code_kind")}
    # If teachers combine SQL with a broader programming topic, the more
    # general programming mode is safer for one fixed applied item.
    code_kind = "SQL" if code_kinds == {"SQL"} else ("PROGRAMMING" if code_kinds else None)
    return {"level": catalog["level"], "source_documents": catalog["source_documents"],
            "chapter_ids": requested_chapters, "chapter_names": [chapters[key]["name"] for key in requested_chapters],
            "topics": selected, "topic_ids": requested_topics,
            "allows_code": any(item["allows_code"] for item in selected), "code_kind": code_kind}
