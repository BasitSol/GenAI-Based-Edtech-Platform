"""Offline contracts for the Phase 2.1 assessment workflow and persistence."""
from __future__ import annotations

from types import SimpleNamespace

from backend.shared.persistence import PlatformStore
from backend.module2_generation.assessments.syllabus_catalog import CATALOG, resolve_topics, retrieval_profile
from backend.module2_generation.assessments.book_catalog import BOOK_CATALOG, public_book_catalog
from backend.module2_generation.assessment_engine import (_plan, _source_catalog, _validate, _validate_mock_test_scope,
                                                 _validate_source_distribution, _question_source_assignments,
                                                 _question_structure_assignments, _validate_marking_quality,
                                                 _scoped_query, _normalise_programming_rubric,
                                                 _normalise_atomic_rubric, _topic_chunk_score,
                                                 _question_matches_assigned_topic, _unsupported_advanced_claim,
                                                 _model_blueprint, _apply_system_owned_question_contract,
                                                 _openai_create_with_retry, _openai_json_with_retry,
                                                 _provider_usage)


def test_assessment_plan_reuses_existing_difficulty_taxonomy():
    plan = _plan("binary search", "medium", "quiz", 5)
    assert plan["difficulty"] == "INTERMEDIATE"
    assert plan["question_count"] == 5


def test_assessment_validation_requires_answer_rubric_and_citations():
    plan = _plan("binary search", "easy", "quiz", 1)
    content = {"status": "PENDING_REVIEW", "questions": [{"number": 1, "question": "What is it?", "question_type": "SHORT_ANSWER", "content_source": "TEXTBOOK",
                "options": [], "correct_option": "", "model_answer": "A search.",
                "marks": 1, "rubric": ["states definition"], "citations": [{"chunk_id": "book_1", "source_role": "TEXTBOOK_CONTENT"}]}]}
    assert _validate(content, plan, [{"document_type": "TEXTBOOK"}])["passed"]
    content["questions"][0]["citations"] = []
    assert not _validate(content, plan, [{"document_type": "TEXTBOOK"}])["passed"]


def test_mcq_contract_requires_four_options_and_a_matching_answer():
    plan = _plan("binary search", "easy", "quiz", 1, "mcq")
    item = {"number": 1, "question": "Which condition is required?", "question_type": "MCQ", "content_source": "TEXTBOOK", "options": ["A", "B", "C", "D"],
            "correct_option": "A", "marks": 1, "model_answer": "A", "rubric": ["identifies sorted data"], "citations": [{"chunk_id": "book_1", "source_role": "TEXTBOOK_CONTENT"}]}
    assert _validate({"status": "PENDING_REVIEW", "questions": [item]}, plan, [{}])["passed"]
    item["options"] = ["A", "B"]
    assert not _validate({"status": "PENDING_REVIEW", "questions": [item]}, plan, [{}])["passed"]


def test_assessment_catalog_marks_source_roles_explicitly():
    catalog, _ = _source_catalog([
        {"chunk_id": "book", "document_type": "TEXTBOOK", "assessment_source_role": "TEXTBOOK_CONTENT"},
        {"chunk_id": "paper", "document_type": "QUESTION_PAPER", "assessment_source_role": "PAST_PAPER_STYLE"},
    ])
    assert catalog[0]["source_role"] == "TEXTBOOK_CONTENT"
    assert catalog[1]["source_role"] == "PAST_PAPER_STYLE"


def test_source_distribution_requires_balanced_paired_past_paper_evidence():
    plan = _plan("binary search", "easy", "quiz", 3, "mcq")
    questions = [
        {"content_source": "TEXTBOOK", "citations": [{"source_role": "TEXTBOOK_CONTENT"}]},
        {"content_source": "TEXTBOOK", "citations": [{"source_role": "TEXTBOOK_CONTENT"}]},
        {"content_source": "PAST_PAPER", "citations": [{"source_role": "PAST_PAPER_STYLE"}, {"source_role": "TEXTBOOK_CONTENT"}]},
    ]
    assert _validate_source_distribution({"questions": questions}, plan)["passed"]
    questions[2]["citations"] = [{"source_role": "PAST_PAPER_STYLE"}]
    assert not _validate_source_distribution({"questions": questions}, plan)["passed"]


def test_question_source_assignments_preallocate_balanced_paired_evidence():
    plan = _plan("binary search", "medium", "mock_test", 8, "mixed", total_marks=25)
    catalog = [
        {"source_key": "T1", "source_role": "TEXTBOOK_CONTENT"},
        {"source_key": "P1", "source_role": "PAST_PAPER_STYLE"},
    ]
    assignments = _question_source_assignments(plan, catalog)
    assert len(assignments) == 8
    assert sum(item["content_source"] == "TEXTBOOK" for item in assignments) == 4
    past_assignment = next(item for item in assignments if item["content_source"] == "PAST_PAPER")
    assert set(past_assignment["required_source_keys"]) == {"T1", "P1"}


def test_system_owns_question_provenance_instead_of_trusting_model_metadata():
    question = {"number": 2, "question": "State the purpose of a primary key.",
                "question_type": "MCQ", "marks": 6, "options": ["wrong"],
                "correct_option": "wrong", "model_answer": "It identifies a record.",
                "rubric": ["Correct purpose."],
                # Deliberately hostile/stale model metadata must be ignored.
                "content_source": "TEXTBOOK", "topic_ids": ["wrong"], "source_keys": ["UNKNOWN"]}
    assignment = {"number": 2, "content_source": "PAST_PAPER",
                  "topic_ids": ["ol_database_concepts"],
                  "required_source_keys": ["P1", "T1"]}
    structure = {"number": 2, "question_type": "SHORT_ANSWER", "marks": 2,
                 "requires_coding": False, "coding_kind": None}
    result = _apply_system_owned_question_contract(
        question, assignment, structure, {"P1": {}, "T1": {}})
    assert result == {"passed": True, "keys": ["P1", "T1"]}
    assert question["content_source"] == "PAST_PAPER"
    assert question["topic_ids"] == ["ol_database_concepts"]
    assert question["question_type"] == "SHORT_ANSWER"
    assert question["marks"] == 2
    assert question["options"] == [] and question["correct_option"] == ""


def test_mock_source_assignments_keep_each_question_inside_its_selected_topic():
    topics = [{"id": "search", "name": "Searching", "allows_code": True},
              {"id": "sort", "name": "Sorting", "allows_code": True}]
    plan = _plan("Searching; Sorting", "medium", "mock_test", 8, "mixed", topics, 25, allows_code=True)
    catalog = [
        {"source_key": "TS", "source_role": "TEXTBOOK_CONTENT", "topic_ids": ["search"]},
        {"source_key": "PS", "source_role": "PAST_PAPER_STYLE", "topic_ids": ["search"]},
        {"source_key": "TT", "source_role": "TEXTBOOK_CONTENT", "topic_ids": ["sort"]},
        {"source_key": "PT", "source_role": "PAST_PAPER_STYLE", "topic_ids": ["sort"]},
    ]
    assignments = _question_source_assignments(plan, catalog)
    assert len(assignments) == 8
    assert assignments[0]["topic_ids"] == ["search"]
    assert assignments[1]["topic_ids"] == ["sort"]
    assert assignments[2]["topic_ids"] == ["sort"]
    assert assignments[3]["topic_ids"] == ["search"]
    for assignment in assignments:
        for key in assignment["required_source_keys"]:
            source = next(item for item in catalog if item["source_key"] == key)
            assert source["topic_ids"] == assignment["topic_ids"]


def test_scoped_query_contains_chapter_topic_and_syllabus_objective_excerpt():
    topic = {"name": "Searching and sorting", "chapter_name": "Algorithm design",
             "retrieval_profile": {"content": ["linear search", "bubble sort"]},
             "syllabus_evidence": [{"chunk_id": "syllabus_1", "excerpt": "Use search and sort algorithms."}]}
    query = _scoped_query(topic, "Teach and assess this curriculum content")
    assert "Algorithm design" in query
    assert "Searching and sorting" in query
    assert "linear search" in query and "bubble sort" in query


def test_topic_evidence_scoring_rejects_contents_and_unrelated_compiler_question():
    topic = {"name": "Searching and sorting", "retrieval_profile": {
        "content": ["linear search", "bubble sort"],
        "past_paper": ["linear search", "bubble sort", "algorithm", "pseudocode"],
    }}
    contents = {"document_type": "TEXTBOOK", "page_start": 4, "text": "Contents: searching and sorting"}
    textbook = {"document_type": "TEXTBOOK", "page_start": 288, "content_type": "EXPLANATION",
                "text": "The bubble sort compares adjacent values and swaps them."}
    compiler = {"document_type": "QUESTION_PAPER", "page_start": 6,
                "text": "Describe how a compiler translates a high-level program."}
    algorithm_paper = {"document_type": "QUESTION_PAPER", "page_start": 6,
                       "text": "Identify errors in this pseudocode algorithm."}
    assert _topic_chunk_score(topic, contents, "TEXTBOOK") < 0
    assert _topic_chunk_score(topic, textbook, "TEXTBOOK") > 0
    assert _topic_chunk_score(topic, compiler, "QUESTION_PAPER") < 0
    assert _topic_chunk_score(topic, algorithm_paper, "QUESTION_PAPER") > 0


def test_generic_topic_heading_does_not_accept_single_word_paper_overlap():
    topic = {"name": "Number systems", "retrieval_profile": {
        "content": ["Number systems"], "past_paper": ["Number systems"],
    }}
    unrelated = {"document_type": "QUESTION_PAPER", "page_start": 8,
                 "text": "A database contains a field named SongNumber. Identify its data type."}
    exact = {"document_type": "QUESTION_PAPER", "page_start": 8,
             "text": "Explain how binary and hexadecimal number systems are related."}
    assert _topic_chunk_score(topic, unrelated, "QUESTION_PAPER") < 0
    assert _topic_chunk_score(topic, exact, "QUESTION_PAPER") > 0


def test_source_assignment_uses_global_style_anchor_when_topic_has_no_paper():
    topics = [{"id": "recursion", "name": "Concept of recursion", "allows_code": True}]
    plan = _plan("Concept of recursion", "medium", "mock_test", 8, "mixed",
                 topics, 25, allows_code=True)
    catalog = [
        {"source_key": "T1", "source_role": "TEXTBOOK_CONTENT", "topic_ids": ["recursion"]},
        {"source_key": "P1", "source_role": "PAST_PAPER_STYLE", "topic_ids": []},
    ]
    assignments = _question_source_assignments(plan, catalog)
    assert len(assignments) == 8
    assert all("T1" in item["required_source_keys"] for item in assignments)
    assert all("P1" in item["required_source_keys"]
               for item in assignments if item["content_source"] == "PAST_PAPER")


def test_assessment_provider_retry_is_bounded_and_transient_only(monkeypatch):
    calls = []

    class APIConnectionError(Exception):
        pass

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            if len(calls) < 3:
                raise APIConnectionError("temporary")
            return "ok"

    client = type("Client", (), {"chat": type("Chat", (), {"completions": Completions()})()})()
    monkeypatch.setenv("ASSESSMENT_PROVIDER_MAX_ATTEMPTS", "3")
    monkeypatch.setenv("ASSESSMENT_RETRY_BASE_SECONDS", "0")
    result, attempts = _openai_create_with_retry(client, operation="test", model="fake")
    assert result == "ok" and attempts == 3 and len(calls) == 3


def test_assessment_provider_retry_handles_status_codes_and_bad_environment(monkeypatch):
    calls = []

    class APIStatusError(Exception):
        status_code = 503

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise APIStatusError("temporary")
            return "ok"

    client = type("Client", (), {"chat": type("Chat", (), {"completions": Completions()})()})()
    monkeypatch.setenv("ASSESSMENT_PROVIDER_MAX_ATTEMPTS", "not-an-integer")
    monkeypatch.setenv("ASSESSMENT_RETRY_BASE_SECONDS", "not-a-number")
    result, attempts = _openai_create_with_retry(client, operation="test", model="fake")
    assert result == "ok" and attempts == 2 and len(calls) == 2


def test_structured_response_retry_recovers_from_empty_json_without_regenerating(monkeypatch):
    bodies = ["", '{"passed": true}']

    class Completions:
        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content=bodies.pop(0)),
                )]
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    monkeypatch.setenv("ASSESSMENT_PROVIDER_MAX_ATTEMPTS", "1")
    monkeypatch.setenv("ASSESSMENT_STRUCTURED_RESPONSE_ATTEMPTS", "2")
    _, payload, attempts = _openai_json_with_retry(
        client,
        operation="semantic_judge",
        model="fake",
    )
    assert payload == {"passed": True}
    assert attempts == 2
    assert bodies == []


def test_provider_usage_populates_token_and_cost_telemetry(monkeypatch):
    monkeypatch.setenv("GENERATOR_INPUT_USD_PER_MILLION", "0.40")
    monkeypatch.setenv("GENERATOR_OUTPUT_USD_PER_MILLION", "1.60")
    response = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=1000, completion_tokens=500)
    )
    assert _provider_usage(response) == {
        "input_tokens": 1000,
        "output_tokens": 500,
        "total_cost": 0.0012,
    }


def test_topic_evidence_scoring_rejects_textbook_front_matter_but_not_question_paper_page_six():
    topic = {"name": "Database concepts", "retrieval_profile": {
        "content": ["database"], "past_paper": ["database"],
    }}
    textbook_contents = {"document_type": "TEXTBOOK", "page_start": 6,
                         "text": "Section 9 Databases and database concepts"}
    paper_question = {"document_type": "QUESTION_PAPER", "page_start": 6,
                      "text": "A database stores customer records. State a suitable primary key."}
    assert _topic_chunk_score(topic, textbook_contents, "TEXTBOOK") < 0
    assert _topic_chunk_score(topic, paper_question, "QUESTION_PAPER") > 0


def test_catalog_uses_supplied_syllabus_programming_sections_without_merged_arrays_files_label():
    o_level = {chapter["id"]: chapter for chapter in CATALOG["O_LEVEL"]}
    programming = o_level["ol_programming"]
    assert programming["name"] == "Programming"
    assert [(topic_id, name) for topic_id, name, _ in programming["topics"]] == [
        ("ol_programming_constructs", "Programming concepts"),
        ("ol_arrays", "Arrays"),
        ("ol_file_handling", "File handling"),
    ]
    all_o_level_labels = {name for chapter in CATALOG["O_LEVEL"] for _, name, _ in chapter["topics"]}
    assert "Arrays and files" not in all_o_level_labels


def test_a_level_catalog_matches_as_and_a_level_content_overview_sections():
    a_level = {chapter["id"]: chapter for chapter in CATALOG["A_LEVEL"]}
    assert [name for _, name, _ in a_level["al_data_types_structures"]["topics"]] == [
        "Data Types and Records", "Arrays", "Files", "Introduction to Abstract Data Types (ADT)"]
    assert [name for _, name, _ in a_level["al_further_programming"]["topics"]] == [
        "Programming Paradigms", "File Processing and Exception Handling"]
    assert any(chapter["name"] == "Artificial Intelligence (AI)" for chapter in CATALOG["A_LEVEL"])


def test_coursebook_catalog_uses_printed_pages_and_exact_o_level_contents():
    chapters = public_book_catalog("O_LEVEL")
    assert len(chapters) == 10
    algorithms = next(item for item in chapters if item["id"] == "ol_algorithms")
    assert algorithms["name"] == "Algorithm design and problem solving"
    assert algorithms["book_page_label"] == "Book pp. 258–298"
    assert [(item["section_number"], item["name"], item["book_page_start"], item["book_page_end"])
            for item in algorithms["topics"][:2]] == [
                ("7.1", "The program development life cycle", 258, 259),
                ("7.2", "Computer systems, sub-systems and decomposition", 260, 270),
            ]
    programming = next(item for item in chapters if item["id"] == "ol_programming")
    assert programming["topics"][-1]["name"] == "File handling"
    assert programming["topics"][-1]["book_page_label"] == "Book pp. 333–338"


def test_coursebook_catalog_covers_all_a_level_chapters_without_invalid_ranges():
    chapters = BOOK_CATALOG["A_LEVEL"]
    assert len(chapters) == 30
    assert chapters[0]["name"] == "Information representation"
    assert chapters[-1]["name"] == "Software development"
    assert (chapters[-1]["book_page_start"], chapters[-1]["book_page_end"]) == (420, 429)
    assert all(topic["book_page_start"] <= topic["book_page_end"]
               for chapter in chapters for topic in chapter["topics"])


def test_file_handling_scope_rejects_compression_and_accepts_program_file_operations():
    profile = retrieval_profile({"id": "ol_file_handling", "name": "File handling",
                                 "chapter_name": "Programming"})
    topic = {"id": "ol_file_handling", "name": "File handling", "retrieval_profile": profile}
    compression = {"document_type": "TEXTBOOK", "page_start": 47, "content_type": "EXPLANATION",
                   "text": "Lossy file compression permanently removes image data to reduce file size."}
    handling = {"document_type": "TEXTBOOK", "page_start": 343, "content_type": "EXPLANATION",
                "text": "File handling requires a program to open a file, read a line and close a file."}
    assert _topic_chunk_score(topic, compression, "TEXTBOOK") < 0
    assert _topic_chunk_score(topic, handling, "TEXTBOOK") > 0


def test_selected_coursebook_section_cannot_retrieve_textbook_pages_outside_its_range():
    topic = {
        "id": "ol_file_handling",
        "name": "File handling",
        "source_pdf_page_start": 345,
        "source_pdf_page_end": 350,
        "retrieval_profile": retrieval_profile({
            "id": "ol_file_handling", "name": "File handling", "chapter_name": "Programming",
        }),
    }
    in_scope = {"document_type": "TEXTBOOK", "page_start": 345, "content_type": "EXPLANATION",
                "text": "File handling opens a file for reading and then closes the file."}
    out_of_scope = {**in_scope, "page_start": 90}
    assert _topic_chunk_score(topic, in_scope, "TEXTBOOK") > 0
    assert _topic_chunk_score(topic, out_of_scope, "TEXTBOOK") < 0


def test_question_topic_alignment_rejects_compiler_drift_and_unsupported_big_o():
    plan = {"selected_topics": [{"id": "search", "name": "Searching", "retrieval_profile": {
        "content": ["linear search", "binary search", "bubble sort"]}}]}
    compiler = {"topic_ids": ["search"], "question": "Explain the role of a compiler.",
                "model_answer": "It translates a program."}
    searching = {"topic_ids": ["search"], "question": "Explain linear search.",
                 "model_answer": "Linear search checks each item."}
    assert not _question_matches_assigned_topic(compiler, plan)
    assert _question_matches_assigned_topic(searching, plan)
    complexity = {"question": "Explain bubble sort.", "model_answer": "It has O(n²) time complexity."}
    assert _unsupported_advanced_claim(complexity, "Bubble sort swaps adjacent items.") == "BIG_O_NOT_IN_SOURCE"


def test_topic_alignment_accepts_database_vocabulary_derived_from_assigned_evidence():
    plan = {"selected_topics": [{"id": "database", "name": "Database concepts", "retrieval_profile": {
        "content": ["Database concepts", "Database"]}}]}
    question = {"topic_ids": ["database"], "question": "State the purpose of a primary key.",
                "model_answer": "A primary key uniquely identifies each record in a table."}
    evidence = "A database table contains fields and records. A primary key uniquely identifies a record."
    assert _question_matches_assigned_topic(question, plan, evidence)


def test_topic_alignment_accepts_one_distinctive_term_from_assigned_evidence():
    plan = {"selected_topics": [{"id": "database", "name": "Database concepts", "retrieval_profile": {
        "content": ["Database concepts", "Database"]}}]}
    question = {"topic_ids": ["database"], "question": "State a suitable data type for DateOfBirth.",
                "model_answer": "Date/time."}
    evidence = "Suitable field types include integer, real, Boolean, character, and date/time."
    assert _question_matches_assigned_topic(question, plan, evidence)


def test_model_blueprint_excludes_raw_syllabus_excerpts_but_keeps_retrieval_profile():
    plan = {"topic": "Databases", "selected_topics": [{
        "id": "ol_sql", "name": "Structured Query Language (SQL)",
        "chapter_id": "ol_databases", "chapter_name": "Databases",
        "allows_code": True, "code_kind": "SQL",
        "retrieval_profile": {"content": ["SQL"], "past_paper": ["SQL"]},
        "syllabus_evidence": [{"excerpt": "unrelated adjacent parser text"}],
    }]}
    compact = _model_blueprint(plan)
    assert compact["selected_topics"][0]["retrieval_profile"]["content"] == ["SQL"]
    assert "syllabus_evidence" not in compact["selected_topics"][0]


def test_mock_question_structure_is_eight_questions_and_exactly_25_marks():
    plan = _plan("binary search", "medium", "mock_test", 8, "mixed", total_marks=25, allows_code=True)
    structure = _question_structure_assignments(plan)
    assert len(structure) == 8
    assert sum(item["marks"] for item in structure) == 25
    assert structure[0]["question_type"] == "MCQ" and structure[0]["marks"] == 1
    assert structure[-1]["requires_coding"]


def test_marking_validator_does_not_misclassify_algorithm_theory_as_code():
    theory = {"number": 2, "question_type": "SHORT_ANSWER", "marks": 2,
              "question": "Explain why an algorithm must terminate.", "model_answer": "It must stop.",
              "rubric": ["states that it stops", "links to a result"]}
    assert _validate_marking_quality({"questions": [theory]}, {"assessment_type": "mock_test", "allows_code": True,
                                                                   "question_count": 8})["passed"]
    code = {**theory, "number": 8, "question": "Write pseudocode to search a list.", "marks": 4}
    assert not _validate_marking_quality({"questions": [code]}, {"assessment_type": "mock_test", "allows_code": True,
                                                                    "question_count": 8})["passed"]
    code["rubric"] = ["1 mark for correct binary-search algorithm logic.",
                      "1 mark for a midpoint comparison condition.",
                      "1 mark for updating bounds each search iteration.",
                      "1 mark for outputting the result or termination condition."]
    assert _validate_marking_quality({"questions": [code]}, {"assessment_type": "mock_test", "allows_code": True,
                                                               "question_count": 8})["passed"]


def test_programming_rubric_normalisation_repairs_generic_model_rubric_without_changing_task():
    question = {"question": "Write pseudocode for a binary search.", "rubric": ["Award marks for a valid answer."],
                "model_answer": "Use a loop and compare the middle value."}
    _normalise_programming_rubric(question, {"requires_coding": True, "marks": 4})
    assert question["question"] == "Write pseudocode for a binary search."
    assert len(question["rubric"]) >= 4
    validated = {"number": 8, "question_type": "LONG_ANSWER", "marks": 4, **question}
    assert _validate_marking_quality({"questions": [validated]}, {"assessment_type": "mock_test", "allows_code": True,
                                                                     "question_count": 8})["passed"]


def test_atomic_rubric_compiler_produces_one_criterion_per_mark_for_structured_items():
    question = {"number": 5, "question_type": "LONG_ANSWER", "marks": 4,
                "question": "Explain the stages of a search.",
                "model_answer": "Set the bounds. Calculate the midpoint. Compare the item. Update the correct bound.",
                "rubric": ["Award marks for correct bounds and midpoint; comparison and update."]}
    _normalise_atomic_rubric(question, {"marks": 4, "requires_coding": False})
    assert len(question["rubric"]) == 4
    assert all(item.lower().startswith("1 mark for") for item in question["rubric"])
    assert _validate_marking_quality({"questions": [question]})["passed"]


def test_atomic_rubric_compiler_forces_one_mcq_marking_point():
    question = {"question_type": "MCQ", "marks": 1, "correct_option": "B. Binary search",
                "model_answer": "Binary search", "rubric": ["Correct option", "Explanation"]}
    _normalise_atomic_rubric(question, {"marks": 1})
    assert question["rubric"] == ["1 mark for selecting B. Binary search."]


def test_mcq_with_applied_subject_vocabulary_is_not_treated_as_written_code():
    """Response format, not topic vocabulary, determines rubric validation."""
    question = {
        "number": 4,
        "question_type": "mcq",
        "marks": 1,
        "question": "Which trace table shows the result of this binary-search algorithm?",
        "model_answer": "Option C correctly shows the updated bounds.",
        "rubric": ["1 mark for selecting option C."],
        "options": ["A", "B", "C", "D"],
        "correct_option": "C",
    }
    result = _validate_marking_quality(
        {"questions": [question]},
        {"assessment_type": "quiz", "question_format": "MCQ", "allows_code": True},
    )
    assert result["passed"]


def test_explanation_about_sql_query_is_not_treated_as_query_construction():
    question = {
        "number": 2,
        "question_type": "SHORT_ANSWER",
        "marks": 2,
        "question": "Explain the purpose of a WHERE clause in an SQL query.",
        "model_answer": "It filters records and returns only rows that satisfy the condition.",
        "rubric": [
            "1 mark for stating that it filters records.",
            "1 mark for explaining that only matching rows are returned.",
        ],
    }
    assert _validate_marking_quality(
        {"questions": [question]},
        {"assessment_type": "quiz", "allows_code": True},
    )["passed"]


def test_true_applied_query_still_rejects_a_generic_rubric():
    question = {
        "number": 2,
        "question_type": "LONG_ANSWER",
        "marks": 1,
        "question": "Write an SQL query to display all student names.",
        "model_answer": "SELECT Name FROM Student;",
        "rubric": ["1 mark for a correct answer."],
    }
    result = _validate_marking_quality(
        {"questions": [question]},
        {"assessment_type": "quiz", "allows_code": True},
    )
    assert result["reason"] == "APPLIED_ITEM_HAS_GENERIC_RUBRIC"


def test_sql_applied_item_uses_query_specific_four_mark_rubric():
    question = {"number": 8, "question_type": "LONG_ANSWER", "marks": 4,
                "question": "Write an SQL query to select names from Student where Mark > 70.",
                "model_answer": "SELECT Name FROM Student WHERE Mark > 70;",
                "rubric": ["Award marks for a correct query."]}
    structure = {"requires_coding": True, "coding_kind": "SQL", "marks": 4}
    _normalise_atomic_rubric(question, structure)
    _normalise_programming_rubric(question, structure)
    assert question["rubric_normalisation"] == "sql-four-dimensions-v1"
    assert _validate_marking_quality({"questions": [question]}, {
        "assessment_type": "mock_test", "allows_code": True, "code_kind": "SQL", "question_count": 8})["passed"]


def test_mock_sql_explanation_is_not_misclassified_as_second_coding_item():
    question = {"number": 4, "question_type": "SHORT_ANSWER", "marks": 2,
                "question": "Explain what the WHERE clause does in an SQL query.",
                "model_answer": "It filters records so only matching rows are returned.",
                "rubric": ["1 mark for stating that records are filtered.",
                           "1 mark for explaining that only matching rows are returned."]}
    plan = {"assessment_type": "mock_test", "allows_code": True,
            "code_kind": "SQL", "question_count": 8}
    assert _validate_marking_quality({"questions": [question]}, plan)["passed"]


def test_mock_test_contract_enforces_25_marks_selected_topics_and_theory_scope():
    selected = [{"id": "ol_number_systems", "name": "Number systems", "allows_code": False}]
    plan = _plan("Number systems", "medium", "mock_test", 8, "mixed", selected, 25, allows_code=False)
    marks = [1, 2, 3, 4, 4, 3, 4, 4]
    questions = [{"number": index, "marks": mark, "topic_ids": ["ol_number_systems"],
                  "question_type": "MCQ" if index == 1 else "SHORT_ANSWER",
                  "question": "Explain a binary number.", "model_answer": "A binary number uses base two."}
                 for index, mark in enumerate(marks, 1)]
    assert _validate_mock_test_scope({"questions": questions}, plan)["passed"]
    questions[0]["question"] = "Write pseudocode for binary conversion."
    assert not _validate_mock_test_scope({"questions": questions}, plan)["passed"]


def test_theory_scope_does_not_misclassify_select_from_or_explanatory_code_terms():
    selected = [{"id": "ol_data_storage", "name": "Data storage and file compression",
                 "allows_code": False}]
    plan = _plan("Data storage and file compression", "medium", "mock_test", 8, "mixed",
                 selected, 25, allows_code=False)
    marks = [1, 2, 3, 3, 4, 4, 4, 4]
    questions = [{
        "number": index,
        "marks": mark,
        "topic_ids": ["ol_data_storage"],
        "question_type": "MCQ" if index == 1 else "SHORT_ANSWER",
        "question": "Select the correct explanation of how data is removed from an image during lossy compression."
                    if index == 1 else "Explain how run-length encoding compresses repeated data.",
        "model_answer": "The encoder selects repeated data from the source and stores a value and repetition count.",
    } for index, mark in enumerate(marks, 1)]
    result = _validate_mock_test_scope({"questions": questions}, plan)
    assert result["passed"], result


def test_mock_test_topic_resolution_rejects_topic_outside_selected_chapter(monkeypatch):
    monkeypatch.setattr("backend.module2_generation.assessments.syllabus_catalog._syllabus_documents", lambda level: {"syllabus"})
    monkeypatch.setattr("backend.module2_generation.assessments.syllabus_catalog._syllabus_evidence_for_topic", lambda *args: [])
    scope = resolve_topics("O_LEVEL", ["ol_data_representation"], ["ol_number_systems"])
    assert scope["allows_code"] is False
    assert scope["topics"][0]["book_page_start"] == 2
    assert scope["topics"][0]["source_pdf_page_start"] == 14
    try:
        resolve_topics("O_LEVEL", ["ol_data_representation"], ["ol_sql"])
    except ValueError as exc:
        assert "selected syllabus chapter" in str(exc)
    else:
        raise AssertionError("A topic from an unselected chapter must be rejected.")


def test_assessment_approval_is_scoped_to_owning_teacher(tmp_path):
    store = PlatformStore(tmp_path / "platform.sqlite")
    teacher = store.create_user("teacher@example.com", "hash", "teacher")
    other_teacher = store.create_user("other@example.com", "hash", "teacher")
    assessment = store.create_assessment(teacher["id"], "SQL", "BEGINNER", "quiz", {"questions": []})
    assert store.approve_assessment(assessment["id"], other_teacher["id"]) is None
    assert store.approve_assessment(assessment["id"], teacher["id"])["status"] == "approved"


def test_teacher_can_delete_only_unapproved_empty_draft(tmp_path):
    store = PlatformStore(tmp_path / "platform.sqlite")
    teacher = store.create_user("teacher@example.com", "hash", "teacher")
    assessment = store.create_assessment(teacher["id"], "SQL", "BEGINNER", "quiz", {"questions": []})
    assert store.delete_draft_assessment(assessment["id"], teacher["id"])
