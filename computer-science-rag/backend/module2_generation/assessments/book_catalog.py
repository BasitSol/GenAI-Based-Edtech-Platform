"""Authoritative coursebook contents used by the mock-test scope selector.

The page numbers in this module are the numbers printed in the supplied
coursebooks. They are deliberately not PDF viewer indexes: both PDFs contain
front matter before printed page 1.
"""
from __future__ import annotations

import re
from typing import Any, Iterable


def _slug(value: str) -> str:
    """Create a stable identifier for a printed coursebook section."""
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", value.lower())).strip("_")


def _chapter(*, level_prefix: str, number: int, name: str, start: int, end: int,
             sections: Iterable[tuple[str, str, int] | tuple[str, str, int, bool]],
             chapter_id: str | None = None) -> dict[str, Any]:
    """Build one chapter and calculate inclusive printed section ranges."""
    raw = list(sections)
    topics: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        section_number, section_name, section_start, *code_flag = item
        next_start = raw[index + 1][2] if index + 1 < len(raw) else end + 1
        topics.append({
            "id": f"{level_prefix}_{section_number.replace('.', '_')}_{_slug(section_name)}",
            "section_number": section_number,
            "name": section_name,
            "book_page_start": section_start,
            "book_page_end": max(section_start, next_start - 1),
            "allows_code": bool(code_flag[0]) if code_flag else False,
        })
    return {
        "id": chapter_id or f"{level_prefix}_chapter_{number:02d}_{_slug(name)}",
        "chapter_number": number,
        "name": name,
        "book_page_start": start,
        "book_page_end": end,
        "page_numbering": "PRINTED_BOOK",
        "topics": topics,
    }


O_LEVEL_BOOK: tuple[dict[str, Any], ...] = (
    _chapter(level_prefix="ol", number=1, name="Data representation", start=2, end=44,
             chapter_id="ol_data_representation", sections=(
                 ("1.1", "Number systems", 2), ("1.2", "Text, sound and images", 25),
                 ("1.3", "Data storage and file compression", 32))),
    _chapter(level_prefix="ol", number=2, name="Data transmission", start=45, end=74,
             chapter_id="ol_data_transmission", sections=(
                 ("2.1", "Types and methods of data transmission", 45),
                 ("2.2", "Methods of error detection", 54),
                 ("2.3", "Symmetric and asymmetric encryption", 63))),
    _chapter(level_prefix="ol", number=3, name="Hardware", start=75, end=146,
             chapter_id="ol_hardware", sections=(
                 ("3.1", "Computer architecture", 75), ("3.2", "Input and output devices", 88),
                 ("3.3", "Data storage", 119), ("3.4", "Network hardware", 133))),
    _chapter(level_prefix="ol", number=4, name="Software", start=147, end=179,
             chapter_id="ol_software", sections=(
                 ("4.1", "Types of software and interrupts", 147),
                 ("4.2", "Types of programming language, translators and integrated development environments (IDEs)", 165))),
    _chapter(level_prefix="ol", number=5, name="The internet and its uses", start=180, end=216,
             chapter_id="ol_internet", sections=(
                 ("5.1", "The internet and the World Wide Web (WWW)", 180),
                 ("5.2", "Digital currency", 186), ("5.3", "Cyber security", 189))),
    _chapter(level_prefix="ol", number=6, name="Automated and emerging technologies", start=217, end=256,
             chapter_id="ol_automated_tech", sections=(
                 ("6.1", "Automated systems", 217), ("6.2", "Robotics", 230),
                 ("6.3", "Artificial intelligence (AI)", 241))),
    _chapter(level_prefix="ol", number=7, name="Algorithm design and problem solving", start=258, end=298,
             chapter_id="ol_algorithms", sections=(
                 ("7.1", "The program development life cycle", 258, True),
                 ("7.2", "Computer systems, sub-systems and decomposition", 260, True),
                 ("7.3", "Explaining the purpose of an algorithm", 271, True),
                 ("7.4", "Standard methods of solution", 272, True),
                 ("7.5", "Validation and verification", 276, True),
                 ("7.6", "Test data", 281, True),
                 ("7.7", "Trace tables to document dry runs of algorithms", 282, True),
                 ("7.8", "Identifying errors in algorithms", 285, True),
                 ("7.9", "Writing and amending algorithms", 288, True))),
    _chapter(level_prefix="ol", number=8, name="Programming", start=299, end=338,
             chapter_id="ol_programming", sections=(
                 ("8.1", "Programming concepts", 302, True), ("8.2", "Arrays", 329, True),
                 ("8.3", "File handling", 333, True))),
    _chapter(level_prefix="ol", number=9, name="Databases", start=339, end=355,
             chapter_id="ol_databases", sections=(("9.1", "Databases", 339),)),
    _chapter(level_prefix="ol", number=10, name="Boolean logic", start=356, end=386,
             chapter_id="ol_boolean_logic", sections=(
                 ("10.1", "Standard logic gate symbols", 356),
                 ("10.2", "The function of the six logic gates", 358),
                 ("10.3", "Logic circuits, logic expressions, truth tables and problem statements", 360))),
)

# Preserve Phase 2's established O Level topic identifiers so saved requests,
# tests, and retrieval profiles remain compatible while labels/page metadata
# become coursebook-accurate.
_O_LEVEL_TOPIC_IDS = {
    "1.1": "ol_number_systems", "1.2": "ol_text_sound_images", "1.3": "ol_data_storage",
    "2.1": "ol_data_transmission_methods", "2.2": "ol_error_detection", "2.3": "ol_encryption",
    "3.1": "ol_computer_architecture", "3.2": "ol_input_output",
    "3.3": "ol_data_storage_hardware", "3.4": "ol_network_hardware",
    "4.1": "ol_system_software", "4.2": "ol_languages_translators",
    "5.1": "ol_internet_networks", "5.2": "ol_digital_currency", "5.3": "ol_cyber_security",
    "6.1": "ol_automated_systems", "6.2": "ol_robotics", "6.3": "ol_artificial_intelligence",
    "7.1": "ol_program_development", "7.2": "ol_decomposition_design",
    "7.3": "ol_algorithm_purpose", "7.4": "ol_searching_sorting",
    "7.5": "ol_programming_validation", "7.6": "ol_test_data", "7.7": "ol_trace_tables",
    "7.8": "ol_algorithm_errors", "7.9": "ol_algorithm_design",
    "8.1": "ol_programming_constructs", "8.2": "ol_arrays", "8.3": "ol_file_handling",
    "9.1": "ol_database_concepts",
    "10.1": "ol_logic_gate_symbols", "10.2": "ol_logic_gate_functions", "10.3": "ol_logic_gates",
}
for _coursebook_chapter in O_LEVEL_BOOK:
    for _coursebook_topic in _coursebook_chapter["topics"]:
        _coursebook_topic["id"] = _O_LEVEL_TOPIC_IDS[_coursebook_topic["section_number"]]


A_LEVEL_BOOK: tuple[dict[str, Any], ...] = (
    _chapter(level_prefix="al", number=1, name="Information representation", start=2, end=17, sections=(
        ("1.01", "Number systems", 2), ("1.02", "Internal coding of numbers", 4),
        ("1.03", "Internal coding of text", 8), ("1.04", "Images", 9),
        ("1.05", "Sound", 12), ("1.06", "Video", 13),
        ("1.07", "Compression techniques and packaging of multimedia content", 14))),
    _chapter(level_prefix="al", number=2, name="Communication and Internet technologies", start=18, end=35, sections=(
        ("2.01", "Transmission media", 18), ("2.02", "The Internet", 20),
        ("2.03", "The World Wide Web (WWW)", 21), ("2.04", "Internet-supporting hardware", 21),
        ("2.05", "Client-server architecture", 22), ("2.06", "Bit streaming", 23),
        ("2.07", "IP addressing", 25), ("2.08", "Domain names", 29),
        ("2.09", "Scripting and HTML in a client-server application", 30, True))),
    _chapter(level_prefix="al", number=3, name="Hardware", start=36, end=48, sections=(
        ("3.01", "The memory system", 36), ("3.02", "Memory components", 37),
        ("3.03", "Secondary storage devices", 38), ("3.04", "Computer graphics", 40),
        ("3.05", "Screens and associated technologies", 41), ("3.06", "Keyboards and keypads", 43),
        ("3.07", "Printers, scanners and plotters", 44), ("3.08", "Input and output of sound", 46))),
    _chapter(level_prefix="al", number=4, name="Logic gates and logic circuits", start=49, end=58, sections=(
        ("4.01", "Boolean logic and problem statements", 49), ("4.02", "Boolean operators", 49),
        ("4.03", "Truth tables", 50), ("4.04", "Logic circuits and logic gates", 51),
        ("4.05", "Alternative circuits", 55))),
    _chapter(level_prefix="al", number=5, name="Processor fundamentals", start=59, end=68, sections=(
        ("5.01", "The Von Neumann model of a computer system", 59),
        ("5.02", "Central processing unit (CPU) architecture", 59),
        ("5.03", "The system bus", 62), ("5.04", "The fetch-execute cycle", 64),
        ("5.05", "Register transfer notation", 65), ("5.06", "Interrupt handling", 65))),
    _chapter(level_prefix="al", number=6, name="Assembly language programming", start=69, end=77, sections=(
        ("6.01", "Machine code instructions", 69, True), ("6.02", "Assembly language", 70, True),
        ("6.03", "Addressing modes", 70, True), ("6.04", "Assembly language instructions", 71, True))),
    _chapter(level_prefix="al", number=7, name="System software", start=78, end=87, sections=(
        ("7.01", "System software", 78), ("7.02", "Operating system activities", 78),
        ("7.03", "Utility programs used by a PC", 80), ("7.04", "Library programs", 82),
        ("7.05", "Language translators", 83))),
    _chapter(level_prefix="al", number=8, name="Data security, privacy and integrity", start=88, end=98, sections=(
        ("8.01", "Definitions of data integrity, privacy and security", 88),
        ("8.02", "Security measures for protecting computer systems", 89),
        ("8.03", "Security measures for protecting data", 91),
        ("8.04", "Data validation and verification", 92))),
    _chapter(level_prefix="al", number=9, name="Ethics and ownership", start=99, end=108, sections=(
        ("9.01", "Ethics", 99), ("9.02", "The ACM/IEEE Software Engineering Code of Ethics", 99),
        ("9.03", "The public good", 101), ("9.04", "Ownership and copyright", 103),
        ("9.05", "The consequences of the development of the Internet", 104),
        ("9.06", "Software licensing", 105))),
    _chapter(level_prefix="al", number=10, name="Database and data modelling", start=109, end=124, sections=(
        ("10.01", "Limitations of a file-based approach", 109),
        ("10.02", "The database approach", 110), ("10.03", "The relational database", 112),
        ("10.04", "Entity-relationship modelling", 113),
        ("10.05", "A logical entity-relationship model", 116), ("10.06", "Normalisation", 117),
        ("10.07", "Structured Query Language (SQL)", 119, True), ("10.08", "DBMS features", 121))),
    _chapter(level_prefix="al", number=11, name="Algorithm design and problem solving", start=125, end=154, sections=(
        ("11.01", "What is an algorithm?", 125, True), ("11.02", "Expressing algorithms", 127, True),
        ("11.03", "Variables", 129, True), ("11.04", "Assignments", 129, True),
        ("11.05", "Logic statements", 133, True), ("11.06", "Loops", 137, True),
        ("11.07", "Working with arrays", 143, True))),
    _chapter(level_prefix="al", number=12, name="Stepwise refinement and structure charts", start=155, end=175, sections=(
        ("12.01", "Stepwise refinement", 155, True), ("12.02", "Modules", 158, True),
        ("12.03", "Structure charts", 166, True),
        ("12.04", "Deriving pseudocode from a structure chart", 169, True))),
    _chapter(level_prefix="al", number=13, name="Programming and data representation", start=176, end=211, sections=(
        ("13.01", "Programming languages", 176, True), ("13.02", "Programming basics", 179, True),
        ("13.03", "Data types", 185, True), ("13.04", "Boolean expressions", 186, True),
        ("13.05", "Selection", 187, True), ("13.06", "Iteration", 193, True),
        ("13.07", "Arrays", 197, True), ("13.08", "Built-in functions", 201, True),
        ("13.09", "Text files", 206, True))),
    _chapter(level_prefix="al", number=14, name="Structured programming", start=212, end=227, sections=(
        ("14.01", "Terminology", 212, True), ("14.02", "Procedures", 212, True),
        ("14.03", "Functions", 214, True), ("14.04", "Passing parameters to subroutines", 217, True),
        ("14.05", "Passing parameters to functions", 217, True),
        ("14.06", "Passing parameters to procedures", 219, True),
        ("14.07", "Putting it all together", 222, True))),
    _chapter(level_prefix="al", number=15, name="Software development", start=228, end=245, sections=(
        ("15.01", "Stages in the program development cycle", 228, True),
        ("15.02", "Features found in a typical Integrated Development Environment (IDE)", 229, True),
        ("15.03", "Testing strategies", 233, True), ("15.04", "Program testing using the IDE", 236, True),
        ("15.05", "Dry-running an algorithm", 239, True), ("15.06", "Corrective maintenance", 242),
        ("15.07", "Adaptive maintenance", 243))),
    _chapter(level_prefix="al", number=16, name="Data representation", start=246, end=257, sections=(
        ("16.01", "User-defined data types", 246, True), ("16.02", "File organisation", 248, True),
        ("16.03", "Real numbers", 250))),
    _chapter(level_prefix="al", number=17, name="Communication and Internet technologies", start=258, end=269, sections=(
        ("17.01", "Isolated network topologies", 258),
        ("17.02", "Communication and transmission concepts", 259),
        ("17.03", "Hardware connection devices", 260), ("17.04", "The TCP/IP protocol suite", 261),
        ("17.05", "Application-layer protocols associated with TCP/IP", 263),
        ("17.06", "Ethernet protocol", 265), ("17.07", "Peer-to-peer (P2P) file sharing", 266),
        ("17.08", "Wireless networks", 266))),
    _chapter(level_prefix="al", number=18, name="Boolean algebra and logic circuits", start=270, end=281, sections=(
        ("18.01", "Boolean algebra basics", 270), ("18.02", "Logic circuits", 271),
        ("18.03", "Boolean algebra applications", 274), ("18.04", "Karnaugh maps (K-maps)", 276))),
    _chapter(level_prefix="al", number=19, name="Processor and computer architecture", start=282, end=286, sections=(
        ("19.01", "The control unit", 282), ("19.02", "CISC and RISC processors", 282),
        ("19.03", "Parallel processing", 284))),
    _chapter(level_prefix="al", number=20, name="System software", start=287, end=302, sections=(
        ("20.01", "The purposes of an operating system (OS)", 287),
        ("20.02", "Process scheduling", 289), ("20.03", "Memory management", 291),
        ("20.04", "Virtual machine", 292), ("20.05", "Translation software", 293))),
    _chapter(level_prefix="al", number=21, name="Security", start=303, end=309, sections=(
        ("21.01", "Encryption fundamentals", 303),
        ("21.02", "Digital signatures and digital certificates", 304),
        ("21.03", "SSL and TLS", 306), ("21.04", "Malware", 306))),
    _chapter(level_prefix="al", number=22, name="Monitoring and control systems", start=310, end=316, sections=(
        ("22.01", "Logistics", 310), ("22.02", "Real-time", 311),
        ("22.03", "Bit manipulation", 312, True))),
    _chapter(level_prefix="al", number=23, name="Computational thinking and problem-solving", start=317, end=336, sections=(
        ("23.01", "What is computational thinking?", 317, True),
        ("23.02", "Standard algorithms", 318, True),
        ("23.03", "Abstract data types (ADTs)", 321, True), ("23.04", "Stacks", 321, True),
        ("23.05", "Queues", 321, True), ("23.06", "Linked lists", 322, True),
        ("23.07", "Binary trees", 328, True), ("23.08", "Hash tables", 331, True),
        ("23.09", "Dictionaries", 333, True))),
    _chapter(level_prefix="al", number=24, name="Algorithm design methods", start=337, end=346, sections=(
        ("24.01", "Decision tables", 337, True),
        ("24.02", "Jackson structured programming (JSP)", 339, True),
        ("24.03", "State-transition diagrams", 340, True))),
    _chapter(level_prefix="al", number=25, name="Recursion", start=347, end=355, sections=(
        ("25.01", "Concept of recursion", 347, True),
        ("25.02", "Programming a recursive subroutine", 348, True),
        ("25.03", "Tracing a recursive subroutine", 349, True),
        ("25.04", "Running a recursive subroutine", 351, True),
        ("25.05", "Benefits and drawbacks of recursion", 353, True))),
    _chapter(level_prefix="al", number=26, name="Further programming", start=356, end=367, sections=(
        ("26.01", "Programming paradigms", 356, True), ("26.02", "Records", 356, True),
        ("26.03", "File processing", 358, True), ("26.04", "Exception handling", 363, True),
        ("26.05", "Programming environments", 364, True))),
    _chapter(level_prefix="al", number=27, name="Object-oriented programming (OOP)", start=368, end=393, sections=(
        ("27.01", "Concept of OOP", 368, True), ("27.02", "Designing classes and objects", 369, True),
        ("27.03", "Writing object-oriented code", 371, True), ("27.04", "Inheritance", 375, True),
        ("27.05", "Polymorphism", 383, True), ("27.06", "Garbage collection", 385, True),
        ("27.07", "Containment (aggregation)", 385, True))),
    _chapter(level_prefix="al", number=28, name="Low level programming", start=394, end=404, sections=(
        ("28.01", "Processor instruction set", 394, True), ("28.02", "Symbolic addresses", 396, True),
        ("28.03", "Problem-solving and assembly-language programs", 396, True),
        ("28.04", "Absolute and relative addressing", 399, True),
        ("28.05", "Indirect addressing", 399, True))),
    _chapter(level_prefix="al", number=29, name="Declarative programming", start=405, end=419, sections=(
        ("29.01", "Imperative and declarative programming languages", 405, True),
        ("29.02", "Prolog basics", 405, True), ("29.03", "Facts in Prolog", 406, True),
        ("29.04", "Prolog variables", 407, True), ("29.05", "The anonymous variable", 409, True),
        ("29.06", "Rules in Prolog", 409, True), ("29.07", "Instantiation and backtracking", 410, True),
        ("29.08", "Recursion", 412, True), ("29.09", "Lists", 413, True))),
    _chapter(level_prefix="al", number=30, name="Software development", start=420, end=429, sections=(
        ("30.01", "Program generators and program libraries", 420, True),
        ("30.02", "Why errors occur and how to find them", 420, True),
        ("30.03", "Testing methods", 420, True), ("30.04", "Test plans and test data", 421, True),
        ("30.05", "How to prevent errors", 423, True), ("30.06", "Project management", 423))),
)


BOOK_CATALOG: dict[str, tuple[dict[str, Any], ...]] = {
    "O_LEVEL": O_LEVEL_BOOK,
    "A_LEVEL": A_LEVEL_BOOK,
}


def printed_page_label(start: int, end: int) -> str:
    """Return a label which unambiguously refers to printed book pages."""
    return f"Book p. {start}" if start == end else f"Book pp. {start}–{end}"


def public_book_catalog(level: str) -> list[dict[str, Any]]:
    """Return a JSON-ready copy with clear printed-page labels."""
    return [{
        **chapter,
        "book_page_label": printed_page_label(chapter["book_page_start"], chapter["book_page_end"]),
        "topics": [{
            **topic,
            "book_page_label": printed_page_label(topic["book_page_start"], topic["book_page_end"]),
        } for topic in chapter["topics"]],
    } for chapter in BOOK_CATALOG[level]]
