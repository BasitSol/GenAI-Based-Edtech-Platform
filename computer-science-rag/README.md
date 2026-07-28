# Computer Science RAG Application

This directory contains the deployable Python and React application for the [GenAI-Based EdTech Platform](../README.md).

It implements two bounded backend modules:

- **Module 1:** immutable corpus ingestion, hybrid retrieval, grounded educational chat, citations, memory, and telemetry.
- **Module 2:** teacher-controlled assessment generation, 25-mark mock tests, PDF/DOCX exports, typed-submission grading, and human review.

For the platform overview and complete architecture diagrams, start with the [repository README](../README.md).

## Module boundaries

```text
React client
    |
FastAPI + JWT/RBAC
    |---------------- Module 1 RAG ----------------|
    | ingestion -> indexes -> retrieval -> answers |
    |----------------------------------------------|
    |
    |---------------- Module 2 --------------------|
    | assessments -> review -> exports -> grading  |
    |----------------------------------------------|
    |
Shared SQLite platform state + operational telemetry
```

Module 1 has no dependency on Module 2. Module 2 consumes Module 1 through its retrieval boundary and does not maintain a second vector-search implementation.

## Local paths

Application paths are resolved from this directory:

```text
computer-science-rag/
|-- Data/                  # Private source PDFs; ignored
|-- data_processed/
|   |-- builds/<build-id>/ # Immutable pages, chunks, figures, and indexes
|   |-- current.json       # Active build pointer
|   `-- runtime/           # Platform, conversation, telemetry, and cache SQLite
|-- backend/
|-- frontend/
|-- configs/
|-- evaluation/
|-- scripts/
`-- tests/
```

## Environment

Python 3.12 is recommended. From the repository root:

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r computer-science-rag\requirements.txt
```

Then configure the application:

```powershell
cd computer-science-rag
Copy-Item .env.example .env
```

Required production values:

```dotenv
OPENAI_API_KEY=your-key
JWT_SECRET_KEY=generate-a-unique-long-random-value
```

Important defaults are documented in [`.env.example`](.env.example). Library imports do not load `.env` implicitly; executable scripts load runtime configuration explicitly.

## Source data

The source layout is declared in [`configs/data_sources.yaml`](configs/data_sources.yaml):

```text
Data/
|-- Books/
|-- Past Papers/
|-- Mark Schemes/
`-- Syllabus/
```

The system classifies and fingerprints PDFs without modifying them. Supported document roles include:

| Document type | Primary role |
|---|---|
| Textbook | Curriculum facts and explanations |
| Syllabus | Scope and learning objectives |
| Question paper | Exact paper identity and sanitized assessment style |
| Mark scheme | Exact official answers and marking patterns |
| Examiner report | Examiner guidance and common-error context |

## Corpus and index build

From this directory:

```powershell
python scripts\build_corpus.py
python scripts\build_indexes.py
```

### Corpus construction

`build_corpus.py`:

1. fingerprints configured source files and implementation settings;
2. classifies documents and extracts structured metadata;
3. extracts native text with PyMuPDF and uses pdfplumber when needed;
4. applies selective PaddleOCR with Tesseract fallback;
5. cleans text and records quality/OCR diagnostics;
6. creates page-local textbook chunks and assessment-aware chunks;
7. writes a staged immutable build and `pending.json`.

### Index promotion

`build_indexes.py`:

1. loads the staged chunks;
2. creates the BM25 sparse index;
3. rebuilds the metadata SQLite index;
4. creates Chroma embeddings;
5. records the embedding/index identity in the manifest;
6. marks the build READY;
7. atomically replaces `current.json`.

Runtime code refuses to mix an active corpus with a different embedding model.

## Running locally

Install frontend packages once:

```powershell
cd frontend
npm ci
Copy-Item .env.example .env
cd ..
```

Start the backend:

```powershell
python scripts\run_api.py
```

Start the frontend in another terminal:

```powershell
python scripts\run_frontend.py
```

Local URLs:

- React: `http://127.0.0.1:5173`
- FastAPI: `http://127.0.0.1:8000`
- OpenAPI: `http://127.0.0.1:8000/docs`

## Configuration reference

| Setting | Purpose |
|---|---|
| `OPENAI_API_KEY` | Production embedding, answer, assessment, and grading calls |
| `EMBEDDING_MODEL` | Must match the model recorded by the active index |
| `GENERATOR_MODEL` | Chat and assessment generation model |
| `RERANKER_ENABLED` | Enables the local second-stage cross-encoder |
| `EXTRACTIVE_COMPRESSION_ENABLED` | Selects relevant source segments without generative rewriting |
| `JWT_SECRET_KEY` | Signs platform access tokens |
| `PLATFORM_DB_PATH` | Optional override for durable platform SQLite |
| `ASSESSMENT_PROVIDER_MAX_ATTEMPTS` | Bound for transient provider retries |
| `ASSESSMENT_VALIDATION_RETRIES` | Bound for failed assessment regeneration |
| `LANGSMITH_TRACING` | Enables optional remote workflow tracing |

See [`.env.example`](.env.example) for every supported setting.

## Assessment contracts

Quiz and assignment generation accept configurable counts and formats through the quiz boundary. The separate mock-test boundary enforces:

- exactly eight questions;
- exactly 25 total marks;
- a one-mark MCQ followed by structured two-to-four-mark items;
- coverage of every selected topic;
- no introduction of unselected topics;
- deterministic coding allowance based on the selected syllabus scope;
- factual grounding in textbook evidence;
- past-paper material used only for sanitized task style;
- teacher review before student publication.

All generated citations retain canonical document, chunk, page, source-role, and topic provenance.

## Exports

Teacher-owned assessments can be exported as:

- professionally styled PDF using ReportLab;
- editable DOCX using python-docx.

Exports are deterministic local rendering operations and do not invoke an AI model. Student-only and teacher-with-solutions variants are supported through the API.

## Testing

The test suite is offline by default. It disables real model downloads and removes any developer OpenAI key during test execution.

```powershell
python -m pytest -q
npm --prefix frontend run build
```

Focused suites:

```powershell
python -m pytest tests\test_retrieval_v2.py -q
python -m pytest tests\test_assessment_workflow.py -q
python -m pytest tests\test_assessment_exports.py -q
python -m pytest tests\test_grading_workflow.py -q
```

Evaluation commands:

```powershell
python scripts\run_evaluation.py
python scripts\run_assessment_evaluation.py
python scripts\run_grading_evaluation.py
python scripts\validate_mock_test_catalog.py
```

RAGAS evaluation is optional and separated under `ragas_evaluation/` because it may require paid model calls.

## Operational behavior

- `/health` reports whether an indexed build is ready without invoking providers.
- Missing evidence produces an explicit abstention or validation rejection.
- Missing API credentials produce a no-call state rather than fabricated output.
- Provider failures are logged and surfaced with bounded, sanitized diagnostics.
- Telemetry failures never break a student answer.
- Assessment drafts are saved only after the complete validation chain passes.
- Approved assessments with student submissions cannot be deleted as drafts.
- AI and human scores are stored separately for auditability.

## Known limitations

- Diagram page images are created during ingestion, but visual delivery is not yet reliable end to end.
- Module 2 currently generates and exports text/table/code assessment content; it does not embed source diagrams.
- Handwriting OCR and file-submission grading are deferred; the current grading workflow accepts typed text.
- Local SQLite is suitable for the current single-node deployment, not horizontal multi-instance operation.
- Quality depends on the completeness and extraction quality of the private source corpus.
