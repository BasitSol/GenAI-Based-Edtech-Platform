# Computer Science Educational RAG — Phase 1

Production-oriented retrieval-augmented generation for Cambridge O Level Computer Science (2210) and A Level Computer Science (9618). This is the `computer-science-rag` module inside the wider GenAI-Based EdTech Platform.

The implementation is a clean v2 architecture. Generated corpora, manifests, embeddings, Chroma data, BM25 indexes, SQLite databases, cached answers, and old evaluation outputs are not part of the source tree. A build becomes active only after corpus creation and every index complete successfully.

## Status

- Fresh v2 source architecture: implemented.
- Offline contract/unit suite: 22 tests passing.
- External services were not invoked during implementation.
- The benchmark generator produces 100 auditable records from exact paper/mark-scheme pairs, curated textbook questions, and clearly labelled source-derived textbook questions when exact pairs are insufficient. Human review remains recommended for source-derived rows.
- OpenAI `text-embedding-3-small` is the active low-memory embedding model.
- `BAAI/bge-reranker-base` is the local second-stage cross-encoder.
- `gpt-4.1-mini` is the default generator and RAGAS judge, configurable in `.env`.

## Architecture

### System overview

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1500 900" width="100%" role="img" aria-label="Phase 1 educational RAG architecture">
<style>text{font-family:Arial,sans-serif;fill:#162536}.title{font-size:28px;font-weight:700;fill:white}.lane{font-size:20px;font-weight:700}.box{font-size:16px;font-weight:600}.small{font-size:13px;fill:#425466}.node{fill:#fff;stroke:#6b8bab;stroke-width:2}.arrow{stroke:#526174;stroke-width:3;fill:none;marker-end:url(#a)}</style>
<defs><marker id="a" markerWidth="10" markerHeight="10" refX="9" refY="5" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#526174"/></marker></defs>
<rect width="1500" height="900" fill="#f7f9fc"/>
<rect x="25" y="20" width="1450" height="65" rx="12" fill="#173b5c"/><text x="55" y="61" class="title">Computer Science Educational RAG · Phase 1</text>
<rect x="35" y="110" width="1430" height="220" rx="14" fill="#e7f3ff" stroke="#94bee1"/><text x="60" y="145" class="lane">1 · OFFLINE KNOWLEDGE BUILD</text>
<g class="node"><rect x="60" y="175" width="175" height="75" rx="10"/><rect x="275" y="175" width="175" height="75" rx="10"/><rect x="490" y="175" width="175" height="75" rx="10"/><rect x="705" y="175" width="175" height="75" rx="10"/><rect x="920" y="175" width="175" height="75" rx="10"/><rect x="1135" y="175" width="270" height="75" rx="10" fill="#d4efd5"/></g>
<text x="147" y="208" text-anchor="middle" class="box">Data/ PDFs</text><text x="147" y="230" text-anchor="middle" class="small">books · papers · schemes</text>
<text x="362" y="208" text-anchor="middle" class="box">Extract + OCR</text><text x="362" y="230" text-anchor="middle" class="small">PyMuPDF · PaddleOCR</text>
<text x="577" y="208" text-anchor="middle" class="box">Metadata</text><text x="577" y="230" text-anchor="middle" class="small">level · year · source</text>
<text x="792" y="208" text-anchor="middle" class="box">Educational chunks</text><text x="792" y="230" text-anchor="middle" class="small">parent/child + enrichment</text>
<text x="1007" y="208" text-anchor="middle" class="box">Indexes</text><text x="1007" y="230" text-anchor="middle" class="small">Chroma · BM25 · SQLite</text>
<text x="1270" y="208" text-anchor="middle" class="box">READY manifest</text><text x="1270" y="230" text-anchor="middle" class="small">immutable build identity</text>
<path d="M235 212H270 M450 212H485 M665 212H700 M880 212H915 M1095 212H1130" class="arrow"/>
<rect x="35" y="360" width="1430" height="300" rx="14" fill="#ecf9e8" stroke="#9bc88e"/><text x="60" y="395" class="lane">2 · ONLINE LANGGRAPH QUERY WORKFLOW</text>
<g class="node"><rect x="60" y="430" width="175" height="75" rx="10"/><rect x="275" y="430" width="175" height="75" rx="10"/><rect x="490" y="430" width="175" height="75" rx="10"/><rect x="705" y="430" width="175" height="75" rx="10"/><rect x="920" y="430" width="175" height="75" rx="10"/><rect x="1135" y="430" width="270" height="75" rx="10" fill="#d4efd5"/></g>
<text x="147" y="463" text-anchor="middle" class="box">Question + memory</text><text x="147" y="485" text-anchor="middle" class="small">session / follow-up</text>
<text x="362" y="463" text-anchor="middle" class="box">Understand query</text><text x="362" y="485" text-anchor="middle" class="small">intent · category · level</text>
<text x="577" y="463" text-anchor="middle" class="box">Hybrid retrieve</text><text x="577" y="485" text-anchor="middle" class="small">exact + dense + BM25</text>
<text x="792" y="463" text-anchor="middle" class="box">BGE rerank</text><text x="792" y="485" text-anchor="middle" class="small">cross-encoder evidence</text>
<text x="1007" y="463" text-anchor="middle" class="box">Sufficiency check</text><text x="1007" y="485" text-anchor="middle" class="small">rewrite once / abstain</text>
<text x="1270" y="463" text-anchor="middle" class="box">Context → answer</text><text x="1270" y="485" text-anchor="middle" class="small">compress + source authority</text>
<path d="M235 467H270 M450 467H485 M665 467H700 M880 467H915 M1095 467H1130" class="arrow"/>
<rect x="440" y="545" width="620" height="70" rx="10" class="node"/><text x="750" y="575" text-anchor="middle" class="box">Exact mark scheme → deterministic official answer</text><text x="750" y="597" text-anchor="middle" class="small">Otherwise → GPT-4.1-mini grounded generation</text><path d="M1270 505V540H1065" class="arrow"/>
<rect x="35" y="690" width="1430" height="170" rx="14" fill="#fff4dc" stroke="#dfbd70"/><text x="60" y="725" class="lane">3 · VALIDATION, OBSERVABILITY, AND EVALUATION</text>
<g class="node"><rect x="60" y="755" width="220" height="65" rx="10"/><rect x="330" y="755" width="220" height="65" rx="10"/><rect x="600" y="755" width="220" height="65" rx="10"/><rect x="870" y="755" width="220" height="65" rx="10"/><rect x="1140" y="755" width="265" height="65" rx="10" fill="#ffe2a8"/></g>
<text x="170" y="783" text-anchor="middle" class="box">Grounding + citations</text><text x="170" y="803" text-anchor="middle" class="small">response contract</text>
<text x="440" y="783" text-anchor="middle" class="box">Streamlit / FastAPI</text><text x="440" y="803" text-anchor="middle" class="small">student response</text>
<text x="710" y="783" text-anchor="middle" class="box">live_answers.xlsx</text><text x="710" y="803" text-anchor="middle" class="small">question-level log</text>
<text x="980" y="783" text-anchor="middle" class="box">Benchmark + RAGAS</text><text x="980" y="803" text-anchor="middle" class="small">per-question metrics</text>
<text x="1272" y="783" text-anchor="middle" class="box">Excel report + telemetry</text><text x="1272" y="803" text-anchor="middle" class="small">aggregate + trace</text>
<path d="M280 787H325 M550 787H595 M820 787H865 M1090 787H1135" class="arrow"/>
</svg>

The diagram is embedded directly in this README and remains scalable on GitHub.

### Why this design

| Decision | Rationale | Cost/latency consequence |
|---|---|---|
| Metadata + dense + BM25 | Paper identities, exact technical terms, and paraphrased concepts require complementary retrieval paths. | One query embedding plus local sparse/metadata search. |
| Reciprocal-rank fusion | Combines rankings without pretending BM25 and vector scores are calibrated. | Negligible local cost. |
| BGE cross-encoder | Joint query/passage scoring improves top-context precision after broad retrieval. | Local CPU/RAM latency; applied only to a bounded candidate set. |
| Contextual enrichment | Embeddings see qualification, source role, paper identity, and page as well as source text. | More embedding tokens during the one-time build. |
| Page-local parent/child chunks | Prevents the old page-drift citation bug while balancing precise retrieval and explanatory context. | More records, but bounded expansion. |
| Extractive compression | Removes irrelevant text without introducing facts through abstractive rewriting. | Small local CPU cost; fewer generation tokens. |
| One corrective retry | Recovers vocabulary mismatch without an uncontrolled agent loop. | At most one additional retrieval pass. |
| Deterministic exact-scheme answers | Official evidence should not be paraphrased unnecessarily. | Avoids a generation call. |
| Structured answer/citation contract | The model writes clean Markdown and returns source keys separately; canonical identities are resolved in code. | Prevents source labels from breaking SQL, code, and prose. |
| Immutable build promotion | Runtime cannot mix pages, chunks, embeddings, or models from different builds. | Model/config changes require a new build. |

Hybrid RRF, contextual retrieval, reranking, graph orchestration, and semantic evaluation follow current primary guidance from [Microsoft hybrid ranking](https://learn.microsoft.com/en-us/azure/search/hybrid-search-ranking), [Anthropic contextual retrieval](https://www.anthropic.com/engineering/contextual-retrieval), [Google retrieval and ranking](https://cloud.google.com/vertex-ai/generative-ai/docs/retrieval-and-ranking), [Cohere reranking](https://docs.cohere.com/docs/reranking-best-practices), [LangGraph](https://langchain-ai.github.io/langgraph/how-tos/state-reducers/), [RAGAS metrics](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/), and [LangSmith evaluation](https://docs.langchain.com/langsmith/evaluation-types).

HyDE, unrestricted agentic retrieval, web fallback, and GraphRAG are not defaults: this corpus has strong document relationships and strict curriculum authority, while those approaches add calls, latency, and failure modes that must first demonstrate benchmark gains.

## Data and source authority

The system discovers textbooks, syllabuses, question papers, mark schemes, and examiner reports below `Data/`. It never modifies the source PDFs.

Authority is explicit:

1. A mark scheme is official only when subject, year, session, component, and question label match.
2. Syllabus and textbooks support curriculum facts and explanations.
3. Examiner reports support examiner feedback.
4. A nonmatching question or scheme cannot be presented as the official answer.

The benchmark combines deterministic exact question/mark-scheme pairs with curated and source-derived textbook curriculum records covering compiler/interpreter, SQL, binary search, memory, networks, databases, and algorithms. Every generated row stores its source page/chunk and provenance; no unsupported gold row is silently invented.

## Actual project structure

```text
computer-science-rag/
├── Data/                         # Local source PDFs; excluded from Git
├── configs/                      # Ingestion, retrieval, and evaluation policy
├── data_processed/               # Generated immutable builds; excluded from Git
│   ├── builds/<build_id>/
│   │   ├── pages/
│   │   ├── chunks/
│   │   ├── indexes/              # Chroma, BM25, metadata SQLite
│   │   └── manifest.json
│   ├── current.json              # Atomically promoted active build
│   └── runtime/                  # Conversation and telemetry SQLite
├── evaluation/
│   ├── datasets/                 # Generated strict benchmark
│   ├── answer_eval.py
│   ├── benchmark.py
│   ├── citation_eval.py
│   ├── excel_report.py
│   ├── failure_analysis.py
│   ├── ingestion_eval.py
│   ├── quality_gates.py
│   ├── ragas_eval.py
│   └── retrieval_eval.py
├── scripts/                      # Explicit user entry points
├── src/
│   ├── api/
│   ├── chunking/
│   ├── generation/
│   ├── indexing/
│   ├── ingestion/
│   ├── memory/
│   ├── observability/
│   ├── retrieval/
│   └── workflows/                # Bounded LangGraph state machine
├── streamlit_app/
├── tests/                        # Offline tests/fakes; no paid calls
├── .env.example
├── requirements.txt
└── README.md
```

## Configuration

Python 3.12 is recommended. From this directory:

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Add credentials only to the local `.env`. It is excluded from Git. Library imports never load `.env`; only explicitly invoked scripts do.

Important defaults:

```dotenv
OPENAI_API_KEY=
GENERATOR_MODEL=gpt-4.1-mini
EMBEDDING_MODEL=text-embedding-3-small
RERANKER_MODEL=BAAI/bge-reranker-base
RERANKER_ENABLED=true
EXTRACTIVE_COMPRESSION_ENABLED=true
LANGSMITH_TRACING=false
LANGSMITH_API_KEY=
```

`text-embedding-3-small` is API-hosted and avoids the BGE-M3 embedding model’s local memory requirement. The BGE reranker remains local. Its first use may download model weights from Hugging Face.

## Build commands

This is a clean build; do not run `python -m src.chunking.pipeline` separately. Corpus construction already performs chunking.

```powershell
python scripts\build_corpus.py
python scripts\build_indexes.py
python scripts\generate_enterprise_benchmark.py --target 100
python scripts\export_quality_review.py --sample-size 50
```

The generator uses exact assessment pairs first, then curated textbook questions, and finally source-derived textbook rows to reach the requested count. Source-derived rows are labelled for later human review.

`build_corpus.py` does not require the OpenAI key. `build_indexes.py` embeds the fresh corpus and incurs embedding API usage. It writes `current.json` only after Chroma, BM25, SQLite, and the READY manifest succeed.

## Run the system

```powershell
python scripts\run_streamlit.py
```

or:

```powershell
python scripts\run_api.py
```

FastAPI documentation is available at `http://127.0.0.1:8000/docs`.

Every question submitted in Streamlit is appended to `evaluation/results/live_answers.xlsx`. The live workbook records the question, generated answer, category, difficulty, execution status, retrieved documents/pages, citations, citation validity, latency, token usage, estimated cost, and trace ID. RAGAS is not silently run for arbitrary live questions because context recall and correctness require a gold reference; use the benchmark command below for per-question and aggregate RAGAS scores.

## Verification and evaluation

Offline unit/contract tests do not load `.env` or use paid APIs:

```powershell
$env:OPENAI_API_KEY=$null
$env:LANGSMITH_TRACING="false"
python -m pytest -q
```

Cost-controlled retrieval pilot:

```powershell
python scripts\run_evaluation.py --retrieval-only --limit 10 --allow-partial
```

Full retrieval validation:

```powershell
python scripts\run_evaluation.py --retrieval-only
```

Full answer evaluation and Excel report:

```powershell
python scripts\run_evaluation.py --excel
```

Full semantic evaluation with paid RAGAS judge calls:

```powershell
python scripts\run_evaluation.py --ragas --excel
```

RAGAS is deliberately opt-in. Without `--ragas`, faithfulness, answer relevancy, correctness, context precision/recall, and noise sensitivity are reported as `NOT_MEASURED`; no lexical proxy is substituted. Every benchmark record remains in JSON/Excel with an execution status and error, including retrieval or generation failures.

Measured retrieval metrics are Precision@5/10, Recall@5/10, MRR, nDCG@10, exact-scheme accuracy, coverage, and latency. Deterministic answer metrics are citation identity accuracy, citation gold precision, citation coverage, execution coverage, technical failures, tokens, cost, and latency. RAGAS supplies semantic context precision/recall, faithfulness, relevancy, correctness, and noise sensitivity.

A final run passes only when all required gates are measured and pass. A pilot, partial run, skipped RAGAS stage, missing answer, or failed row cannot produce `all_required_gates_passed: true`.

Metadata, assessment-boundary, and OCR text accuracy are human gates. Complete the three CSV files created under `evaluation/review_packets/` by adding reviewer, review date, and the requested correctness fields. They remain `NOT_MEASURED` until every exported row is reviewed.

## Observability and memory

Local SQLite telemetry records execution status, session, category, latency, tokens, cost, retrieved document/chunk identities, prompt version, model, and failure category. LangSmith tracing is optional and disabled by default. Enabling it can send prompts, source context, and responses to LangSmith, so it should be a deliberate deployment decision.

Conversation memory stores a bounded recent window plus a compact summary. Only linguistically dependent follow-ups are rewritten; a complete question such as “Explain binary search” cannot inherit an earlier SQL topic.

## Reproducibility and security

- `.env`, PDFs, virtual environments, generated builds, Chroma data, databases, caches, evaluation results, and model weights are excluded from Git.
- The manifest records source fingerprints, build identity, embedding model, index identity, counts, and build time.
- Runtime rejects an embedding model that does not match the active index.
- Changing source files, chunking schema, or embedding model requires a new immutable build.
- Never publish a LangSmith or OpenAI key. Revoke any credential that is accidentally committed.

## Known limitations

- Empirical quality is unknown until the fresh user-run build and benchmark finish.
- Quality is bounded by the supplied PDFs and exact mark-scheme coverage.
- OCR and diagram questions require manual spot-checking even when automated quality gates pass.
- RAGAS is judge-model dependent and complements, rather than replaces, gold retrieval metrics and human review.
- CPU cross-encoder reranking can dominate latency on low-memory machines.
