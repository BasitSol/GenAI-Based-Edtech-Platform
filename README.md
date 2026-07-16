# Computer Science RAG — Phase 1

The project implements the deterministic-first Phase 1 architecture for curriculum-grounded O-Level (2210) and A-Level (9618) Computer Science retrieval.

`Data/` is treated as read-only. Every generated artifact is written below `data_processed/`.

The source PDFs are intentionally excluded from this public repository. Place the shared 25-document snapshot under `Data/` using the original folder structure before building the corpus.

## Run

```powershell
python -m pip install -r requirements.txt
python scripts/build_corpus.py
python -m src.chunking.pipeline
python scripts/build_indexes.py
python -m uvicorn src.api.main:app --reload
```

Run the Streamlit prototype with `python scripts/run_streamlit.py`. Add `OPENAI_API_KEY` to a local `.env` only when generation is required; without it the API safely returns retrieved evidence rather than inventing an answer.

For a local answer-generation setup, create `.env` from `.env.example` and set `OPENAI_API_KEY`. The model name may be left at its default or changed to one available to your OpenAI account. Never commit `.env`.

The final dense index uses local `BAAI/bge-m3` normalized 1,024-dimensional embeddings with an 8,192-token ceiling and deterministic contextual enrichment v2. Up to 24 candidates are reranked locally with full-precision `BAAI/bge-reranker-base` at 512 tokens, then compressed extractively before generation. Original source text and citation identity remain unchanged. Generation uses `gpt-4.1-mini`; the OpenAI key is never used for embeddings.

OpenAI embedding configurations and their historical evaluation results are retired. Only a validation report produced after the final BGE index rebuild is a current System A result.

Embedding vectors are cached under `data_processed/indexes/chroma/embedding_cache.sqlite`, keyed by provider, model, maximum input length, normalization settings and enriched source text. Re-running the index build with unchanged chunks reuses cached vectors. Changing the embedding model or contextual-enrichment input requires one complete vector rebuild.

## API

- `POST /ask`
- `POST /retrieve`
- `GET /documents`
- `GET /health`
- `GET /evaluation/status`

The answer schema reports the required official-status fields, citations, retrieval trace, latency, and mark-scheme availability. Exactness is only set after matching paper identity; nonmatching schemes are never labelled official.

## Rebuild and evaluate

After changing parser, cleaning, or chunking code, rebuild generated artifacts in this order. If only embedding/enrichment settings changed, run only `build_indexes.py` and the following verification/evaluation commands.

```powershell
python scripts/build_corpus.py
python -m src.chunking.pipeline
python scripts/build_indexes.py
python -m pytest -q
python scripts/run_evaluation.py --dataset evaluation/datasets/validation.jsonl
```

The evaluation scripts separate ingestion coverage, retrieval, answer/citation identity, grounding, cost, and latency. The benchmark contains 220 approved records split 120/50/50 across development, validation, and hidden test with no gold-source overlap.

Run `python scripts/create_benchmark_candidates.py` to generate review-gated exact-question candidates from the corpus. Move only human-approved records into the development, validation, or hidden-test datasets. `python scripts/run_evaluation.py` then reports Recall@5/10, MRR, nDCG@10, routing/exact-scheme accuracy, citation metrics, and latency; metrics are `null` until approved records include gold sources.

Generated answers are cached below `evaluation/cache/answers/`. The key includes the complete benchmark record, generator model, index manifest, and retrieval/generation source code. An unchanged evaluation rerun therefore makes no generation calls; a meaningful model, index, question, prompt, or pipeline change invalidates the relevant cache automatically. `estimated_cost` records the measured answer cost, while `evaluation_run_api_cost` records charges incurred by the current run.

Create the mandatory human-review packets with:

```powershell
python scripts/create_manual_quality_review_packets.py
```

Review instructions are in `evaluation/manual_review/README.md`. The completed values, reviewer, and review date are stored in `evaluation/manual_quality_gates.json`. System A completion is independent; building or evaluating a System B is outside this project's scope.

Detailed comparison reports are distributed separately from this code-only repository. Phase 1 deployment acceptance requires a fresh final-BGE validation result with `all_measured_gates_passed: true`.
