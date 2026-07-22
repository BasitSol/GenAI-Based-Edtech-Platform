"""Opt-in RAGAS evaluation using real metric objects and explicit coverage."""
from __future__ import annotations

import os
import sys
import types
import math


class _EmbeddingCompatibilityAdapter:
    """Bridge RAGAS 0.4 native embeddings to LangChain-style metric calls."""

    def __init__(self, inner):
        self.inner = inner

    def embed_query(self, text: str):
        return self.inner.embed_text(text)

    # RAGAS metrics in 0.4.3 call both its native names and the older
    # LangChain-compatible names depending on the metric implementation.
    def embed_text(self, text: str):
        return self.inner.embed_text(text)

    def embed_documents(self, texts: list[str]):
        return self.inner.embed_texts(texts)

    def embed_texts(self, texts: list[str]):
        return self.inner.embed_texts(texts)

    async def aembed_query(self, text: str):
        # The configured OpenAI client is synchronous; execute it safely in a
        # worker rather than invoking the native async method with a sync client.
        import asyncio
        return await asyncio.to_thread(self.embed_query, text)

    async def aembed_documents(self, texts: list[str]):
        import asyncio
        return await asyncio.to_thread(self.embed_documents, texts)

    async def aembed_text(self, text: str):
        return await self.aembed_query(text)

    async def aembed_texts(self, texts: list[str]):
        return await self.aembed_documents(texts)
from statistics import mean


METRICS = ("context_precision", "context_recall", "faithfulness", "answer_relevancy", "answer_correctness", "noise_sensitivity")


def _compatibility_shim() -> None:
    """Avoid a RAGAS optional-Vertex import failure when Vertex is unused."""
    name = "langchain_community.chat_models.vertexai"
    if name not in sys.modules:
        module = types.ModuleType(name)
        module.ChatVertexAI = type("ChatVertexAI", (), {})
        sys.modules[name] = module


def evaluate_ragas(records: list[dict], answer_rows: list[dict]) -> dict:
    if not os.getenv("OPENAI_API_KEY"):
        return {"status": "NOT_MEASURED", "reason": "OPENAI_API_KEY is required for opt-in RAGAS judge calls", "rows": []}
    by_id = {record["id"]: record for record in records}
    eligible = [row for row in answer_rows if row.get("execution_status") == "COMPLETED" and row.get("contexts") and row.get("answer") and row["id"] in by_id]
    if not eligible:
        return {"status": "NOT_MEASURED", "reason": "No successfully generated answers with contexts", "rows": []}
    try:
        _compatibility_shim()
        # Use RAGAS 0.4's native OpenAI adapters. The LangChain wrapper API
        # changed in the installed 1.x releases (embed_query/aembed_text),
        # which otherwise produces misleading partial scores and NaN values.
        from openai import OpenAI
        from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings
        from ragas.llms import llm_factory
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"],
                        timeout=float(os.getenv("RAGAS_REQUEST_TIMEOUT_SECONDS", "60")),
                        max_retries=0)
        judge_model = os.getenv("RAGAS_JUDGE_MODEL", os.getenv("GENERATOR_MODEL", "gpt-4.1-mini"))
        judge_llm = llm_factory(judge_model, provider="openai", client=client,
                                temperature=0, max_tokens=int(os.getenv("RAGAS_MAX_TOKENS", "2048")))
        judge_embeddings = _EmbeddingCompatibilityAdapter(RagasOpenAIEmbeddings(
            client=client, model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")))
        from ragas import EvaluationDataset, evaluate
        from ragas.run_config import RunConfig
        from ragas.dataset_schema import SingleTurnSample
        # These exports are initialized Metric instances.  Passing collection
        # dataclasses was the root cause of the old "metrics must be objects" failure.
        from ragas.metrics import (answer_correctness, answer_relevancy, context_precision,
                                  context_recall, faithfulness)
        from ragas.metrics._noise_sensitivity import NoiseSensitivity

        metric_objects = [context_precision, context_recall, faithfulness, answer_relevancy,
                          answer_correctness, NoiseSensitivity()]
        for metric in metric_objects:
            if hasattr(metric, "llm"):
                metric.llm = judge_llm
            if hasattr(metric, "embeddings"):
                metric.embeddings = judge_embeddings
        # One generation is sufficient for a deterministic evaluation and
        # avoids incomplete multi-sample responses on small judge models.
        answer_relevancy.strictness = 1
        samples = [SingleTurnSample(
            user_input=by_id[row["id"]]["question"], retrieved_contexts=row["contexts"],
            response=row["answer"], reference=by_id[row["id"]]["gold_answer"],
        ) for row in eligible]
        result = evaluate(EvaluationDataset(samples=samples), metrics=metric_objects,
                          show_progress=True, raise_exceptions=False,
                          run_config=RunConfig(
                              timeout=int(os.getenv("RAGAS_JOB_TIMEOUT_SECONDS", "90")),
                              max_retries=1, max_workers=int(os.getenv("RAGAS_MAX_WORKERS", "4")),
                          ))
        frame = result.to_pandas()
        rows = []
        for source, (_, scored) in zip(eligible, frame.iterrows()):
            row = {"id": source["id"]}
            for metric in METRICS:
                if metric in scored and scored[metric] is not None:
                    try:
                        value = float(scored[metric])
                        if math.isfinite(value):
                            row[metric] = value
                    except (TypeError, ValueError):
                        pass
            rows.append(row)
        summary = {metric: mean([row[metric] for row in rows if metric in row]) if any(metric in row for row in rows) else None for metric in METRICS}
        return {"status": "COMPLETED", "attempted_count": len(eligible), "scored_count": len(rows),
                "coverage": len(rows) / len(eligible), **summary, "rows": rows}
    except Exception as exc:
        return {"status": "FAILED", "reason": f"{type(exc).__name__}: {str(exc)[:700]}",
                "attempted_count": len(eligible), "scored_count": 0, "coverage": 0.0, "rows": []}
