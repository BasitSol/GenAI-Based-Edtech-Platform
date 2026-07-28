"""Optional LLM-as-judge RAGAS evaluation, isolated from deterministic metrics."""

from .evaluator import evaluate_ragas

__all__ = ["evaluate_ragas"]
