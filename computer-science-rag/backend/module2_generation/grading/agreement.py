"""Transparent agreement metrics for AI versus teacher scores."""
from __future__ import annotations

from math import sqrt


def score_agreement(rows: list[dict]) -> dict:
    """Calculate MAE and Pearson correlation for teacher-reviewed samples.

    Rows lacking either an AI or human score are excluded rather than silently
    treated as zero. Correlation is only meaningful with at least two varying
    pairs, so it is reported as ``None`` otherwise.
    """
    pairs = [(float(row["ai_score"]), float(row["human_score"])) for row in rows
             if row.get("ai_score") is not None and row.get("human_score") is not None]
    if not pairs:
        return {"count": 0, "mae": None, "pearson_correlation": None}
    mae = sum(abs(ai - human) for ai, human in pairs) / len(pairs)
    ai_mean = sum(ai for ai, _ in pairs) / len(pairs)
    human_mean = sum(human for _, human in pairs) / len(pairs)
    numerator = sum((ai - ai_mean) * (human - human_mean) for ai, human in pairs)
    denominator = sqrt(sum((ai - ai_mean) ** 2 for ai, _ in pairs) * sum((human - human_mean) ** 2 for _, human in pairs))
    return {"count": len(pairs), "mae": round(mae, 4), "pearson_correlation": round(numerator / denominator, 4) if denominator else None}
