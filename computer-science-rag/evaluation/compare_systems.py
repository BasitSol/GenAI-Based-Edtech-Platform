from __future__ import annotations
def weighted_score(metrics:dict)->float:
    weights={"ingestion":.10,"retrieval":.25,"correctness":.20,"coverage":.15,"grounding":.15,"status":.05,"abstention":.05,"performance":.05}
    return round(sum(metrics.get(key,0)*weight for key,weight in weights.items()),4)
