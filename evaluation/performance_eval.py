from __future__ import annotations
import statistics,math
def evaluate(answer_rows:list[dict]):
    samples=[row['latency_ms'] for row in answer_rows]
    ordered=sorted(samples)
    return {"count":len(samples),"median_latency_ms":statistics.median(samples) if samples else None,"p95_latency_ms":ordered[max(0,math.ceil(.95*len(samples))-1)] if samples else None,"minimum_latency_ms":ordered[0] if samples else None,"maximum_latency_ms":ordered[-1] if samples else None,"technical_failure_rate":sum(bool(row.get('technical_failure')) for row in answer_rows)/len(answer_rows) if answer_rows else None,"estimated_cost":sum(row.get('estimated_cost',0.0) for row in answer_rows)}
