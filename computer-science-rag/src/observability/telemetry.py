"""Privacy-aware local query telemetry used alongside optional LangSmith traces."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from src.core import ROOT


class TelemetryStore:
    def __init__(self, path: Path = ROOT / "data_processed/runtime/telemetry.sqlite"):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with sqlite3.connect(path) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS query_traces (
                trace_id TEXT PRIMARY KEY, session_id TEXT, created_at TEXT, category TEXT, intent TEXT,
                success INTEGER, failure_category TEXT, latency_ms REAL, input_tokens INTEGER,
                output_tokens INTEGER, total_cost REAL, cache_hit INTEGER, data TEXT NOT NULL)""")
            conn.execute("CREATE INDEX IF NOT EXISTS trace_created_at ON query_traces(created_at)")

    def record(self, trace: dict) -> None:
        values = (
            trace["trace_id"], trace.get("session_id"), trace.get("created_at") or datetime.now(timezone.utc).isoformat(),
            trace.get("category"), trace.get("intent"), int(bool(trace.get("success"))), trace.get("failure_category"),
            float(trace.get("latency_ms", 0)), int(trace.get("input_tokens", 0)), int(trace.get("output_tokens", 0)),
            float(trace.get("total_cost", 0)), int(bool(trace.get("cache_hit"))), json.dumps(trace, ensure_ascii=False),
        )
        with sqlite3.connect(self.path) as conn:
            conn.execute("INSERT OR REPLACE INTO query_traces VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", values)

    def summary(self, days: int = 30) -> dict:
        with sqlite3.connect(self.path) as conn:
            row = conn.execute("""SELECT COUNT(*), COUNT(DISTINCT session_id), AVG(latency_ms),
                AVG(input_tokens + output_tokens), AVG(total_cost), SUM(total_cost), AVG(success), AVG(cache_hit)
                FROM query_traces WHERE created_at >= datetime('now', ?)""", (f"-{days} days",)).fetchone()
            failures = dict(conn.execute("""SELECT COALESCE(failure_category,'NONE'), COUNT(*) FROM query_traces
                WHERE created_at >= datetime('now', ?) GROUP BY failure_category ORDER BY COUNT(*) DESC""", (f"-{days} days",)).fetchall())
            categories = dict(conn.execute("""SELECT COALESCE(category,'UNKNOWN'), COUNT(*) FROM query_traces
                WHERE created_at >= datetime('now', ?) GROUP BY category ORDER BY COUNT(*) DESC""", (f"-{days} days",)).fetchall())
            expensive = dict(conn.execute("""SELECT COALESCE(category,'UNKNOWN'), AVG(total_cost) FROM query_traces
                WHERE created_at >= datetime('now', ?) GROUP BY category ORDER BY AVG(total_cost) DESC""", (f"-{days} days",)).fetchall())
            models = dict(conn.execute("""SELECT COALESCE(json_extract(data,'$.model'),'non_llm'), COUNT(*) FROM query_traces
                WHERE created_at >= datetime('now', ?) GROUP BY json_extract(data,'$.model') ORDER BY COUNT(*) DESC""", (f"-{days} days",)).fetchall())
        daily=(row[5] or 0)/max(1,days)
        success=row[6]
        retrieval_failures=sum(value for key,value in failures.items() if key in {"RETRIEVAL_FAILURE","MISSING_CONTEXT"})
        incorrect_citations=failures.get("INVALID_CITATION_IDENTITY",0)
        total=row[0] or 0
        return {
            "window_days": days, "total_requests": row[0] or 0, "active_sessions": row[1] or 0,
            "average_latency_ms": row[2], "average_tokens": row[3], "average_cost": row[4],
            "total_cost": row[5] or 0, "daily_cost":daily, "weekly_cost":daily*7, "monthly_projected_cost":daily*30,
            "success_rate": success, "retrieval_success_rate":1-retrieval_failures/max(1,total),
            "hallucination_rate":None,
            "hallucination_measurement_status":"NOT_MEASURED_REQUIRES_SEMANTIC_EVALUATION",
            "citation_identity_accuracy":(1-incorrect_citations/total) if total else None,
            "cache_hit_rate": row[7], "failure_categories": failures, "question_categories": categories,
            "most_expensive_query_types":expensive, "model_utilization":models,
        }
