"""SQLite metadata and exact-paper relationship index."""
from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

from backend.shared.core import read_jsonl


class MetadataStore:
    def __init__(self, path: Path):
        self.path = path
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row

    def rebuild(self, documents_path: Path, chunks_path: Path) -> None:
        cursor = self.connection.cursor()
        cursor.executescript("""
            DROP TABLE IF EXISTS documents;
            DROP TABLE IF EXISTS chunks;
            CREATE TABLE documents(document_id TEXT PRIMARY KEY, exact_pair_id TEXT, document_type TEXT, payload TEXT NOT NULL);
            CREATE TABLE chunks(chunk_id TEXT PRIMARY KEY, document_id TEXT, document_type TEXT, level TEXT, subject_code TEXT, year INTEGER,
                session TEXT, component TEXT, question_number TEXT, page_start INTEGER, parent_chunk_id TEXT, payload TEXT NOT NULL);
            CREATE INDEX chunks_route ON chunks(level, subject_code, year, session, component, question_number, document_type);
            CREATE INDEX chunks_document ON chunks(document_id, page_start);
        """)
        with documents_path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                cursor.execute("INSERT INTO documents VALUES (?,?,?,?)", (
                    row["document_id"], row.get("exact_pair_id") or None, row["document_type"], json.dumps(row),
                ))
        for item in read_jsonl(chunks_path):
            cursor.execute("INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (
                item["chunk_id"], item["document_id"], item["document_type"], item.get("level"), item.get("subject_code"), item.get("year"),
                item.get("session"), item.get("component"), item.get("question_number"), item.get("page_start"),
                item.get("parent_chunk_id"), json.dumps(item, ensure_ascii=False),
            ))
        self.connection.commit()

    def exact_paper_chunks(self, *, subject_code: str, year: int, session: str, component: str, question_number: str | None = None) -> list[dict]:
        query = "SELECT payload FROM chunks WHERE subject_code=? AND year=? AND session=? AND component=?"
        values: list = [subject_code, year, session, component]
        if question_number:
            query += " AND (question_number=? OR question_number LIKE ?)"
            values.extend([question_number, f"{question_number}(%"])
        return [json.loads(row[0]) for row in self.connection.execute(query, values)]

    def get_chunks(self, chunk_ids: list[str]) -> list[dict]:
        if not chunk_ids:
            return []
        placeholders = ",".join("?" for _ in chunk_ids)
        found = {row[0]: json.loads(row[1]) for row in self.connection.execute(
            f"SELECT chunk_id,payload FROM chunks WHERE chunk_id IN ({placeholders})", chunk_ids
        )}
        return [found[chunk_id] for chunk_id in chunk_ids if chunk_id in found]

    def parent(self, parent_chunk_id: str | None) -> dict | None:
        if not parent_chunk_id:
            return None
        row = self.connection.execute("SELECT payload FROM chunks WHERE chunk_id=?", (parent_chunk_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def close(self) -> None:
        """Close the SQLite handle; safe to call more than once."""
        try:
            self.connection.close()
        except sqlite3.ProgrammingError:
            pass
