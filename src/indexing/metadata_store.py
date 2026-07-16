from __future__ import annotations
import json, sqlite3
from pathlib import Path
from src.core import ROOT, read_jsonl
class MetadataStore:
    def __init__(self,path: Path=ROOT/"data_processed/databases/metadata.sqlite"):
        path.parent.mkdir(parents=True,exist_ok=True); self.conn=sqlite3.connect(path); self.conn.row_factory=sqlite3.Row
    def rebuild(self,manifest_csv: Path, chunks_path: Path):
        cur=self.conn.cursor(); cur.executescript("DROP TABLE IF EXISTS documents; DROP TABLE IF EXISTS pages; DROP TABLE IF EXISTS chunks; DROP TABLE IF EXISTS questions; DROP TABLE IF EXISTS mark_scheme_entries; DROP TABLE IF EXISTS examiner_report_entries; DROP TABLE IF EXISTS marking_patterns; DROP TABLE IF EXISTS relationships; CREATE TABLE documents (document_id TEXT PRIMARY KEY, data TEXT NOT NULL); CREATE TABLE pages (document_id TEXT, page_number INTEGER, data TEXT NOT NULL, PRIMARY KEY(document_id,page_number)); CREATE TABLE chunks (chunk_id TEXT PRIMARY KEY, document_id TEXT, document_type TEXT, level TEXT, year INTEGER, session TEXT, component TEXT, question_number TEXT, page_start INTEGER, data TEXT NOT NULL); CREATE INDEX chunks_lookup ON chunks(level,year,session,component,question_number); CREATE TABLE questions (chunk_id TEXT PRIMARY KEY, data TEXT NOT NULL); CREATE TABLE mark_scheme_entries (chunk_id TEXT PRIMARY KEY, data TEXT NOT NULL); CREATE TABLE examiner_report_entries (chunk_id TEXT PRIMARY KEY, data TEXT NOT NULL); CREATE TABLE marking_patterns (chunk_id TEXT PRIMARY KEY, data TEXT NOT NULL); CREATE TABLE relationships (source_id TEXT, target_id TEXT, relationship TEXT, PRIMARY KEY(source_id,target_id,relationship));")
        import csv
        with manifest_csv.open(encoding="utf-8") as f:
            for row in csv.DictReader(f): cur.execute("INSERT INTO documents VALUES (?,?)",(row["document_id"],json.dumps(row)))
        pages_root=chunks_path.parent.parent / "pages"
        # Page records remain authoritative for raw/clean text and OCR provenance.
        for row in cur.execute("SELECT document_id FROM documents").fetchall():
            for page in read_jsonl(pages_root / row[0] / "pages.jsonl"):
                cur.execute("INSERT INTO pages VALUES (?,?,?)",(row[0],page["page_number"],json.dumps(page)))
        for c in read_jsonl(chunks_path) + read_jsonl(chunks_path.parent / "marking_patterns.jsonl"):
            cur.execute("INSERT OR REPLACE INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?)",(c["chunk_id"],c["document_id"],c["document_type"],c.get("level"),c.get("year"),c.get("session"),c.get("component"),c.get("question_number"),c.get("page_start"),json.dumps(c)))
            if c.get("parent_chunk_id"): cur.execute("INSERT OR IGNORE INTO relationships VALUES (?,?,?)",(c["chunk_id"],c["parent_chunk_id"],"CHILD_OF"))
            table={"QUESTION_PAPER":"questions","MARK_SCHEME":"mark_scheme_entries","EXAMINER_REPORT":"examiner_report_entries","MARKING_PATTERN":"marking_patterns"}.get(c["document_type"])
            if table: cur.execute(f"INSERT OR REPLACE INTO {table} VALUES (?,?)",(c["chunk_id"],json.dumps(c)))
        # Exact pairing is deliberately only based on full paper identity.
        docs=[json.loads(x["data"]) for x in cur.execute("SELECT data FROM documents")]
        for question in [x for x in docs if x["document_type"]=="QUESTION_PAPER"]:
            for scheme in [x for x in docs if x["document_type"]=="MARK_SCHEME" and x.get("exact_pair_id") and x.get("exact_pair_id")==question.get("exact_pair_id")]: cur.execute("INSERT OR IGNORE INTO relationships VALUES (?,?,?)",(scheme["document_id"],question["document_id"],"EXACT_MARK_SCHEME_FOR"))
        self.conn.commit()
    def exact_chunks(self, level=None, year=None, session=None, component=None, question_number=None):
        query="SELECT data FROM chunks WHERE 1=1"; vals=[]
        for field,value in [("level",level),("year",year),("session",session),("component",component),("question_number",question_number)]:
            if value is not None:
                if field == "question_number":
                    value=str(value)
                    if "(" in value:
                        parents=[]; parent=value
                        while "(" in parent:
                            parent=parent.rsplit("(",1)[0]; parents.append(parent)
                        forms=[value]+parents
                        query += f" AND question_number IN ({','.join('?' for _ in forms)})"; vals.extend(forms)
                    else:
                        query += " AND (question_number=? OR question_number LIKE ?)"; vals.extend([value, f"{value}(%"])
                else: query+=f" AND {field}=?"; vals.append(value)
        return [json.loads(x[0]) for x in self.conn.execute(query,vals)]
