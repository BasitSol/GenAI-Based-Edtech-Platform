"""Small, local conversation store. It deliberately retains only relevant state."""
from __future__ import annotations
import json, sqlite3, uuid
from pathlib import Path
from src.core import ROOT

class ConversationStore:
    def __init__(self, path: Path = ROOT / "data_processed/databases/conversations.sqlite"):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn=sqlite3.connect(path); self.conn.row_factory=sqlite3.Row
        self.conn.execute("CREATE TABLE IF NOT EXISTS conversations (conversation_id TEXT PRIMARY KEY, state TEXT NOT NULL)"); self.conn.commit()
    def get(self, conversation_id: str | None) -> dict:
        if not conversation_id: return {"conversation_id":str(uuid.uuid4()),"recent_messages":[]}
        row=self.conn.execute("SELECT state FROM conversations WHERE conversation_id=?",(conversation_id,)).fetchone()
        return json.loads(row["state"]) if row else {"conversation_id":conversation_id,"recent_messages":[]}
    def save(self,state:dict) -> str:
        state["recent_messages"]=state.get("recent_messages",[])[-4:]
        self.conn.execute("INSERT OR REPLACE INTO conversations VALUES (?,?)",(state["conversation_id"],json.dumps(state))); self.conn.commit(); return state["conversation_id"]
    def record(self, conversation_id, query, answer, route):
        state=self.get(conversation_id); state.update({"selected_level":route.get("level") or state.get("selected_level"),"selected_exam_year":route.get("exam_year") or state.get("selected_exam_year"),"selected_paper":route.get("subject_code") and f"{route.get('subject_code')}/{route.get('component','')}","selected_question":route.get("question_number") or state.get("selected_question")})
        state.setdefault("recent_messages",[]).extend([{"role":"user","content":query},{"role":"assistant","content":answer[:800]}]); return self.save(state)
