from pydantic import BaseModel
class AskRequest(BaseModel):
    query: str
    level: str | None = None
    exam_year: int | None = None
    conversation_id: str | None = None
    difficulty: str | None = None
