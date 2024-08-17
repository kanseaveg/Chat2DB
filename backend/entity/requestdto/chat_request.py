from pydantic import BaseModel
from typing import Optional


class ChatEntity(BaseModel):
    parser: Optional[str] = ""
    engine: Optional[str] = ""
    question: Optional[str] = ""
    conv_id: Optional[str] = ""
    db: Optional[str] = ""
    db_source: Optional[int] = 0
    selected_schemas: Optional[list] = []
