from pydantic import BaseModel
from typing import Optional


class DBEntity(BaseModel):
    db: Optional[str] = ""
    user_id: Optional[str] = ""
    sql: Optional[str] = ""
    db_source: Optional[int] = 0
    current_page: Optional[int] = 0
    page_size: Optional[int] = 20
