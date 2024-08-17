from pydantic import BaseModel
from typing import Union, Optional


class ModelRequest(BaseModel):
    entry: dict


class LinkingRequest(BaseModel):
    db: str
    question: str
    selected_schemas: list
    # label: dict

class CkptRequest(BaseModel):
    selected_ckpt_path: str