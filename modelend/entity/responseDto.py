from pydantic import BaseModel
from typing import Union


class ModelResponse(BaseModel):
    code: int
    data: Union[None, bool, str, list, dict]

