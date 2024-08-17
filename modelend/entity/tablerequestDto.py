from pydantic import BaseModel
from typing import Union


class TableRequest(BaseModel):
    db_id: str
    column_names: Union[None, bool, str, list, dict]
    column_names_original: Union[None, bool, str, list, dict]
    column_types: Union[None, bool, str, list, dict]
    table_names: Union[None, bool, str, list, dict]
    table_names_original: Union[None, bool, str, list, dict]
    foreign_keys: Union[None, bool, str, list, dict]
    primary_keys: Union[None, bool, str, list, dict]