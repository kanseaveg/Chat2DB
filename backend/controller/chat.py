"""
chatapi :

The interfaces that need to be implemented are:
1) user chat
2) get user history
"""
import json
import logging

from fastapi import APIRouter, Request
from ..entity.requestdto.chat_request import ChatEntity
# add service here!
from ..service.chat_service import chat_with_db_service

router = APIRouter(prefix='/api/chat', tags=['basic'])


@router.post("/chat_with_db")
async def chat_with_db(chat_entity: ChatEntity, request: Request):
    parser = chat_entity.parser
    engine = chat_entity.engine
    question = chat_entity.question
    selected_schemas = chat_entity.selected_schemas
    conv_id = chat_entity.conv_id
    db = chat_entity.db
    db_source = chat_entity.db_source
    state = request.app.state
    response = await chat_with_db_service(state, parser, engine, question, conv_id, db, db_source, selected_schemas)
    return response