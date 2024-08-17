"""
execute sql api

The interfaces that need to be implemented are:
1) db list display
2) db tree display
"""
from fastapi import APIRouter, Request
from ..entity.requestdto.db_request import DBEntity
# add service here!
from ..service.db_service import db_tree_service, db_list_service, execute_sql_service, db_delete_service, \
    db_update_service, get_ckpt_list_service

router = APIRouter(prefix='/api/db', tags=['db'])



@router.post('/list')
async def db_list(db_entity: DBEntity, request: Request):
    state = request.app.state
    db_source = db_entity.db_source
    response = await db_list_service(state, db_source)
    return response


@router.post('/tree')
async def db_tree(db_entity: DBEntity, request: Request):
    db = db_entity.db
    db_source = db_entity.db_source
    state = request.app.state
    response = await db_tree_service(state, db, db_source)
    return response


@router.post('/delete')
async def db_delete(db_entity: DBEntity, request: Request):
    db = db_entity.db
    state = request.app.state
    response = await db_delete_service(state, db)
    return response


@router.post('/update')
async def db_update(db_entity: DBEntity, request: Request):
    db = db_entity.db
    state = request.app.state
    response = await db_update_service(state, db)
    return response


@router.post('/execute')
async def execute_sql(db_entity: DBEntity, request: Request):
    db = db_entity.db
    db_source = db_entity.db_source
    state = request.app.state
    sql = db_entity.sql
    current_page = db_entity.current_page
    page_size = db_entity.page_size
    response = await execute_sql_service(state, db, db_source, sql, current_page, page_size)
    return response


@router.get("/ckpt/list")
async def get_ckpt_list(request: Request):
    state = request.app.state
    response = await get_ckpt_list_service(state)
    return response
