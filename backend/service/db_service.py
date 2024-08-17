"""
    This class is mainly used to put the SQL submitted by the user into the database for execution,
    and return the corresponding execution result to the front-end.
"""
import os
# execution
from ..engine.db.db_operation import get_db_schema, get_db_list, execute_sql_query, delete_db, update_db
# post check
from ..engine.post.security_check import post_check_sql_security
# common status code
from ..utils.constants import POST_CHECK, OK, DB_NOT_EXIST, RESULT_IS_EMPTY_TEXT
# common status text
from ..utils.constants import POST_CHECK_SECURITY_ERROR, SUCCESS, DB_NOT_EXIST_ERROR, RESULT_IS_EMPTY

"""
    DB list acquisition logic: Just return the DB tree obtained from the database query.
"""


async def db_list_service(state, db_source):
    settings = state.settings
    schema = await get_db_list(settings, db_source)
    if schema == DB_NOT_EXIST_ERROR:
        return {"msg": DB_NOT_EXIST_ERROR, "data": None, "code": DB_NOT_EXIST}
    else:
        return {"msg": SUCCESS, "data": schema, "code": OK}


"""
    DB tree acquisition logic: Just return the DB tree obtained from the database query.
"""


async def db_tree_service(state, db, db_source):
    settings = state.settings
    schema = await get_db_schema(settings, db, db_source)
    if schema == DB_NOT_EXIST_ERROR:
        return {"msg": DB_NOT_EXIST_ERROR, "data": None, "code": DB_NOT_EXIST}
    else:
        # construct the response as db_tree
        schema_list = []
        for item in schema:
            for k, v in item.items():
                schema_list.append({"tableName": k, "tableColumns": v})
        return {"msg": SUCCESS, "data": schema_list, "code": OK}


"""
    DB delete acquisition logic: Deletes the specified database and returns a new list.
"""


async def db_delete_service(state, db):
    settings = state.settings
    schema = await delete_db(settings, db)
    if schema == DB_NOT_EXIST_ERROR:
        return {"msg": DB_NOT_EXIST_ERROR, "data": None, "code": DB_NOT_EXIST}
    else:
        return {"msg": SUCCESS, "data": schema, "code": OK}


"""
    DB update acquisition logic: Update the tables to Model end.
"""


async def db_update_service(state, db):
    schema = await update_db(state, db)
    if schema == DB_NOT_EXIST_ERROR:
        return {"msg": DB_NOT_EXIST_ERROR, "data": None, "code": DB_NOT_EXIST}
    else:
        return {"msg": SUCCESS, "data": schema, "code": OK}


"""
    Execution logic:
    1. Receive SQL statements submitted by the front end.
    2. This SQL statement will first undergo post security checks.
    3. And, put it into the database and execute 
    4. Finally, Return the results to streamlit front end.
"""


async def execute_sql_service(state, db, db_source, sql, current_page, page_size):
    settings = state.settings
    # post!
    if post_check_sql_security(sql) is False:
        return {"msg": POST_CHECK_SECURITY_ERROR, "code": POST_CHECK}
    # execute!
    result = await execute_sql_query(settings, db, db_source, sql, current_page, page_size)
    if result == DB_NOT_EXIST_ERROR:
        return {"msg": DB_NOT_EXIST_ERROR, "data": None, "code": DB_NOT_EXIST}
    elif result == RESULT_IS_EMPTY_TEXT:
        return {"msg": RESULT_IS_EMPTY, "data": None, "code": RESULT_IS_EMPTY}
    else:
        return {"msg": SUCCESS, "data": result, "code": OK}


async def get_ckpt_list_service(state):
    settings = state.settings
    ADAPTIVE_RETRAINING_BASE_CHECKPOINTS_PATH = settings.ADAPTIVE_RETRAINING_BASE_CHECKPOINTS_PATH
    not_ckpt_list = ["all-mpnet-base-v2", "text2sql_schema_item_classifier"]
    ckpt_list = []
    for ckpts in os.listdir(ADAPTIVE_RETRAINING_BASE_CHECKPOINTS_PATH):
        for ckpt in os.listdir(os.path.join(ADAPTIVE_RETRAINING_BASE_CHECKPOINTS_PATH, ckpts)):
            if ckpt not in not_ckpt_list and ckpt.startswith("checkpoint"):
                ckpt_list.append(ckpts + "/" + ckpt)
    return {"msg": SUCCESS, "data": ckpt_list, "code": OK}
