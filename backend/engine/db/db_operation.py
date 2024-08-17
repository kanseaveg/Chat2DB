"""
   Retrieve the path from the configuration file, connect to the database, execute SQL, and return it to the frontend.
"""

import os, json, logging
import sqlite3
from ...utils.constants import DB_NOT_EXIST_ERROR
import logging

logging.basicConfig(level=logging.INFO)


def check_database_exists(SQLITE_DATABASE_PATH, db_id):
    db_path = os.path.join(SQLITE_DATABASE_PATH, db_id, f"{db_id}.sqlite")
    return os.path.exists(db_path)


# get db list
async def get_db_list(settings, db_source):
    if db_source == settings.PUBLIC_DATABASE:
        SQLITE_DATABASE_PATH = settings.PUBLIC_SQLITE_DATABASE_PATH
    elif db_source == settings.PRIVATE_DATABASE:
        SQLITE_DATABASE_PATH = settings.PRIVATE_SQLITE_DATABASE_PATH
    else:
        raise "Database Path error"
    db_list = []
    for item in os.listdir(SQLITE_DATABASE_PATH):
        item_path = os.path.join(SQLITE_DATABASE_PATH, item)
        if os.path.isdir(item_path):
            db_list.append(item)

    return db_list


# get db schema by db_id
async def get_db_schema(settings, db, db_source):
    if db_source == settings.PUBLIC_DATABASE:
        SQLITE_DATABASE_PATH = settings.PUBLIC_SQLITE_DATABASE_PATH
    elif db_source == settings.PRIVATE_DATABASE:
        SQLITE_DATABASE_PATH = settings.PRIVATE_SQLITE_DATABASE_PATH
    else:
        raise "Database Path error"
    if not check_database_exists(SQLITE_DATABASE_PATH, db):
        return DB_NOT_EXIST_ERROR

    db_path = os.path.join(SQLITE_DATABASE_PATH, db, f"{db}.sqlite")
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        # Get tables and columns
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        db_schema = []
        for table in tables:
            table_name = table[0]
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = [column[1] for column in cursor.fetchall()]
            db_schema.append({table_name: columns})

    return db_schema


async def delete_db(settings, db):
    SQLITE_DATABASE_PATH = settings.PRIVATE_SQLITE_DATABASE_PATH
    try:
        os.remove(os.path.join(SQLITE_DATABASE_PATH, db, db + ".sqlite"))  # delete sqlite file
        os.rmdir(os.path.join(SQLITE_DATABASE_PATH, db))  # delete folder
        print("Folder deleted successfully！")
    except OSError as e:
        print(f"Error deleting folder：{e}")
    return


async def update_db(state, db):
    # translate db.sqlite to tables( json )
    SQLITE_DATABASE_PATH = state.settings.PRIVATE_SQLITE_DATABASE_PATH
    table = {}
    db_path = os.path.join(SQLITE_DATABASE_PATH, db, db + ".sqlite")
    print(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' order by name")  # 获取表名
    res = cursor.fetchall()
    table_names = [name[0] for name in res]
    pk = []
    fk = []
    pk_id = []
    fk_id = []
    tab_col_names = [[-1, "*"]]
    tab_col_tyeps = ['text']
    for i, name in enumerate(table_names):
        cursor.execute("PRAGMA table_info({})".format(name))  # 获取列名
        res = cursor.fetchall()
        col_names = []
        col_types = []
        for col_info in res:
            col_name = col_info[1]
            tab_col_names.append([i, col_name])
            col_type = col_info[2].lower()
            tab_col_tyeps.append(col_type)
            col_names.append(col_name)
            col_types.append(col_type.lower())

            if col_info[5] == 1:
                pk.append({"table_name_original": name,
                           "column_name_original": col_name})
        cursor.execute("PRAGMA foreign_key_list({})".format(name))  # 获取外键信息
        res = cursor.fetchall()
        if res:
            for fk_info in res:
                fk.append({"source_table_name_original": name,
                           "source_column_name_original": fk_info[3],
                           "target_table_name_original": fk_info[2],
                           "target_column_name_original": fk_info[4]})
    conn.close()

    table["column_names"] = tab_col_names
    table["column_names_original"] = tab_col_names
    table["column_types"] = tab_col_tyeps
    table["db_id"] = db
    table["table_names"] = table_names
    table["table_names_original"] = table_names
    for key in pk:
        table_id = table_names.index(key["table_name_original"])
        for i, col in enumerate(tab_col_names):
            if col[0] != table_id:
                continue
            if col[1] == key["column_name_original"]:
                pk_id.append(i)
                break

    for key in fk:
        sour_table_id = table_names.index(key["source_table_name_original"])
        tart_table_id = table_names.index(key["target_table_name_original"])
        fk_pair = []
        for i, col in enumerate(tab_col_names):
            if col[0] != sour_table_id:
                continue
            if col[1] == key["source_column_name_original"]:
                fk_pair.append(i)
                break
        for i, col in enumerate(tab_col_names):
            if col[0] != tart_table_id:
                continue
            if col[1] == key["target_column_name_original"]:
                fk_pair.append(i)
                break
        fk_id.append(fk_pair)
    table["foreign_keys"] = fk_id
    table["primary_keys"] = pk_id

    # send update signal to Model end
    settings = state.settings
    client = state.client

    try:
        url = settings.ATTENTION_BASED_SINGLE_TURN_UPDATE_DB_ADDRESS
        response = await send_table(client, table, url)
        logging.info("successfully add to attention-based parser. The response message is : " + str(response))
    except Exception as e:
        logging.error(e)
        response = None
    logging.info(response)
    
    # send singal to modeling for automatically starting adaptive retraining
    url = settings.ATTENTION_BASED_SINGLE_TURN_RETRAIN_ADDRESS
    retrain_response = await send_retrain(client, url)
    logging.info("successfully send retrain signal to modeling. The response message is : " + str(retrain_response))
    
    return response


async def send_retrain(client, url):
    try:
        response = await client.get(url)
        return response
    except Exception as e:
        logging.error(e)
        return None
        


async def send_table(client, table, url):
    try:
        response = await client.post(
            url,
            json=table
        )
        response = json.loads(response)
        response = response.get("msg", None)
        return response
    except Exception as e:
        logging.error(e)
        return None


# execute sql on corresponding db_id
async def execute_sql_query(settings, db, db_source, sql, current_page, page_size):
    if db_source == settings.PUBLIC_DATABASE:
        SQLITE_DATABASE_PATH = settings.PUBLIC_SQLITE_DATABASE_PATH
    elif db_source == settings.PRIVATE_DATABASE:
        SQLITE_DATABASE_PATH = settings.PRIVATE_SQLITE_DATABASE_PATH
    else:
        raise "Database Path error"
    if not check_database_exists(SQLITE_DATABASE_PATH, db):
        return DB_NOT_EXIST_ERROR

    db_path = os.path.join(SQLITE_DATABASE_PATH, db, f"{db}.sqlite")
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # Calculate OFFSET based on current_page and page_size
        offset = (current_page - 1) * page_size
        # Add LIMIT and OFFSET to the SQL query
        paginated_sql = f"{sql} LIMIT {page_size} OFFSET {offset}"

        cursor.execute(paginated_sql)
        result = cursor.fetchall()
        description = cursor.description
        res = {}
        res["result"] = result
        res["description"] = description
    return res
