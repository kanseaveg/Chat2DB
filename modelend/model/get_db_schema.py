import os, json, logging
import sqlite3

import logging

logging.basicConfig(level=logging.INFO)

# get db list
def get_db_list(settings, SQLITE_DATABASE_PATH):
    db_list = []
    for item in os.listdir(SQLITE_DATABASE_PATH):
        item_path = os.path.join(SQLITE_DATABASE_PATH, item)
        if os.path.isdir(item_path):
            db_list.append(item)

    return db_list


# get db schema by db_id
def get_db_schema(settings, db, SQLITE_DATABASE_PATH):
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