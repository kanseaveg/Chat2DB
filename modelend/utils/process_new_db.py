import logging
import os
import sqlite3
import argparse

def trans_nat_hyper_param():
    parser = argparse.ArgumentParser()


    parser.add_argument('--db_path', default='./uploaded_database', type=str)


    args = parser.parse_args()
    return args

def res_db_tab_col_name(db_name:str)-> list:
    db_path = os.path.join("uploaded_database", db_name, db_name + ".sqlite")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' order by name")  # 获取表名
    res = cursor.fetchall()
    db_info_list = []
    table_names = []
    table_names = [name[0] for name in res]
    for name in table_names:
        cursor.execute("PRAGMA table_info({})".format(name))  # 获取列名
        res = cursor.fetchall()
        col_names = []
        for col_info in res:
            col_name = col_info[1]
            col_names.append(col_name)
        db_info_list.append({"tableName": name, "tableColumns": col_names})
    conn.close()
    return db_info_list

def res_db_schema(db_name:str)->dict:
    schema = {}
    tables = {}
    schema_items = []
    db_path = os.path.join("uploaded_database", db_name, db_name + ".sqlite")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' order by name")  # 获取表名
    res = cursor.fetchall()
    table_names = [name[0] for name in res]
    pk = []
    fk = []
    pk_id = []
    fk_id = []
    tab_col_names = [[-1,"*"]]
    tab_col_tyeps = ['text']
    for i,name in enumerate(table_names):
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
        schema_items.append({"table_name_original": name,
                             "table_name": name,
                             "column_names": col_names,
                             "column_names_original": col_names,
                             "column_types":col_types})
        cursor.execute("PRAGMA foreign_key_list({})".format(name))  # 获取外键信息
        res = cursor.fetchall()
        if res:
            for fk_info in res:
                fk.append({"source_table_name_original": name,
                           "source_column_name_original": fk_info[3],
                           "target_table_name_original": fk_info[2],
                           "target_column_name_original": fk_info[4]})
    conn.close()
    schema["pk"] = pk
    schema["fk"] = fk
    schema["schema_items"] = schema_items

    tables["column_names"] = tab_col_names
    tables["column_names_original"] = tab_col_names
    tables["column_types"] = tab_col_tyeps
    tables["db_id"] = db_name
    tables["table_names"] = table_names
    tables["table_names_original"] = table_names
    for key in pk:
        table_id = table_names.index(key["table_name_original"])
        for i,col in enumerate(tab_col_names):
            if col[0] != table_id:
                continue
            if col[1] == key["column_name_original"]:
                pk_id.append(i)
                break

    for key in fk:
        sour_table_id = table_names.index(key["source_table_name_original"])
        tart_table_id = table_names.index(key["target_table_name_original"])
        fk_pair = []
        for i,col in enumerate(tab_col_names):
            if col[0] != sour_table_id:
                continue
            if col[1] == key["source_column_name_original"]:
                fk_pair.append(i)
                break
        for i,col in enumerate(tab_col_names):
            if col[0] != tart_table_id:
                continue
            if col[1] == key["target_column_name_original"]:
                fk_pair.append(i)
                break
        fk_id.append(fk_pair)
    tables["foreign_keys"] = fk_id
    tables["primary_keys"] = pk_id

    return {db_name: tables}

if __name__ == '__main__':
    from utils.common_utils import load_json_file

    tables = load_json_file("../schema_tables.json")
    tables = tables[0]