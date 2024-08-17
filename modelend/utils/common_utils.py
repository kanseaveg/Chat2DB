import json
def load_json_file(path: str):
    with open(path, 'r',encoding='utf-8') as f:
        data = f.read()
        data = json.loads(data)
    return data

def save_json_file(path:str,data):
    with open(path,'w',encoding='utf-8') as f:
        json.dump(data,f,indent=4,ensure_ascii=False)

def list2dict(tables):
    new_tables = {}
    for t in tables:
        if t["db_id"] not in list(new_tables.keys()):
            new_tables[t["db_id"]] = t
    return new_tables

def get_db_schemas(db):
    db_schemas = {}


    table_names_original = db["table_names_original"]
    table_names = db["table_names"]
    column_names_original = db["column_names_original"]
    column_names = db["column_names"]
    column_types = db["column_types"]

    primary_keys, foreign_keys = [], []
    # record primary keys
    for pk_column_idx in db["primary_keys"]:
        pk_table_name_original = table_names_original[column_names_original[pk_column_idx][0]]
        pk_column_name_original = column_names_original[pk_column_idx][1]

        primary_keys.append(
            {
                "table_name_original": pk_table_name_original.lower(),
                "column_name_original": pk_column_name_original.lower()
            }
        )

    db_schemas["pk"] = primary_keys

    # record foreign keys
    for source_column_idx, target_column_idx in db["foreign_keys"]:
        fk_source_table_name_original = table_names_original[column_names_original[source_column_idx][0]]
        fk_source_column_name_original = column_names_original[source_column_idx][1]

        fk_target_table_name_original = table_names_original[column_names_original[target_column_idx][0]]
        fk_target_column_name_original = column_names_original[target_column_idx][1]

        foreign_keys.append(
            {
                "source_table_name_original": fk_source_table_name_original.lower(),
                "source_column_name_original": fk_source_column_name_original.lower(),
                "target_table_name_original": fk_target_table_name_original.lower(),
                "target_column_name_original": fk_target_column_name_original.lower(),
            }
        )
    db_schemas["fk"] = foreign_keys

    db_schemas["schema_items"] = []
    for idx, table_name_original in enumerate(table_names_original):
        column_names_original_list = []
        column_names_list = []
        column_types_list = []

        for column_idx, (table_idx, column_name_original) in enumerate(column_names_original):
            if idx == table_idx:
                column_names_original_list.append(column_name_original.lower())
                column_names_list.append(column_names[column_idx][1].lower())
                column_types_list.append(column_types[column_idx])

        db_schemas["schema_items"].append({
            "table_name_original": table_name_original.lower(),
            "table_name": table_names[idx].lower(),
            "column_names": column_names_list,
            "column_names_original": column_names_original_list,
            "column_types": column_types_list
        })

    return db_schemas