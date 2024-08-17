# -*- coding:utf-8 -*-
"""
    服务端终端接口，包括：
    1. 初始化模型
    2. nl-sql
"""
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from fastapi import FastAPI, Response
import json
import time
from entity.requestDto import ModelRequest, LinkingRequest, CkptRequest
from entity.tablerequestDto import TableRequest
import requests
from model.entry import start_demo, inference_sql, inference_linking, adaptive_retraining, update_demo
import logging
import asyncio

logging.basicConfig(level=logging.INFO)
import uvicorn
from functools import lru_cache
from settings import SystemSettings
from transformers import AutoTokenizer, AutoModel

# load settings
@lru_cache()
def get_settings():
    return SystemSettings()


settings = get_settings()


app = FastAPI(title="Chat2DB Model")
cur_state = app.state


@app.on_event("startup")
async def before_first_request():
    logging.info("starting to process the database schema")
    
    stb_model = AutoModel.from_pretrained(settings.SENTENCE_BERT_MODEL_URL)
    stb_tokenizer = AutoTokenizer.from_pretrained(settings.SENTENCE_BERT_MODEL_URL)
    cur_state.stb_model = stb_model
    cur_state.stb_tokenizer = stb_tokenizer
    cur_state.settings = settings
    
    args, db_list, tables, table_nat, tokenizer_classifier, model_classifier, tokenizer, model = start_demo()
    cur_state.args = args
    cur_state.tables = tables
    cur_state.tokenizer_classifier = tokenizer_classifier
    cur_state.model_classifier = model_classifier
    cur_state.tokenizer = tokenizer
    cur_state.model = model
    cur_state.db_list = db_list
    cur_state.table_nat = table_nat
    logging.info("finishing initial the model.")


@app.get("/retrain")
async def retrain():
    res = await adaptive_retraining(cur_state)
    # asyncio.create_task(adaptive_retraining(cur_state))
    return "Notification Successful"


@app.post("/ckpt/update")
async def update_parser(data: CkptRequest):
    del cur_state.model
    del cur_state.tokenizer
    selected_ckpt_path = data.selected_ckpt_path
    tokenizer, model = update_demo(selected_ckpt_path)
    cur_state.model = model
    cur_state.tokenizer = tokenizer
    return "Checkpoint updated successfully!"
    


@app.post("/update")
def update(data: TableRequest):
    if data.db_id in cur_state.db_list:
        ret_msg = {"code": 200,
                   "msg": "db in db list"}
    elif data.db_id is not None:
        logging.info(data)
        db_id: str
        data_dict = {"db_id": data.db_id, "column_names": data.column_names,
                     "column_names_original": data.column_names_original,
                     "column_types": data.column_types, "table_names": data.table_names,
                     "table_names_original": data.table_names_original,
                     "foreign_keys": data.foreign_keys, "primary_keys": data.primary_keys}
        cur_state.tables[data_dict["db_id"]] = data_dict
        cur_state.db_list.append(data_dict["db_id"])
        ret_msg = {"code": 200, "msg": "success"}
    else:
        ret_msg = {"code": 400, "msg": "please ether the corret parametes. db:{}"}
    logging.info(ret_msg)
    return Response(json.dumps(ret_msg).replace('\\', ''))


@app.post("/linking")
def linking(data: LinkingRequest):
    if data.db not in cur_state.db_list:
        ret_msg = {"code": 400,
                   "q_s": "QUERY ERROR. Your database is illegal or relevant. Please adjust the database."}
        logging.info(ret_msg)
    elif data.db is not None and data.question is not None:
        question = data.question
        q_s = inference_linking(
            question,
            data.db,
            data.selected_schemas,
            cur_state.args,
            cur_state.tables,
            cur_state.tokenizer_classifier,
            cur_state.model_classifier,
            # inter_label=data.label,
        )
        logging.info("q_s: {}".format(q_s))
        ret_msg = {"code": 200, "q_s": q_s}
    else:
        ret_msg = {"code": 400, "q_s": "please ether the corret parametes. db:{}, question:{}"}
        logging.info(ret_msg)

    return Response(json.dumps(ret_msg))


@app.post("/ask")
def ask(data: ModelRequest):
    entry = data.entry
    db = entry["db_id"]
    input_sequence = entry["input_sequence"]
    start = time.time()
    if db not in cur_state.db_list:
        ret_msg = {"code": 400,
                   "pre_sql": "QUERY ERROR. Your database is illegal or relevant. Please adjust the database."}
    elif db is not None and input_sequence is not None:

        pre_sql = inference_sql(
            [entry],
            cur_state.args,
            cur_state.tokenizer,
            cur_state.model,
            cur_state.table_nat,
        )
        logging.info("pre_sql: {}".format(pre_sql))
        ret_msg = {"code": 200, "pre_sql": pre_sql}
    else:
        ret_msg = {"code": 400, "pre_sql": "please ether the corret parametes. db:{}, question:{}"}
    end = time.time()
    logging.info(ret_msg)
    logging.info(f"It took more than {round(end - start, 2)} s to inference.")
    return Response(json.dumps(ret_msg))


if __name__ == "__main__":
    # uvicorn.run(app, host="localhost", port=8091)
    uvicorn.run("run_agent:app", host="localhost", port=8091, reload=True)
