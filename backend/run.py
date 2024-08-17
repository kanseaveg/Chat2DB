import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from contextlib import asynccontextmanager
from fastapi import FastAPI
from functools import lru_cache
from settings import SystemSettings
from backend.controller import chat, db
from transformers import AutoTokenizer, AutoModel
from utils.httpclient import HttpClient
from aip import AipContentCensor
import logging
import uvicorn
import argparse


# load settings
@lru_cache()
def get_settings():
    return SystemSettings()


settings = get_settings()

@asynccontextmanager
async def before_first_request(app: FastAPI):
    logging.info("starting to process the database schema")
    # Cache the settings of the service
    app.state.settings = settings

    # Model Related
    print(os.getcwd())
    stb_model = AutoModel.from_pretrained(settings.SENTENCE_BERT_MODEL_URL)
    stb_tokenizer = AutoTokenizer.from_pretrained(settings.SENTENCE_BERT_MODEL_URL)
    app.state.sts_model = stb_model
    app.state.sts_tokenizer = stb_tokenizer

    # Client Related
    # baidu_client = AipContentCensor(settings.BAIDU_APP_ID, settings.BAIDU_API_KEY, settings.BAIDU_SECRET_KEY)
    # app.state.baidu_client = baidu_client

    # Different Parsers
    app.state.client = HttpClient(headers=settings.headers, timeout=60)
    app.state.llm_http_client = HttpClient(headers=settings.llm_headers, timeout=60)

    # Chat Logs
    app.state.chat_logs = {}

    # Check text2sql model's heart | TODO: Write a heartbeat detection interface in each parser.
    # TODO: add things to startup
    logging.info("finishing initial the model.")
    yield
    app.state.clear()


# set server
app = FastAPI(
    version=settings.VERSION,
    title=settings.TITLE,
    lifespan=before_first_request,
)
# add router
app.include_router(chat.router)
app.include_router(db.router)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Chat2DB')
    args = parser.parse_args()
    """
        currently on debug mode.
        fix fastapi problem: https://github.com/tiangolo/fastapi/issues/1495#issuecomment-643676192
    """
    uvicorn.run("run:app", host=settings.SERVER_HOST, port=settings.SERVER_PORT, reload=True, workers=1)

    """
        prod mode
    """
