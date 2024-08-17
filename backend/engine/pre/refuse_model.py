"""
    We will use Sentence-BERT to calculate the vector distance between each word in the sentence
    and each word in the database, and set a certain threshold to determine the relevance of the current problem.
    STB Adopted from: https://huggingface.co/sentence-transformers/all-mpnet-base-v2
"""
import logging
logging.basicConfig(level=logging.INFO)
from sentence_transformers import util
import sqlite3
import os
import torch
import torch.nn.functional as F
from ..db.db_operation import check_database_exists
from ...utils.constants import DB_NOT_EXIST_ERROR


def mean_pooling(model_output, attention_mask):
    # First element of model_output contains all token embeddings
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


async def refuse_model_check(state, question, db, db_source):
    settings = state.settings
    if db_source == settings.PUBLIC_DATABASE:
        SQLITE_DATABASE_PATH = settings.PUBLIC_SQLITE_DATABASE_PATH
    elif db_source == settings.PRIVATE_DATABASE:
        SQLITE_DATABASE_PATH = settings.PRIVATE_SQLITE_DATABASE_PATH
    else:
        raise "Database Path error"
    if not check_database_exists(SQLITE_DATABASE_PATH, db):
        return DB_NOT_EXIST_ERROR

    db_path = os.path.join(SQLITE_DATABASE_PATH, db, f"{db}.sqlite")
    conn = sqlite3.connect(os.path.abspath(db_path))  # Replace with your SQLite database path
    cursor = conn.cursor()

    # Get all table name and column name for calculating the vector distance
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    # collect all table name and column name into a list
    columns = {}
    for table in tables:
        cursor.execute("PRAGMA table_info({})".format(table[0]))
        columns[table[0]] = cursor.fetchall()

    # Collect all table and column names into a list
    content_list = []
    for key in columns.keys():
        content_list.append(key)
        for column in range(len(columns[key])):
            content_list.append(columns[key][column][1])
    conn.close()

    # Calc the vector distance
    sentences = question.split()
    tokens = content_list

    # May Load in the startup of the server
    tokenizer = state.sts_tokenizer
    model = state.sts_model
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    encoded_token_input = tokenizer(tokens, padding=True, truncation=True, return_tensors='pt')
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
        model_token_output = model(**encoded_token_input)
    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    token_embeddings = mean_pooling(model_token_output, encoded_token_input['attention_mask'])
    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    token_embeddings = F.normalize(token_embeddings, p=2, dim=1)
    res = torch.max(util.pytorch_cos_sim(sentence_embeddings, token_embeddings))
    logging.info("The similarity of the question is: " + str(res))
    REFUSE_MODEL_THRESHOLD = settings.REFUSE_MODEL_THRESHOLD
    if res > REFUSE_MODEL_THRESHOLD:
        logging.info(f"The question is relevant to the current database. current REFUSE_MODEL_THRESHOLD is {REFUSE_MODEL_THRESHOLD}")
        return True
    else:
        logging.info("The question is irrelevant to the current database")
        return False  # irrelevant question to current database



