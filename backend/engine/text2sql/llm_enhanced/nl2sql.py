"""
    Use few-shot samples to request LLM for solving the text-to-SQL problem.
"""
from backend.engine.db.db_operation import get_db_schema
import logging
import json

TASK_DEFINITION = ("Now you're a helpful Text-to-SQL assistant. Please help me to convert the following natural "
                   "language query into SQL query. For example, \n")

FEW_SHOT_SAMPLES = (
    "Q: Find the room number of the rooms which can sit 50 to 100 students and their buildings.\n"
    "SQL: SELECT DISTINCT building FROM classroom WHERE capacity > 50\n"
    "Q: Find the room number of the rooms which can sit 50 to 100 students and their buildings.\n"
    "SQL: SELECT building ,  room_number FROM classroom WHERE capacity BETWEEN 50 AND 100\n"
    "Q:Find the total budgets of the Marketing or Finance department.\n"
    "SQL:SELECT sum(budget) FROM department WHERE dept_name  =  'Marketing' OR dept_name  =  'Finance'\n"
    "Q:Find the total number of students and total number of instructors for each department.\n"
    "SQL:SELECT count(DISTINCT T2.id) , count(DISTINCT T3.id) ,  T3.dept_name FROM department AS T1 JOIN student AS "
    "T2 ON T1.dept_name = T2.dept_name JOIN instructor AS T3 ON T1.dept_name  =  T3.dept_name GROUP BY T3.dept_name\n"
    "Q:Find the title of courses that have two prerequisites?\n"
    "SQL: SELECT T1.title FROM course AS T1 JOIN prereq AS T2 ON T1.course_id  =  T2.course_id GROUP BY T2.course_id "
    "HAVING count(*) = 2\n"
)

PRE_INSTRUCTION = ("Given a natural language question and the corresponding schema information of a database, "
                   "Please return the SQL statement that corresponds to this natural language question."
                   "Your answer should strictly follow the following format:\n"
                   "{\n"
                    "\"reasoning\": \"\", // The reasoning steps for generating SQL.\n"
                    "\"sql\": \"\", // The final generated SQL.\n"
                    "}\n")

SCHEMA = ""
QUESTION = ""


async def nl2sql(state, q_s):
    settings = state.settings
    client = state.llm_http_client
    question = q_s["input_sequence"].split("|")[:1]
    schema = q_s["input_sequence"].split("|")[1:]
    schema = "|".join(schema)
    LLM_MODEL = settings.OPENAI_MODEL_NAME
    LLM_URL = settings.OPENAI_API_URL + "/v1/chat/completions"
    PROXY = settings.HTTP_PROXY_ADDRESS
    PROMPTS = {
        "model": LLM_MODEL,
        "messages": [
            {
                "role": "system",
                "content": f"{TASK_DEFINITION}"
            },
            {
                "role": "system",
                "content": f"{FEW_SHOT_SAMPLES}"
            },
            {
                "role": "system",
                "content": f"{PRE_INSTRUCTION}"
            },
            {
                "role": "assistant",
                "content": f"{schema}"
            },
            {
                "role": "user",
                "content": f"{question}"
            }
        ],
        "max_tokens": 256,
        "temperature": 0,
        "top_p": 1,
        "stop": [";"]
    }
    logging.info(f"prompts is : {PROMPTS}")

    try:
        response = await client.post_with_proxy(
            LLM_URL,
            json=PROMPTS,
            proxy=PROXY,
        )
        resp = json.loads(response)
        if "choices" in resp:
            return resp["choices"][0]["message"]["content"]
        elif "error" in resp:
            return resp["error"]["message"]
        else:
            raise Exception("LLM response error")
    except Exception as e:
        logging.error(e)
        return None


