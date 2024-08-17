"""
    Use few-shot samples to request LLM for solving the text-to-SQL problem.
"""
import json

from backend.engine.db.db_operation import get_db_schema
import logging


TASK_DEFINITION = ("### Now you are an expert in SQL writing, and you will fill in and supplement ON conditions for "
                   "the SQL statements returned by the multi-round model for me.Complete Mysql SQL query only and "
                   "with no explanation.\n ### \n ### You must complete three things:(1) If there is a placeholder "
                   "'value' in the SQL, please extract the value from the previous questions and current question to "
                   "replace the 'value'.Please note the order in the previous questions and current question, "
                   "and you should consider the co-reference relationships.; (2)If there is a \"JOIN\" keyword in the "
                   "SQL, please help me to add the \"ON\" condition; Only ON allowed to be added and Do not modify "
                   "other SQL skeleton ### \n")


async def get_value_extraction(question, db, db_source, predicted_sql, state, question_history=None):
    settings = state.settings
    client = state.llm_http_client
    db_schema = await get_db_schema(settings, db, db_source)

    # Environment Definition
    LLM_MODEL = settings.OPENAI_MODEL_NAME
    LLM_URL = settings.OPENAI_API_URL + "/v1/chat/completions"
    PROXY = settings.HTTP_PROXY_ADDRESS

    # Prompt Definition
    SUPPLY_MENT_INFORMATION = (
        f"### The db Scheam is: {db_schema} and i will give you the question and the sql as follows. ### \n"
    )

    if question_history is None:
        WIHOUT_HISTORY_VALUE_EXTRACTION = (
                "### \n The current question: '" + question + "' and the sql is: " + predicted_sql +
                " \n ### Please return the SELECT SQL statement to me directly and with no explanation."
        )
        prompt = {
            "model": LLM_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": f"{TASK_DEFINITION}"
                },
                {
                    "role": "system",
                    "content": f"{SUPPLY_MENT_INFORMATION}"
                },
                {
                    "role": "system",
                    "content": f"{WIHOUT_HISTORY_VALUE_EXTRACTION}"
                }
            ],
            "max_tokens": 256,
            "temperature": 0,
            "top_p": 1,
            "stop": [";"]
        }
    else:
        WITH_HISTORY_VALUE_EXTRACTION = (
                "### \n Previous question is: '" + question_history + "', the current question: '" + question +  "' and the sql is: "
                + predicted_sql + " \n ### Please return the SELECT SQL statement to me directly and with no explanation."
        )
        prompt = {
            "model": LLM_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": f"{TASK_DEFINITION}"
                },
                {
                    "role": "system",
                    "content": f"{SUPPLY_MENT_INFORMATION}"
                },
                {
                    "role": "system",
                    "content": f"{WITH_HISTORY_VALUE_EXTRACTION}"
                }
            ],
            "max_tokens": 256,
            "temperature": 0,
            "top_p": 1,
            "stop": [";"]
        }

    # Ask!
    try:
        response = await client.post_with_proxy(
            LLM_URL,
            json=prompt,
            proxy=PROXY,
        )
        # TODO: You can modify here according to the response format of the llm model.
        if response is not None:
            response = json.loads(response)
            response = response["choices"][0]["message"]["content"]
        return response
    except Exception as e:
        logging.error(e)
        return None







