"""
    This is mainly used to assemble the intermediate results of various engines and return them to the controller layer
    file chat.py, and then return them to the front end.
"""
import logging
logging.basicConfig(level=logging.INFO)
import json
import re

# pre
from ..engine.pre.security_check import pre_check_question_security
from ..engine.pre.content_review import content_review
from ..engine.pre.refuse_model import refuse_model_check
from ..engine.pre.schema_linking import schema_linking

# post
from ..engine.post.value_extraction import get_value_extraction
from ..engine.post.security_check import post_check_sql_security

# common status code
from backend.utils.constants import PRE_CHECK, POST_CHECK, MODEL_NOT_SUPPORTED, PREDICTED_SQL_IS_NONE, OK
# common status text
from backend.utils.constants import PRE_CHECK_SECURITY_ERROR, PRE_CHECK_CONTENT_ILLEGAL, PRE_CHECK_QUESTION_REFUSE, POST_CHECK_SECURITY_ERROR, MODEL_NOT_SUPPORTED_ERROR, PREDICTED_SQL_IS_NONE_ERROR, SUCCESS
# parser currently supported
from backend.utils.constants import ATTENTION_BASED, LLM_ENHANCED
# engine currently supported
from backend.utils.constants import SINGLE_TURN, MULTI_TURN, NEED_VALUE_EXTRACTION_SYMBOL1, NEED_VALUE_EXTRACTION_SYMBOL2

"""
   

    This class mainly does the following things.
    Accept the question and parse it into SQL.
    1) Deliver the problem to the preprocessing engine to complete content review, rejection model,
          and security check, in order to achieve the purpose of pre-screening the problem.
    2) Deliver according to the requirements of the front-end model for parsing in single-turn model,
          multi-turn model, and LLM model to obtain predicted SQL results.
    3) If there is a "value" value in the result, it means that value extraction is required.
          Then request LLM to perform value extraction on the predicted SQL.
    4) If there is a "value" value in the result, it means that value extraction is required.
          Then request LLM to perform value extraction on the predicted SQL.
    
    Pay Attention : For all the checks, false indicates failure (existence of vulnerabilities or deficiencies),
                    and true indicates success. (means all inspection items have been passed, reasonable and legal.)
"""


async def chat_with_db_service(state, parser, engine, question, conv_id, db, db_source, selected_schemas):
    # pre!
    if await pre_check_question_security(question) is False:
        return {"msg": PRE_CHECK_SECURITY_ERROR, "code": PRE_CHECK}
    # temporarily close content review
    # if await content_review(state, question) is False:
    #     return {"msg": PRE_CHECK_CONTENT_ILLEGAL, "code": PRE_CHECK}
    if await refuse_model_check(state, question, db, db_source) is False:
        return {"msg": PRE_CHECK_QUESTION_REFUSE, "code": PRE_CHECK}

    # multi history retrieval
    question_buf = ""
    if engine == MULTI_TURN:
        # TODO: Get the latest 4 chat history using blank space as delimiter
        if conv_id in state.chat_logs:
            question_buf = " ".join(state.chat_logs[conv_id][-3:]) + " " + question
        else:
            question_buf = question
    elif engine == SINGLE_TURN:
        question_buf = question
    else:
        return {"msg": MODEL_NOT_SUPPORTED_ERROR, "code": MODEL_NOT_SUPPORTED}
            
    # schema linking
    q_s = await schema_linking(state, question_buf, db, selected_schemas)
    logging.info("q_s: " + str(q_s))
    selected_schemas = q_s["selected_schemas"]

    # using different text to sql parser!
    predicted_sql = ""
    if parser == ATTENTION_BASED:
        from ..engine.text2sql.plm_based.nl2sql import nl2sql
        predicted_sql = await nl2sql(state, q_s)
    elif parser == LLM_ENHANCED:
        from ..engine.text2sql.llm_enhanced.nl2sql import nl2sql
        predicted_sql_response = await nl2sql(state, q_s)
        try:
            predicted_sql_data = json.loads(predicted_sql_response)
            predicted_sql = predicted_sql_data["sql"]
        except Exception as e:
            logging.error("Error in parsing predicted_sql_response: " + str(e))
            try:
                json_match = re.search(r'\{.*\}', predicted_sql_response, re.DOTALL)
                if json_match:
                    predicted_sql_data = json.loads(json_match.group(0))
                    predicted_sql = predicted_sql_data["sql"]
                else:
                    raise ValueError("No valid JSON found in the response.")
            except Exception as inner_e:
                logging.error("Error in extracting JSON from predicted_sql_response: " + str(inner_e))
                return {"msg": PREDICTED_SQL_IS_NONE_ERROR, "code": PREDICTED_SQL_IS_NONE}
    else:
        return {"msg": MODEL_NOT_SUPPORTED_ERROR, "code": MODEL_NOT_SUPPORTED}

    logging.info("predicted_sql: " + str(predicted_sql))
    if predicted_sql is None:
        return {"msg": PREDICTED_SQL_IS_NONE_ERROR, "code": PREDICTED_SQL_IS_NONE}

    # post!
    if post_check_sql_security(predicted_sql) is False:
        return {"msg": POST_CHECK_SECURITY_ERROR, "code": POST_CHECK}

    # store legal chat history
    chat_logs = state.chat_logs
    if conv_id not in chat_logs:
        chat_logs[conv_id] = [question]
    else:
        chat_logs[conv_id].append(question)
    state.chat_logs = chat_logs

    # all finished!
    return {"msg": SUCCESS, "code": OK, "data": predicted_sql, "selected_schemas": selected_schemas}


"""
    Maintain user chat history. Used for multi-turn result concatenation.
    User chat history is currently temporarily cached in memory, and subsequent persistence operations 
    will be performed on the history.
"""
