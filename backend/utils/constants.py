"""
    Define some common status text here.
"""

PRE_CHECK_SECURITY_ERROR = "Pre-check security error, please adjust your question."
PRE_CHECK_CONTENT_ILLEGAL = "Pre-check content illegal, please adjust your question."
PRE_CHECK_QUESTION_REFUSE = "Pre-check question refuse, your question is Irrelevant to current database, please adjust your question."
POST_CHECK_SECURITY_ERROR = "Post-check security error,please check if the SQL statement you submitted contains sensitive and dangerous words that could harm the system or if there are sensitive and dangerous words in the generated SQL."
MODEL_NOT_SUPPORTED_ERROR = "The Model you requested currently not supported."
SUCCESS = "success"
DB_NOT_EXIST_ERROR = "The database you requested does not exist."
RESULT_IS_EMPTY_TEXT = "The result is empty."
PREDICTED_SQL_IS_NONE_ERROR = "The predicted sql is none."

"""
    Define some common status code here.
"""
MODEL_NOT_SUPPORTED = 400
MODEL_ERROR = 401
PRE_CHECK = 402
POST_CHECK = 403
OK = 200
DB_NOT_EXIST = 410
RESULT_IS_EMPTY = 411
MODEL_PARSER_ERROR = 412
PREDICTED_SQL_IS_NONE = 413


"""
    Define Parser Type and Engine Type here and additional things
"""
ATTENTION_BASED = "attention-based"
LLM_ENHANCED = "llm-based"

SINGLE_TURN = "single-turn"
MULTI_TURN = "multi-turn"

NEED_VALUE_EXTRACTION_SYMBOL1 = "'value'"
NEED_VALUE_EXTRACTION_SYMBOL2 = "join"

