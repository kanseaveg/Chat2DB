from pydantic_settings import BaseSettings
from frontend.entity.message import Message


class SystemSettings(BaseSettings):
    # Server Related
    # SERVER_HOST: str = "0.0.0.0"
    SERVER_HOST: str = "127.0.0.1"
    SERVER_PORT: int = 8299
    TITLE: str = "Chat2DB"
    VERSION: str = "0.0.1"

    # URL Backend Base Url ★ Custom for every host that needs to modify ★
    BACKEND_BASE_URL: str = "http://"+str(SERVER_HOST)+":"+str(SERVER_PORT)
    INITIAL_MESSAGE: list = [
        Message(role="user", question="Hi!", answer="", is_init=True, all_schemas=[], selected_schemas=[]),
        Message(role="assistant",
                question="Hi!",
                answer="Hey there, I'm Chat2DB, your SQL-speaking copliot. I can turn your question into SQL and execute it in the databases 🔍",
                is_init=True,
                all_schemas=[],
                selected_schemas=[],
                )
    ]

    # Interface Related
    BASIC_CHAT_URL: str = BACKEND_BASE_URL + "/api/chat/chat_with_db"
    DB_LIST_URL: str = BACKEND_BASE_URL + "/api/db/list"
    DB_TREE_URL: str = BACKEND_BASE_URL + "/api/db/tree"
    DB_DELETE_URL: str = BACKEND_BASE_URL + "/api/db/delete"
    DB_UPDATE_URL: str = BACKEND_BASE_URL + "/api/db/update"
    DB_EXECUTION_URL: str = BACKEND_BASE_URL + "/api/db/execute"
    CKPT_LIST_URL: str = BACKEND_BASE_URL + "/api/db/ckpt/list"

    # Invalid response
    INVALID_RESPONSE: list = [
        "Pre-check question refuse, your question is Irrelevant to current database, please adjust your question.",
        "QUERY ERROR. Your database is illegal or relevant. Please adjust the database.",
        "The predicted sql is none.",
    ]

    # Database Related
    PUBLIC_SQLITE_DATABASE_PATH: str = "./data/spider/database"
    PRIVATE_SQLITE_DATABASE_PATH: str = "./data/usr/database"
    PUBLIC_DATABASE: int = 0
    PRIVATE_DATABASE: int = 1

    # Refuse Model Related Address
    BAIDU_APP_ID: str = ""
    BAIDU_API_KEY: str = ""
    BAIDU_SECRET_KEY: str = ""
    SENTENCE_BERT_MODEL_URL: str = "./checkpoints/all-mpnet-base-v2"
    REFUSE_MODEL_THRESHOLD: float = 0.4

    # Schema Linking Related Address
    SCHEMA_LINKING_SERVICE_ADDRESS: str = "http://"+SERVER_HOST+":8091/linking"

    # Attention-based Text-to-SQL Model Service Address and Ports
    ATTENTION_BASED_SINGLE_TURN_SERVICE_ADDRESS: str = "http://"+SERVER_HOST+":8091/ask"
    ATTENTION_BASED_SINGLE_TURN_UPDATE_DB_ADDRESS: str = "http://"+SERVER_HOST+":8091/update"

    # LLM-based Text-to-SQL Model Service Related
    OPENAI_API_URL: str = "https://api.deepseek.com"
    OPENAI_API_KEY: str = "Your-Keys-Here."
    OPENAI_MODEL_NAME: str = "deepseek-coder"

    # HTTP Header & PROXY Related
    HTTP_PROXY_ADDRESS: str = ""
    headers: dict = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0"
    }
    llm_headers: dict = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
        "Authorization": "Bearer " + OPENAI_API_KEY
    }
    
    # Retraining Related
    ATTENTION_BASED_SINGLE_TURN_RETRAIN_ADDRESS: str = "http://"+SERVER_HOST+":8091/retrain"
    CKPT_UPDATE_URL: str = "http://"+SERVER_HOST+":8091/ckpt/update"
    ADAPTIVE_RETRAINING_THRESHOLD: float = 0.4
    ADAPTIVE_RETRAINING_BASE_CHECKPOINTS_PATH: str = "./checkpoints"
    ADAPTIVE_RETRAINING_BASE_CHECKPOINTS: list = ["./checkpoints/text2sql-t5-base/checkpoint-39312"]
    ADAPTIVE_RETRAINING_BATCH_SIZE: int = 4
    ADAPTIVE_RETRAINING_EPOCHS: int = 1
    ADAPTIVE_RETRAINING_SAVE_STEPS: int = 1000
