import os, json, sys
import requests

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import streamlit as st
import uuid
from functools import lru_cache
from frontend.utils.message_decorator import StreamlitUICallbackHandler, message_func
from frontend.sidebar.db_schema_show import show_schema
from frontend.utils.message_utils import append_message
from settings import SystemSettings
from backend.entity.requestdto.db_request import DBEntity
from frontend.entity.message import Message

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@lru_cache()
def get_settings():
    return SystemSettings()


settings = get_settings()


def main():

    # ---------- Main page ----------
    st.title(settings.TITLE)
    engine_option_dict = {"Single Turn": "single-turn", "Multi Turn": "multi-turn"}
    engine = st.radio(
        "select your engine",
        options=list(engine_option_dict.keys()),
        index=1,
        horizontal=True,
    )
    parser_option_dict = {"PLM-Based Parser": "attention-based", "LLM-Based Parser": "llm-based"}
    # parser_option_dict = {"PLM-Based Parser": "attention-based", "LLM-Based Parser": "llm-enhanced"}
    parser = st.radio(
        "select your parser",
        options=list(parser_option_dict.keys()),
        index=1,
        horizontal=True,
    )

    st.session_state["engine"] = engine_option_dict[engine]
    st.session_state["parser"] = parser_option_dict[parser]
    with open(os.path.join(root_path, "frontend/resource/style/styles.md"), "r") as styles_file:
        styles_content = styles_file.read()

    st.write(styles_content, unsafe_allow_html=True)

    # Initialize the chat messages history
    if "messages" not in st.session_state.keys():
        st.session_state["messages"] = settings.INITIAL_MESSAGE
    if "history" not in st.session_state:
        st.session_state["history"] = []
    if "db_source" not in st.session_state.keys():
        st.session_state["db_source"] = settings.PUBLIC_DATABASE

    # Prompt for user input and save
    if prompt := st.chat_input():
        st.session_state.messages.append(Message(role="user", question=prompt, answer="", all_schemas=[], selected_schemas=[]))

    for i, message in enumerate(st.session_state.messages):
        st.session_state.messages[i] = message_func(message, settings)

    with open(os.path.join(root_path, "frontend/resource/style/sidebar.html"), "r", encoding="UTF-8") as sidebar_file:
        sidebar_html = sidebar_file.read()
    with st.sidebar:
        db, all_schemas = show_schema(settings)

    # ---------- Trigger a conversation ----------
    if st.session_state.messages[-1].role != "assistant":
        question = st.session_state.messages[-1].question
        if isinstance(question, str):
            db_source = st.session_state["db_source"]
            params = {
                "parser": st.session_state["parser"],
                "db_source": db_source,
                "engine": st.session_state["engine"],
                "question": question,
                "db": db,
            }
            BASIC_CHAT_URL = settings.BASIC_CHAT_URL
            selected_schemas = []
            try:
                resp = requests.post(BASIC_CHAT_URL, data=json.dumps(params), headers={"Content-Type": "application/json"}).text
                resp = json.loads(resp)
                status_code = resp["code"]
                if status_code == 200:
                    result = resp["data"]
                    selected_schemas = resp["selected_schemas"]
                else:
                    result = resp["msg"]
            except Exception as e:
                st.error(f"Error: {e}")
                return
            is_sql = True if result not in settings.INVALID_RESPONSE else False
            append_message(result, question, settings, db, is_sql, all_schemas, selected_schemas)


if __name__ == '__main__':
    main()
