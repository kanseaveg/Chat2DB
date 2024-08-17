import streamlit as st
from frontend.utils.message_decorator import message_func
from frontend.entity.message import Message
import uuid


def append_chat_history(question, answer):
    st.session_state["history"].append((question, answer))


def append_message(answer, question, settings, db, is_sql, all_schemas, selected_schemas, role="assistant"):
    if is_sql:
        exec_key = str(uuid.uuid4())
        content_key = str(uuid.uuid4())
        edit_key = str(uuid.uuid4())
        reset_key = str(uuid.uuid4())
        select_box_key = str(uuid.uuid4())
    else:
        exec_key = None
        content_key = None
        edit_key = None
        reset_key = None
        select_box_key = None
    message = Message(role=role,
                      question=question,
                      answer=answer,
                      all_schemas=all_schemas,
                      selected_schemas=selected_schemas,
                      db=db,
                      is_sql=is_sql,
                      exec_key=exec_key,
                      content_key=content_key,
                      edit_key=edit_key,
                      reset_key=reset_key,
                      select_box_key=select_box_key
                      )
    st.session_state.messages.append(message)
    message_func(message, settings)
    append_chat_history(st.session_state.messages[-2].question, answer)
