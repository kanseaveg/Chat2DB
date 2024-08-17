import html
import re
import os, json, requests
import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
import base64
from frontend.entity.message import Message
import pandas as pd
from st_aggrid import AgGrid
from streamlit_ace import st_ace
from frontend.sidebar.db_schema_show import show_schema
import uuid

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))

def format_message(text):
    """
    This function is used to format the messages in the chatbot UI.

    Parameters:
    text (str): The text to be formatted.
    """
    text_blocks = re.split(r"```[\s\S]*?```", text)
    code_blocks = re.findall(r"```([\s\S]*?)```", text)

    text_blocks = [html.escape(block) for block in text_blocks]

    formatted_text = ""
    for i in range(len(text_blocks)):
        formatted_text += text_blocks[i].replace("\n", "<br>")
        if i < len(code_blocks):
            formatted_text += f'<pre style="white-space: pre-wrap; word-wrap: break-word;"><code>{html.escape(code_blocks[i])}</code></pre>'

    return formatted_text

def get_base64_encoded_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string



def message_func(message: Message, settings):
    """
    This function is used to display the messages in the chatbot UI.

    Parameters:
    role (str): user or assistant.
    content (str):
    is_sql (bool):
    execute_sql (str):
    """
    if message.role == "user":
        image_path = os.path.join(root_path, "resource/images/working.png")
        image_base64 = get_base64_encoded_image(image_path)
        message_alignment = "flex-end"
        message_bg_color = "#6495ED"
        avatar_class = "user-avatar"
        st.write(
            f"""
                <div style="display: flex; align-items: center; margin-bottom: 10px; justify-content: {message_alignment};">
                    <div style="background: {message_bg_color}; color: white; border-radius: 20px; padding: 10px; margin-right: 5px; max-width: 75%; font-size: 14px;">
                        {message.question} \n </div>
                    <img src="data:image/png;base64,{image_base64}" class="{avatar_class}" alt="avatar" style="width: 50px; height: 50px;" />
                </div>
                """,
            unsafe_allow_html=True,
        )
        return message
    else:
        image_path = os.path.join(root_path, "resource/images/automation.png")
        image_base64 = get_base64_encoded_image(image_path)
        message_alignment = "flex-start"
        avatar_class = "bot-avatar"

        text = format_message(message.answer)
        with st.container():
            style = """
                        <style>
                            .message-container {
                                display: flex;
                                align-items: center;
                                justify-content: flex-start;
                                margin-bottom: 10px;
                            }

                            .message-container.right {
                                justify-content: flex-end;
                            }

                            .message-bubble {
                                background-color: #0e101c;
                                color: #fff;
                                border-radius: 20px;
                                padding: 10px 20px;
                                margin-left: 5px;
                                max-width: 75%;
                                font-size: 14px;
                                font-family: 'Roboto', sans-serif;
                                box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2);
                            }

                            .avatar {
                                width: 40px;
                                height: 40px;
                                border-radius: 50%;
                                object-fit: cover;
                                margin-right: 10px;
                            }
                        </style>
                        """

            html = f"""
                        <div class="message-container {message_alignment}">
                            <img src="data:image/png;base64,{image_base64}" class="{avatar_class}" alt="avatar" style="width: 50px; height: 50px;" />
                            <div class="message-bubble">
                                {text}
                            </div>
                        </div>
                        """
            st.markdown(style + html, unsafe_allow_html=True)
            if message.is_sql:

                # Interactive shcema linking
                if message.select_box_key is not None and message.reset_key is not None:
                    reset_com, schemas = st.columns([0.2, 0.8])
                    schemas = schemas.multiselect(
                        "none",
                        message.all_schemas,
                        message.selected_schemas,
                        key=message.select_box_key,
                        label_visibility="hidden")
                    with reset_com:
                        st.write("")
                        st.write("")
                        reset = st.button('**ReGenerate**', key=message.reset_key, type="primary")
                        if reset:
                            message.selected_schemas = schemas
                            db_source = st.session_state["db_source"]
                            params = {
                                "parser": st.session_state["parser"],
                                "db_source": db_source,
                                "engine": st.session_state["engine"],
                                "question": message.question,
                                "db": message.db,
                                "selected_schemas": message.selected_schemas
                            }
                            BASIC_CHAT_URL = settings.BASIC_CHAT_URL
                            try:
                                resp = requests.post(BASIC_CHAT_URL, data=json.dumps(params),
                                                     headers={"Content-Type": "application/json"}).text
                                resp = json.loads(resp)
                                status_code = resp["code"]
                                if status_code == 200:
                                    result = resp["data"]
                                else:
                                    result = resp["msg"]
                            except Exception as e:
                                st.error(f"Error: {e}")
                                return
                            new_message = Message(role=message.role,
                                              question=message.question,
                                              answer=result,
                                              all_schemas=message.all_schemas,
                                              selected_schemas=message.selected_schemas,
                                              db=message.db,
                                              is_sql=True,
                                              exec_key=message.exec_key,
                                              content_key=message.content_key,
                                              edit_key=message.edit_key,
                                              reset_key=None,
                                              select_box_key=None
                                              )
                            message.exec_key = None
                            message.edit_key = None
                            message.content_key = None
                            st.session_state.messages.append(new_message)
                            show_schema(settings, specified_db=message.db)
                            st.rerun()

                if message.content_key is not None and message.edit_key is not None and message.exec_key is not None:
                    # execute and Modify
                    is_execute, edit = st.columns([0.2, 0.8])
                    is_execute = is_execute.button('**Execute**', key=message.exec_key)
                    edit = edit.toggle('**Edit**', key=message.edit_key)
                    if edit:
                        content = st_ace(value=text, language="sql", min_lines=4, auto_update=True, key=message.content_key, theme="tomorrow_night_bright")
                        message.answer = content
                    if is_execute:
                        if message.execute_result is None or edit:
                            db_source = st.session_state["db_source"]
                            params = {
                                "db_source": db_source,
                                "sql": message.answer,
                                "db": message.db,
                            }
                            DB_EXECUTION_URL = settings.DB_EXECUTION_URL
                            resp = requests.post(DB_EXECUTION_URL, data=json.dumps(params),
                                                 headers={"Content-Type": "application/json"}).text
                            resp = json.loads(resp)
                            status_code = resp["code"]
                            if status_code == 200:
                                execute_result = resp["data"]
                            else:
                                execute_result = resp["msg"]

                            message.execute_result = execute_result
                            result = execute_result["result"]
                            description = execute_result["description"]
                            column_names = [description[0] for description in description]
                            df = pd.DataFrame(result, columns=column_names)
                            AgGrid(df)
                        else:
                            result = message.execute_result["result"]
                            description = message.execute_result["description"]
                            column_names = [description[0] for description in description]
                            df = pd.DataFrame(result, columns=column_names)
                            AgGrid(df)


        return message



#
class StreamlitUICallbackHandler(BaseCallbackHandler):
    def __init__(self):
        # Buffer to accumulate tokens
        self.token_buffer = []
        self.placeholder = None
        self.has_streaming_ended = False

    def _get_bot_message_container(self, text):
        """Generate the bot's message container style for the given text."""
        avatar_url = "https://avataaars.io/?avatarStyle=Transparent&topType=WinterHat2&accessoriesType=Kurt&hatColor=Blue01&facialHairType=MoustacheMagnum&facialHairColor=Blonde&clotheType=Overall&clotheColor=Gray01&eyeType=WinkWacky&eyebrowType=SadConcernedNatural&mouthType=Sad&skinColor=Light"
        message_alignment = "flex-start"
        message_bg_color = "#71797E"
        avatar_class = "bot-avatar"
        formatted_text = format_message(text)
        container_content = f"""
            <div style="display: flex; align-items: center; margin-bottom: 10px; justify-content: {message_alignment};">
                <img src="{avatar_url}" class="{avatar_class}" alt="avatar" style="width: 50px; height: 50px;" />
                <div style="background: {message_bg_color}; color: white; border-radius: 20px; padding: 10px; margin-right: 5px; max-width: 75%; font-size: 14px;">
                    {formatted_text} \n </div>
            </div>
        """
        return container_content

    def on_llm_new_token(self, token, run_id, parent_run_id=None, **kwargs):
        """
        Handle the new token from the model. Accumulate tokens in a buffer and update the Streamlit UI.
        """
        self.token_buffer.append(token)
        complete_message = "".join(self.token_buffer)
        if self.placeholder is None:
            container_content = self._get_bot_message_container(complete_message)
            self.placeholder = st.markdown(container_content, unsafe_allow_html=True)
        else:
            # Update the placeholder content
            container_content = self._get_bot_message_container(complete_message)
            self.placeholder.markdown(container_content, unsafe_allow_html=True)

    def display_dataframe(self, df):
        """
        Display the dataframe in Streamlit UI within the chat container.
        """
        avatar_url = "https://avataaars.io/?avatarStyle=Transparent&topType=WinterHat2&accessoriesType=Kurt&hatColor=Blue01&facialHairType=MoustacheMagnum&facialHairColor=Blonde&clotheType=Overall&clotheColor=Gray01&eyeType=WinkWacky&eyebrowType=SadConcernedNatural&mouthType=Sad&skinColor=Light"
        message_alignment = "flex-start"
        avatar_class = "bot-avatar"

        st.write(
            f"""
            <div style="display: flex; align-items: center; margin-bottom: 10px; justify-content: {message_alignment};">
                <img src="{avatar_url}" class="{avatar_class}" alt="avatar" style="width: 50px; height: 50px;" />
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write(df)

    def on_llm_end(self, response, run_id, parent_run_id=None, **kwargs):
        """
        Reset the buffer when the LLM finishes running.
        """
        self.token_buffer = []  # Reset the buffer
        self.has_streaming_ended = True

    def __call__(self, *args, **kwargs):
        pass