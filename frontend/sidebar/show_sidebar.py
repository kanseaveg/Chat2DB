# import streamlit as st
# from .chat_history_window import show_chat_window
# from .db_schema_show import show_schema
# def show_sidebar(db_list: list, root_path: str):
#     with open("chat2db/sidebar.html", "r", encoding="UTF-8") as sidebar_file:
#         sidebar_html = sidebar_file.read()
#     with st.sidebar:
#         # st.components.v1.html(sidebar_html, height=100)
#         # show_chat_window()
#         db = show_schema(db_list, root_path)
#     return db