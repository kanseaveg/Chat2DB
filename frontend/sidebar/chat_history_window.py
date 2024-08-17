import streamlit as st
def show_chat_window():
    st.write("---")
    st.markdown("# 🤖 Chat history window")
    chat_container = st.container()
    with chat_container:
        current_chat = st.radio(
            label="Chat Window",
            format_func=lambda x: x.split("_")[0] if "_" in x else x,
            options=st.session_state["history_chats_keys"],
            label_visibility="collapsed",
            index=st.session_state["current_chat_index"],
            key="current_chat"
                + st.session_state["history_chats_keys"][st.session_state["current_chat_index"]][0],
        )

        # update current_chat_index message
        st.session_state["current_chat_index"] = st.session_state["history_chats_keys"].index(current_chat)
        index = st.session_state["current_chat_index"]
        st.write(st.session_state["all_messages"][index])
        st.session_state["messages"] = st.session_state["all_messages"][index]

        st.session_state["history"] = st.session_state["all_history"][index]

        c1, c2 = st.columns(2)
        create_chat_button = c1.button(
            "**Create**", use_container_width=True, key="create_chat_button"
        )
        if create_chat_button:
            create_chat_fun()
            st.experimental_rerun()

        delete_chat_button = c2.button(
            "**Delete**", use_container_width=True, key="delete_chat_button"
        )
        if delete_chat_button:
            delete_chat_fun(current_chat)
            st.experimental_rerun()
    st.write("---")
    st.write("")
    st.write("")