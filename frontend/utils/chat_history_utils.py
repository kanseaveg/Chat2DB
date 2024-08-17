import streamlit as st


def create_chat_fun():
    # save history
    st.session_state["all_history"].append(st.session_state["history"])
    st.session_state["all_messages"].append(st.session_state["messages"])
    # update history chat name
    if len(st.session_state["history"]) > 0:
        st.session_state["history_chats_keys"][st.session_state["current_chat_index"]] = st.session_state["history"][0][
                                                                                             0] + "_" + str(
            uuid.uuid4())
    # init message and history
    st.session_state["messages"] = INITIAL_MESSAGE
    st.session_state["history"] = []

    #create new chat
    st.session_state["history_chats_keys"] = [
        "New Chat_" + str(uuid.uuid4())
    ] + st.session_state["history_chats_keys"]

def delete_chat_fun(current_chat):
    if len(st.session_state["history_chats_keys"]) == 1:
        chat_init = "New Chat_" + str(uuid.uuid4())
        st.session_state["history_chats_keys"].append(chat_init)
    pre_chat_index = st.session_state["history_chats_keys"].index(current_chat)
    if pre_chat_index > 0:
        st.session_state["current_chat_index"] = (
            st.session_state["history_chats_keys"].index(current_chat) - 1
        )
    else:
        st.session_state["current_chat_index"] = 0
    st.session_state["history_chats_keys"].remove(current_chat)
def reset_chat_name_fun(chat_name):
    chat_name = chat_name + "_" + str(uuid.uuid4())