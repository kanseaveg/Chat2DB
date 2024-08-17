import streamlit as st
import os, json, requests
from streamlit_extras.stoggle import stoggle
import time

def show_schema(settings, specified_db=None):
    with st.expander("💡**Select your mode**", expanded=True):

        st.write("")  # 空行，用于调整框的位置
        public_source, private_source = st.columns([1, 1])
        public_source = public_source.button('**Public**', use_container_width=True, )
        private_source = private_source.button('**Private**', use_container_width=True, )
        if public_source:
            st.session_state["db_source"] = settings.PUBLIC_DATABASE
        elif private_source:
            st.session_state["db_source"] = settings.PRIVATE_DATABASE
        # else:
        #     st.session_state["db_source"] = settings.PUBLIC_DATABASE  # default!

        DB_LIST_URL = settings.DB_LIST_URL
        DB_TREE_URL = settings.DB_TREE_URL
        DB_DELETE_URL = settings.DB_DELETE_URL
        DB_UPDATE_URL = settings.DB_UPDATE_URL
        all_schemas = []
        try:
            db_list_resp = requests.post(DB_LIST_URL, data=json.dumps({"db_source": st.session_state["db_source"]}))
            db_list = json.loads(db_list_resp.text)["data"]
            db_list = sorted(db_list)
        except json.JSONDecodeError as e:
            st.error(f"Error decoding JSON: {e}")
            db_list = []

        if st.session_state["db_source"] == settings.PUBLIC_DATABASE:
            db = st.selectbox("select database", db_list)

        elif st.session_state["db_source"] == settings.PRIVATE_DATABASE:
            # database_type = st.selectbox("select your database engine", ["SQLite", "MySql", "SqlServer"])
            database_type = st.selectbox("select your database engine", ["SQLite", "Only Release SQLite Now."])

            uploaded_file = st.file_uploader("upload your database", type="sqlite")     # upload database !
            if uploaded_file:
                file_name = uploaded_file.name
                if file_name.strip():
                    file_name = file_name.replace(".sqlite", "")
                    file_dir = os.path.join(settings.PRIVATE_SQLITE_DATABASE_PATH, file_name)
                    if not (os.path.exists(file_dir)):
                        os.mkdir(file_dir)
                    file_path = os.path.join(file_dir, file_name + ".sqlite")
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())              # write path : data/usr/database/name/name.sqlite
                    try:
                        db_list_resp = requests.post(DB_LIST_URL,
                                                     data=json.dumps({"db_source": st.session_state["db_source"]}))     # refresh db list
                        db_list = json.loads(db_list_resp.text)["data"]
                    except json.JSONDecodeError as e:
                        st.error(f"Error decoding JSON: {e}")
                        db_list = []
                    _ = requests.post(DB_UPDATE_URL, data=json.dumps({"db": file_name}))        # update db to backend and modelend
                    
            # Not publicable for the url upload function.
            # link_file = st.text_input(
            #     ".",
            #     label_visibility = "hidden",
            #     placeholder = "Optional Database Url: //"
            # )
            db, delete_com = st.columns([0.90, 0.1])
            db = db.selectbox("select your database", db_list)
            with delete_com:        # add delete operator
                st.write("")
                st.write("")
                delete = st.button('**X**', type="primary")
                if delete:
                    resp = requests.post(DB_DELETE_URL, data=json.dumps({"db": db}))  # delete db
                    st.rerun()

        if db:
            if specified_db is not None:
                db = specified_db
            db_tree_resp = requests.post(DB_TREE_URL,
                                         data=json.dumps({"db": db, "db_source": st.session_state["db_source"]}))
            db_schema = json.loads(db_tree_resp.text)["data"]
            table_name = []
            column_names = []
            for i, unit in enumerate(db_schema):
                table_name.append(unit["tableName"])
                column_names.append(unit["tableColumns"])
            for i, name in enumerate(table_name):  # version: stoggle
                col_names = column_names[i]
                col_names_str = """"""
                head = '&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;<span style="color: rgb(255, 99, 71);">'
                tail = '</span>&nbsp; &nbsp; &nbsp; &nbsp; Column <br>'
                all_schemas.append(name.lower() + "." + "*")
                for col in col_names:
                    all_schemas.append(name.lower() + "." + col.lower())
                    col_names_str = col_names_str + head + col.lower() + tail
                stoggle(
                    ' <span style="color: rgb(30, 144, 255);">{}</span>&nbsp; &nbsp; &nbsp; &nbsp; Table'.format(
                        name.lower()),
                    col_names_str)

    if st.session_state["db_source"] == settings.PRIVATE_DATABASE:
        
        CKPT_LIST_URL = settings.CKPT_LIST_URL
        CKPT_UPDATE_URL = settings.CKPT_UPDATE_URL
        ADAPTIVE_RETRAINING_BASE_CHECKPOINTS_PATH = settings.ADAPTIVE_RETRAINING_BASE_CHECKPOINTS_PATH
        def get_ckpt_list():
            try:
                ckpt_list_resp = requests.get(CKPT_LIST_URL)
                ckpt_list = json.loads(ckpt_list_resp.text)["data"]
            except json.JSONDecodeError as e:
                st.error(f"Error decoding JSON: {e}")
                ckpt_list = []
            return ckpt_list
            
        if 'checkpoints' not in st.session_state:
            st.session_state['checkpoints'] = get_ckpt_list()
    
        selected_checkpoint = st.selectbox("Select Finetuned checkpoints", st.session_state['checkpoints'])
        
        
        
        refresh_ckpts_btn, update_ckpts_btn  = st.columns([0.5, 0.5])
        
        
        with refresh_ckpts_btn:
            if st.button('Refresh Ckpts'):
                st.session_state['checkpoints'] = get_ckpt_list()
        
        with update_ckpts_btn:
            if st.button('Update Parser', type="primary"):
                try:
                    update_resp = requests.post(CKPT_UPDATE_URL, json={'selected_ckpt_path': ADAPTIVE_RETRAINING_BASE_CHECKPOINTS_PATH + "/" + selected_checkpoint})
                    st.success(update_resp.text)
                except requests.RequestException as e:
                    st.error(f"Error updating checkpoint: {e}")

    return db, all_schemas
