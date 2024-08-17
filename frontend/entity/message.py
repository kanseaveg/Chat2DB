

class Message():
    def __init__(self,
                 role: str,
                 question: str,
                 answer: str,
                 all_schemas: [],
                 selected_schemas: [],
                 is_sql=False,
                 db=None,
                 is_init=False,
                 reset_key=None,
                 select_box_key=None,
                 exec_key=None,
                 content_key=None,
                 edit_key=None,
                 ):
        self.role = role
        self.question = question
        self.answer = answer
        self.is_sql = is_sql
        self.db = db
        self.is_init = is_init
        self.execute_result = None
        self.exec_key = exec_key
        self.content_key = content_key
        self.edit_key = edit_key
        self.reset_key = reset_key
        self.select_box_key = select_box_key
        self.all_schemas = all_schemas
        self.selected_schemas = selected_schemas
