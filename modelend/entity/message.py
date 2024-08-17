class HumanMessage():
    def __init__(self,content):
        self.content = content

class AIMessage():
    def __init__(self, content, answer_type):
        self.content = content
        assert answer_type in ["sql", "ex_result"]
        self.answer_type = answer_type