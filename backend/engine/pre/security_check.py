from ...utils.security_rules import QUESTION_SECURITY_CHECK_RULES

rules = QUESTION_SECURITY_CHECK_RULES + [rule.upper() for rule in QUESTION_SECURITY_CHECK_RULES]


async def pre_check_question_security(text):
    """
    Check if the question contains unsafe SQL.
    :param text: question
    :return: True or False
    """
    for rule in rules:
        if rule in text:
            return False  # Unsafe SQL detected
    return True