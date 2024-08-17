from ...utils.security_rules import DATABASE_SECURITY_CHECK_RULES

rules = DATABASE_SECURITY_CHECK_RULES + [rule.upper() for rule in DATABASE_SECURITY_CHECK_RULES]


def post_check_sql_security(text):
    """
    Check if the SQL contains unsafe SQL.
    :param text: SQL
    :return: True or False
    """
    for rule in rules:
        if rule in text:
            return False  # Unsafe SQL detected
    return True