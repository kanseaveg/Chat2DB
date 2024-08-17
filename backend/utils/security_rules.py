"""
    Add some security check filtering rules for the question here.
    Here mainly focuses on the inspection of unsafe SQL queries carried in the question.
    All rules are derived from common safety rules in life and papers.
    On the Security Vulnerabilities of Text-to-SQL Models: https://arxiv.org/pdf/2211.15363.pdf
"""

QUESTION_SECURITY_CHECK_RULES = [
    'SELECT user()',
    'SELECT current_user()',
    'SELECT @@version',
    'SELECT database()',
    'DROP database()',
    'DROP table()',
    'DROP column()',
    'DROP index()',
    'DROP view()',
    'DROP user()',
    'DROP function()',
    'DROP procedure()',
    'DROP trigger()',
    'UNION SELECT user()',
]


DATABASE_SECURITY_CHECK_RULES = QUESTION_SECURITY_CHECK_RULES + [
    '\' UNION SELECT user()',
    '\' UNION SELECT current_user()',
    '\' UNION SELECT @@version',
    '\' UNION SELECT database()',
    '\' UNION DROP database()',
    '\' UNION DROP table()',
    '\' UNION DROP column()',
    '\' UNION DROP index()',
    '\' UNION DROP view()',
    '\' UNION DROP user()',
    '\' UNION DROP function()',
    '\' UNION DROP procedure()',
    '\' UNION DROP trigger()',
    'SELECT user() FROM',
    'SELECT current_user() FROM',
    'SELECT @@version FROM',
    'SELECT database() FROM',
    'DROP database() FROM',
    'DROP table() FROM',
    'DROP column() FROM',
    'DROP index() FROM',
    'DROP view() FROM',
    'DROP user() FROM',
    'DROP function() FROM',
    'DROP procedure() FROM',
    'DROP trigger() FROM',
]