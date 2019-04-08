import re

IDENTIFIER_PATTERN = '[a-zA-Z]([-a-zA-Z0-9])*'
IDENTIFIER_COMPILED_PATTERN = re.compile(IDENTIFIER_PATTERN)


def is_identifier(name):
    return IDENTIFIER_COMPILED_PATTERN.fullmatch(name) is not None


# Gets the value of the operand, searching in the dict counters.
# The operand must be either an identifier (counter) or an integer

def get_value(operand, counters):
    if type(operand).__name__ == 'int':
        return operand
    # If the operand is an identifier, gets its current value, or default to
    # zero if not present yet
    elif is_identifier(operand):
        # Returns the current value of the counter, or 0 if doesn' exist yet
        result = counters.get(operand)
        return 0 if result is None else len(result)
