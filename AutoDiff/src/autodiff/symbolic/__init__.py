from .expression import Symbol


def symbols(names: str):
    l = names.split()
    if len(l) > 1:
        return (Symbol(name) for name in l)
    return Symbol(names)


def diff(expression, respect_to):
    if isinstance(expression, float):
        # If it's a constant (not something wrapped by us), assume it's 0
        return 0
    return expression._symdiff(respect_to)


def get_jacobian_expression(expressions, respect_to_lst):
    return [[diff(i, j) for j in respect_to_lst] for i in expressions]


def get_jacobian_value(expressions, respect_to_lst, values):
    return [[diff(i, j).evaluate(values) for j in respect_to_lst] for i in expressions]
