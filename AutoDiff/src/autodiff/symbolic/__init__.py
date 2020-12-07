from __future__ import annotations

from .expression import Symbol, Expression


def symbols(names: str):
    l = names.split()
    if len(l) > 1:
        return (Symbol(name) for name in l)
    return Symbol(names)


def diff(expr: Expression, *args):
    if not isinstance(expr, Expression):
        # If it's a constant (not something wrapped by us), assume it's 0
        return 0
    for symbol in args:
        expr = expr._symdiff(symbol)
    return expr


def get_jacobian_expression(expressions, respect_to_lst):
    return [[diff(i, j) for j in respect_to_lst] for i in expressions]


def get_jacobian_value(expressions, respect_to_lst, values):
    return [[col.evaluate(values) for col in row] for row in get_jacobian_expression(expressions, respect_to_lst)]
