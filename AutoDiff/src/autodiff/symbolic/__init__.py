from __future__ import annotations

from .expression import Symbol, Expression


def symbols(names: str):
    l = names.split()
    if len(l) > 1:
        return (Symbol(name) for name in l)
    return Symbol(names)


def diff(expr: Expression, *args):
    """
    differentiate w.r.t. args
    Args:
        expr: Expression class
        *args: variable symbol, x or x,y or x,x for higher order differentiation

    Returns:
        derivative expression

    Examples:
        diff(f, x, x)
        diff(f, x)
    """
    if not isinstance(expr, Expression):
        # If it's a constant (not something wrapped by us), assume it's 0
        return 0
    for symbol in args:
        expr = expr._symdiff(symbol)
    return expr


def get_jacobian_expression(expressions, respect_to_lst):
    """
    get Jacobian expression list
    Args:
        expressions: Expression lst
        respect_to_lst: variable lst w.r.t for jacobian matrix

    Returns:
        Jacobian matrix
    """
    return [[diff(i, j) for j in respect_to_lst] for i in expressions]


def get_jacobian_value(expressions, respect_to_lst, values):
    """
    get Jacobian matrix
    Args:
        expressions: Expression lst
        respect_to_lst: variable lst w.r.t for jacobian matrix
        values: dictionary for values to be evaluated at

    Returns:
        Jacobian matrix

    Examples:
        get_jacobian_value([f3, f4], [x, y, z], {x: math.pi, y: math.pi / 2, z: 0})
    """
    return [[col.evaluate(values) for col in row] for row in get_jacobian_expression(expressions, respect_to_lst)]
