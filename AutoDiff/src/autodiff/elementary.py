# elementary functions

## Exponentials with natural base
## __pow__ can handle different bases 

def exp(var):
    """Calculate the exponential of the input 

    Parameters
    ----------
    var: Dual, Node, Expression, or real number
        
    Returns
    ------- 
    the exponential e^{<var>} 
        
    Examples
    -------- 
    >>> x = Dual(1, 1)
    >>> assert exp(x).val == np.e
    True
    >>> assert exp(x).der == np.e
    True
    >>> x = symbols('x')
    >>> val = {x: 1}
    >>> assert exp(x).evaluate(val) == np.e
    True
    >>> assert diff(exp(x), x).evaluate(val) == np.e
    True

    """
    if isinstance(var, Dual):
        val = np.e ** var.val
        der = np.e ** var.val * var.der
        return Dual(val, der)


    elif isinstance(var, Expression):
        return PowerExpression(var)

    else:
        return np.e ** var


## Trig functions (at the very least, you must have sine, cosine, tangent)

def sin(var):
    """Calculate the sine of the input 

    Parameters
    ----------
    var: Dual, Node, Expression, or real number
        
    Returns
    ------- 
    the sine value: sin{<var>} 
        
    Examples
    -------- 
    >>> x = Dual(np.pi, 1)
    >>> assert sin(x).val == 0
    True
    >>> assert sin(x).der == -1
    True
    >>> x = symbols('x')
    >>> val = {x: np.pi}
    >>> assert sin(x).evaluate(val) == 0
    True
    >>> assert diff(sin(x), x).evaluate(val) == -1
    True
    """
    if isinstance(var, Dual):
        der = np.cos(var.val) * var.der
        val = np.sin(var.val)
        return Dual(val, der)

    elif isinstance(var, Expression):
        return SinExpression(var)

    else:
        return np.sin(var)


def cos(var):
    """Calculate the cos of the input 

    Parameters
    ----------
    var: Dual, Node, Expression, or real number
        
    Returns
    ------- 
    the cosine value: sin{<var>} 
        
    Examples
    -------- 
    >>> x = Dual(np.pi, 1)
    >>> assert cos(x).val == -1
    True
    >>> assert cos(x).der == 0
    True
    >>> x = symbols('x')
    >>> val = {x: np.pi}
    >>> assert cos(x).evaluate(val) == -1
    True
    >>> assert diff(cos(x), x).evaluate(val) == 0
    True
    """
    if isinstance(var, Dual):
        der = -1 * np.sin(var.val) * var.der
        val = np.cos(var.val)
        return Dual(val, der)

    elif isinstance(var, Expression):
        return CosExpression(var)

    else:
        return np.cos(var)


def tan(var):
    """Calculate the tangent of the input 

    Parameters
    ----------
    var: Dual, Node, Expression, or real number
        
    Returns
    ------- 
    the tangent value: tan{<var>} 
        
    Examples
    -------- 
    >>> x = Dual(np.pi, 1)
    >>> assert tan(x).val == np.tan(np.pi)
    True
    >>> assert tan(x).der == 1 / np.cos(x.val) ** 2
    >>> x = symbols('x')
    >>> val = {x: np.pi}
    >>> assert tan(x).evaluate(val) == np.tan(np.pi)
    True
    >>> assert diff(tan(x), x).evaluate(val) == 1 / np.cos(np.pi) ** 2
    True
    """
    if isinstance(var, Dual):
        der = 1 / np.cos(var.val) ** 2 * var.der
        val = np.tan(var.val)
        return Dual(val, der)

    elif isinstance(var, Expression):
        return TanExpression(var)

    else:
        return np.tan(var)


## Logarithms, should be able to handle different bases
def log(var):
    """Calculate the natural log of the input 

    Parameters
    ----------
    var: Dual, Node, Expression, or real number
        
    Returns
    ------- 
    the log value with natural number base value: log_e{<var>} 
        
    Examples
    -------- 
    >>> x = Dual(1, 1)
    >>> assert log(x).val == 0
    True
    >>> assert log(x).der == 1
    True
    >>> x = symbols('x')
    >>> val = {x: 1}
    >>> assert log(x).evaluate(val) == 0
    True
    >>> assert diff(log(x), x).evaluate(val) == 1
    True
    """
    if isinstance(var, Dual):
        val = np.log(var.val)
        der = 1 / var.val * var.der
        return Dual(val, der)


    elif isinstance(var, Expression):
        return LnExpression(var)

    else:
        return np.log(var)


def logb(var, base):
    """Calculate the log of the input, with any base

    Parameters
    ----------
    var: Dual, Node, Expression, or real number
        
    Returns
    ------- 
    the log value with any base: log_b{<var>} 
        
    Examples
    -------- 
    >>> x = Dual(10, 1)
    >>> assert logb(x, 10).val == 1
    True
    >>> assert logb(x, 10).der == 1 / 10 / np.log(10)
    True
    >>> x = symbols('x')
    >>> val = {x: 10}
    >>> assert logb(x, 10).evaluate(val) == -1
    True
    >>> assert diff(logb(x, 10), x).evaluate(val) == 1 / 10 / np.log(10)
    True
    """
    if isinstance(var, Dual):
        val = np.log(var.val) / np.log(base)
        der = (1 / var.val / np.log(base)) * var.der
        return Dual(val, der)


    elif isinstance(var, Expression):
        return LnExpression(var) / make_ln_expression(base)

    else:
        return np.log(var) / np.log(base)


## Square root
def sqrt(var):
    """Calculate the square root of the input 

    Parameters
    ----------
    var: Dual, Node, Expression, or real number
        
    Returns
    ------- 
    the sqaure root value: sqrt{<var>} or {<var>}^{0.5}
        
    Examples
    -------- 
    >>> x = Dual(4, 1)
    >>> assert sqrt(x).val == 2
    True
    >>> assert sqrt(x).der == 0.25
    True
    >>> x = symbols('x')
    >>> val = {x: 4}
    >>> assert sqrt(x).evaluate(val) == 2
    True
    >>> assert diff(sqrt(x), x).evaluate(val) == 0.5 / 2
    True
    """
    if isinstance(var, Dual):
        der = 0.5 / np.sqrt(var.val) * var.der
        val = np.sqrt(var.val)
        return Dual(val, der)

    elif isinstance(var, Expression):
        return PowerExpression(exponent=Constant(1 / 2), base=var)

    else:
        return np.sqrt(var)


## Inverse trig functions (e.g. arcsine, arccosine, arctangent)
def arcsin(var):
    """Calculate the inverse sine of the input 

    Parameters
    ----------
    var: Dual, Node, Expression, or real number
        
    Returns
    ------- 
    the inverse sine value: arcsin{<var>} or sin^{-1}{<var>}
        
    Examples
    -------- 
    >>> x = Dual(2, 1)
    >>> assert arcsin(x).val == np.arcsin(2)
    True
    >>> assert arcsin(x).der == 1 / np.sqrt(1 - 2** 2)
    True
    >>> x = symbols('x')
    >>> val = {x: 2}
    >>> assert arcsin(x).evaluate(val) == np.arcsin(2)
    True
    >>> assert diff(arcsin(x), x).evaluate(val) == 1 / np.sqrt(1 - 2** 2)
    True
    """
    if isinstance(var, Dual):
        der = 1 / np.sqrt(1 - var.val ** 2) * var.der
        val = np.arcsin(var.val)
        return Dual(val, der)

    elif isinstance(var, Expression):
        return ArcsinExpression(var)

    else:
        return np.arcsin(var)


def arccos(var):
    """Calculate the inverse cosine of the input 

    Parameters
    ----------
    var: Dual, Node, Expression, or real number
        
    Returns
    ------- 
    the inverse cosine value: arccos{<var>} or cos^{-1}{<var>}
        
    Examples
    -------- 
    >>> x = Dual(2, 1)
    >>> assert arccos(x).val == np.arccos(2)
    True
    >>> assert arccos(x).der == -1 / np.sqrt(1 - 2** 2)
    True
    >>> x = symbols('x')
    >>> val = {x: 2}
    >>> assert arccos(x).evaluate(val) == np.arccos(2)
    True
    >>> assert diff(arccos(x), x).evaluate(val) == -1 / np.sqrt(1 - 2** 2)
    True
    """
    if isinstance(var, Dual):
        der = -1 / np.sqrt(1 - var.val ** 2) * var.der
        val = np.arccos(var.val)
        return Dual(val, der)

    elif isinstance(var, Expression):
        return ArccosExpression(var)

    else:
        return np.arccos(var)


def arctan(var):
    """Calculate the inverse tangent of the input 

    Parameters
    ----------
    var: Dual, Node, Expression, or real number
        
    Returns
    ------- 
    the inverse tangent value: arctan{<var>} or tan^{-1}{<var>}
        
    Examples
    -------- 
    >>> x = Dual(1, 1)
    >>> assert arctan(x).val == np.arctan(1)
    True
    >>> assert arctan(x).der == 1 / (1 + 1)
    True
    >>> x = symbols('x')
    >>> val = {x: 1}
    >>> assert arctan(x).evaluate(val) == np.arctan(1)
    True
    >>> assert diff(arctan(x), x).evaluate(val) == 1 / (1 + 1)
    True
    """
    if isinstance(var, Dual):
        der = 1 / (1 + var.val ** 2) * var.der
        val = np.arctan(var.val)
        return Dual(val, der)

    elif isinstance(var, Expression):
        return ArctanExpression(var)

    else:
        return np.arctan(var)


## Hyperbolic functions (sinh, cosh, tanh)
def sinh(var):
    """Calculate the hyperbolic sine of the input 

    Parameters
    ----------
    var: Dual, Node, Expression, or real number
        
    Returns
    ------- 
    the hyperbolic sine value: sinh{<var>} 

    Examples
    -------- 
    >>> x = Dual(1, 1)
    >>> assert sinh(x).val == np.sinh(1)
    True
    >>> assert sinh(x).der == np.cosh(1)
    True
    >>> x = symbols('x')
    >>> val = {x: 1}
    >>> assert sinh(x).evaluate(val) == np.sinh(1)
    True
    >>> assert diff(sinh(x), x).evaluate(val) == np.cosh(1)
    True
    """
    if isinstance(var, Dual):
        der = np.cosh(var.val) * var.der
        val = np.sinh(var.val)
        return Dual(val, der)

    elif isinstance(var, Expression):
        return SinhExpression(var)

    else:
        return np.sinh(var)


def cosh(var):
    """Calculate the hyperbolic cosine of the input 

    Parameters
    ----------
    var: Dual, Node, Expression, or real number
        
    Returns
    ------- 
    the hyperbolic cosine value: cosh{<var>} 

    Examples
    -------- 
    >>> x = Dual(1, 1)
    >>> assert cosh(x).val == np.cosh(1)
    True
    >>> assert cosh(x).der == np.sinh(1)
    True
    >>> x = symbols('x')
    >>> val = {x: 1}
    >>> assert cosh(x).evaluate(val) == np.cosh(1)
    True
    >>> assert diff(cosh(x), x).evaluate(val) == np.sinh(1)
    True
    """
    if isinstance(var, Dual):
        der = np.sinh(var.val) * var.der
        val = np.cosh(var.val)
        return Dual(val, der)

    elif isinstance(var, Expression):
        return CoshExpression(var)

    else:
        return np.cosh(var)


def tanh(var):
    """Calculate the hyperbolic tangent of the input 

    Parameters
    ----------
    var: Dual, Node, Expression, or real number
        
    Returns
    ------- 
    the hyperbolic tangent value: tanh{<var>} 

    Examples
    -------- 
    >>> x = Dual(1, 1)
    >>> assert tanh(x).val == np.tanh(1)
    True
    >>> assert tanh(x).der == (np.cosh(1) ** 2 - np.sinh(1) ** 2) / (np.cosh(1) ** 2)
    True
    >>> x = symbols('x')
    >>> val = {x: 1}
    >>> assert tanh(x).evaluate(val) == np.tanh(1)
    True
    >>> assert diff(tanh(x), x).evaluate(val) == (np.cosh(1) ** 2 - np.sinh(1) ** 2) / (np.cosh(1) ** 2)
    True
    """
    if (isinstance(var, Dual)):
        der = var.der * (np.cosh(var.val) ** 2 - np.sinh(var.val) ** 2) / (np.cosh(var.val) ** 2)
        val = np.tanh(var.val)
        return Dual(val, der)

    elif isinstance(var, Expression):
        return TanhExpression(var)

    else:
        return np.tanh(var)


## Logistic function
# https://en.wikipedia.org/wiki/Logistic_function

def help_logistic(x, L=1, k=1, x0=0):
    """Helper function in generating logistics function
    
    Parameters
    ----------
    x: Dual, Node, Expression, or real number
    L: the curve's maximum value, default = 1
    k: the logistic growth rate or steepness of the curve, default = 1
    x0: the x value of the sigmoid's midpoint, default = 0
        
    Returns
    ------- 
    the losgitics value, default to standard: tanh{<var>} 

    Examples
    -------- 
    >>> help_logistic(x)
    0.7310585786300049
    """
    return L / (1 + np.exp(-k * (x - x0)))


def logistic(var, L=1, k=1, x0=0):
    """Calculate the losgistic value of the input, default is the standard logistics function 
    
    Parameters
    ----------
    var: Dual, Node, Expression, or real number
    L: the curve's maximum value, default = 1
    k: the logistic growth rate or steepness of the curve, default = 1
    x0: the x value of the sigmoid's midpoint, default = 0
        
    Returns
    ------- 
    the logitics value, default to standard: 1/1+e^{-<var>} 
    
    Examples
    -------- 
    >>> x = Dual(1, 1)
    >>> temp = help_logistic(1, L=2,k=1,x0=5)
    >>> logistic(x,L=2,k=1,x0=5 ).val
    0.03597241992418312
    >>> assert logistic(x,L=2,k=1,x0=5 ).der == temp * (1 - temp/2)
    True
    >>> x = symbols('x')
    >>> val = {x: 1}
    >>> logistic(x,L=2,k=1,x0=5 ).evaluate(val)
    0.03597241992418312
    >>> assert diff(logistic(x,L=2,k=1,x0=5 ), x).evaluate(val) == temp * (1 - temp/2)
    True
    >>> logistic(1,L=2,k=1,x0=5 )
    0.03597241992418312
    """
    if isinstance(var, Dual):
        temp = help_logistic(var.val, L, k, x0)
        der = k * temp * (1 - temp/L) * var.der
        val = temp
        return Dual(val, der)

    elif isinstance(var, Expression):
        return L / (1 + exp(-k * (var - x0)))

    else:
        return help_logistic(var, L, k, x0)


# import dependencies

from autodiff.dual import *
#from autodiff.node import *
from autodiff.symbolic.expression import *
