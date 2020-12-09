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
    >>> exp()
    """
    if isinstance(var, Dual):
        val = np.e ** var.val
        der = np.e ** var.val * var.der
        return Dual(val, der)

    elif isinstance(var, Node):
        new_node = Node(0)
        new_node.inputs = [var]
        new_node.val = np.e ** var.val
        new_node.gradients = [np.e ** var.val]
        return new_node

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
    >>> sin()
    """
    if isinstance(var, Dual):
        der = np.cos(var.val) * var.der
        val = np.sin(var.val)
        return Dual(val, der)

    elif isinstance(var, Node):
        new_node = Node(np.sin(var.val))
        new_node.inputs = [var]
        new_node.gradients = [np.cos(var.val)]
        return new_node

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
    >>> cos()
    """
    if isinstance(var, Dual):
        der = -1 * np.sin(var.val) * var.der
        val = np.cos(var.val)
        return Dual(val, der)

    elif isinstance(var, Node):
        new_node = Node(np.cos(var.val))
        new_node.inputs = [var]
        new_node.gradients = [-np.sin(var.val)]
        return new_node

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
    >>> tan()
    """
    if isinstance(var, Dual):
        der = 1 / np.cos(var.val) ** 2 * var.der
        val = np.tan(var.val)
        return Dual(val, der)

    elif isinstance(var, Node):
        new_node = Node(np.tan(var.val))
        new_node.inputs = [var]
        new_node.gradients = [1 / np.cos(var.val) ** 2]
        return new_node

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
    >>> log()
    """
    if isinstance(var, Dual):
        val = np.log(var.val)
        der = 1 / var.val * var.der
        return Dual(val, der)

    elif isinstance(var, Node):
        new_node = Node(np.log(var.val))
        new_node.inputs = [var]
        new_node.gradients = [1 / var.val]
        return new_node

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
    >>> logb(<var>,base)
    """
    if isinstance(var, Dual):
        val = np.log(var.val) / np.log(base)
        der = (1 / var.val / np.log(base)) * var.der
        return Dual(val, der)

    elif isinstance(var, Node):
        new_node = Node(np.log(var.val) / np.log(base))
        new_node.inputs = [var]
        new_node.gradients = [1 / var.val / np.log(base)]
        return new_node

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
    >>> sqrt()
    """
    if isinstance(var, Dual):
        der = 0.5 / np.sqrt(var.val) * var.der
        val = np.sqrt(var.val)
        return Dual(val, der)

    elif isinstance(var, Node):
        new_node = Node(np.sqrt(var.val))
        new_node.inputs = [var]
        new_node.gradients = [0.5 / np.sqrt(var.val)]
        return new_node

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
    >>> arcsin()
    """
    if isinstance(var, Dual):
        der = 1 / np.sqrt(1 - var.val ** 2) * var.der
        val = np.arcsin(var.val)
        return Dual(val, der)

    elif isinstance(var, Node):
        new_node = Node(np.arcsin(var.val))
        new_node.inputs = [var]
        new_node.gradients = [1 / np.sqrt(1 - var.val ** 2)]
        return new_node

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
    >>> arccos()
    """
    if isinstance(var, Dual):
        der = -1 / np.sqrt(1 - var.val ** 2) * var.der
        val = np.arccos(var.val)
        return Dual(val, der)

    elif isinstance(var, Node):
        new_node = Node(np.arccos(var.val))
        new_node.inputs = [var]
        new_node.gradients = [-1 / np.sqrt(1 - var.val ** 2)]
        return new_node

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
    >>> arctan()
    """
    if isinstance(var, Dual):
        der = 1 / (1 + var.val ** 2) * var.der
        val = np.arctan(var.val)
        return Dual(val, der)

    elif isinstance(var, Node):
        new_node = Node(np.arctan(var.val))
        new_node.inputs = [var]
        new_node.gradients = [1 / (1 + var.val ** 2)]
        return new_node

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
    >>> sinh()
    """
    if isinstance(var, Dual):
        der = np.cosh(var.val) * var.der
        val = np.sinh(var.val)
        return Dual(val, der)

    elif isinstance(var, Node):
        new_node = Node(np.sinh(var.val))
        new_node.inputs = [var]
        new_node.gradients = [np.cosh(var.val)]
        return new_node

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
    >>> cosh()
    """
    if isinstance(var, Dual):
        der = np.sinh(var.val) * var.der
        val = np.cosh(var.val)
        return Dual(val, der)

    elif isinstance(var, Node):
        new_node = Node(np.cosh(var.val))
        new_node.inputs = [var]
        new_node.gradients = [np.sinh(var.val)]
        return new_node

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
    >>> tanh()
    """
    if (isinstance(var, Dual)):
        der = var.der * (np.cosh(var.val) ** 2 - np.sinh(var.val) ** 2) / (np.cosh(var.val) ** 2)
        val = np.tanh(var.val)
        return Dual(val, der)

    elif isinstance(var, Node):
        new_node = Node(np.tanh(var.val))
        new_node.inputs = [var]
        new_node.gradients = [(np.cosh(var.val) ** 2 - np.sinh(var.val) ** 2) / (np.cosh(var.val) ** 2)]
        return new_node

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
    the losgitics value, default to standard: 1/1+e^{-<var>} 
    
    Examples
    -------- 
    >>> logistics()   
    >>> logistics(<var>,L=2,k=1,x0=5 )
    """
    if isinstance(var, Dual):
        temp = help_logistic(var.val, L, k, x0)
        der = k * temp * (1 - temp/L) * var.der
        val = temp
        return Dual(val, der)

    elif isinstance(var, Node):
        temp = help_logistic(var.val, L, k, x0)
        new_node = Node(temp)
        new_node.inputs = [var]
        new_node.gradients = [k * temp * (1 - temp/L)]
        return new_node

    elif isinstance(var, Expression):
        return L / (1 + exp(-k * (var - x0)))

    else:
        return help_logistic(var, L, k, x0)


# import dependencies

from autodiff.dual import *
from autodiff.node import *
from autodiff.symbolic.expression import *
