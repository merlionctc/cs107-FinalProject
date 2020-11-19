 #elementary functions
import numpy as np

def exp(dual):
    """Calculate the exponential of the input 
        Keyword arguments:
        dual -- a dual number or a real number
        Return:
        the exponential value
    """
    if isinstance(dual, Dual):
        val = np.e**dual.val
        der = np.e**dual.val * dual.der
        return Dual(val,der)
    else:
        return np.e**dual

def sin(dual):
    """Calculate the sine of the input
        Keyword arguments:
        dual -- a dual number or a real number
        Return:
        the sine value
    """
    if isinstance(dual, Dual):
        der = np.cos(dual.val)*dual.der
        val = np.sin(dual.val)
        return Dual(val,der)
    else:
        return np.sin(dual)
             
def cos(dual):
    """Calculate the cosine of the input
        Keyword arguments:
        dual -- a dual number or a real number
        Return:
        the cosine value
    """
    if isinstance(dual, Dual):
        der = -1 * np.sin(dual.val)*dual.der
        val = np.cos(dual.val)
        return Dual(val,der)
    else:
        return np.cos(dual)
        
def tan(dual):
    """Calculate the tangent of the input
        Keyword arguments:
        dual -- a dual number or a real number
        Return:
        the tangent value
    """
    if isinstance(dual, Dual):
        der = 1/np.cos(dual.val)**2*dual.der
        val = np.tan(dual.val)
        return Dual(val,der)
    else:
        return np.tan(dual)

def log(dual):
    """Calculate the natural log of the input
        Keyword arguments:
        dual -- a dual number or a real number
        Return:
        the natural log value
    """
    if isinstance(dual,Dual):
        val = np.log(dual.val)
        der = 1/dual.val * dual.der
        return Dual(val,der)
    else:
        return np.log(dual)

def sqrt(dual):
    """Calculate the square root of the input
        Keyword arguments:
        dual -- a dual number or a real number
        Return:
        the square root value
    """
    if isinstance(dual,Dual):
        der = 0.5/np.sqrt(dual.val) * dual.der
        val = np.sqrt(dual.val)
        return Dual(val,der)
    else:
        return np.sqrt(dual)

def arcsin(dual):
    """Calculate the inverse sine (arcsin or sin^(-1)) of the input
        Keyword arguments:
        dual -- a dual number or a real number
        Return:
        the inverse sine value
    """
    if isinstance(dual,Dual):
        der = 1 / np.sqrt(1 - dual.val **2) * dual.der
        val = np.arcsin(dual.val)
        return Dual(val,der)
    else:
        return np.arcsin(dual)

def arccos(dual):
    """Calculate the inverse cosine (arccos or cos^(-1)) of the input
        Keyword arguments:
        dual -- a dual number or a real number
        Return:
        the inverse cosine value
    """
    if isinstance(dual,Dual):
        der = -1 / np.sqrt(1 - dual.val**2) * dual.der
        val = np.arccos(dual.val)
        return Dual(val,der)
    else:
        return np.arccos(dual)

def arctan(dual):
    """Calculate the inverse tangent (arctan or tan^(-1)) of the input
        Keyword arguments:
        dual -- a dual number or a real number
        Return:
        the inverse tangent value
    """
    if isinstance(dual,Dual):
        der = 1 / (1 + dual.val**2) * dual.der
        val = np.arctan(dual.val)
        return Dual(val,der)
    else:
        return np.arctan(dual)


from autodiff.dual import *