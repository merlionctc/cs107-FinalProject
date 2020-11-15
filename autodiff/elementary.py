 #elementary functions
import numpy as np

def exp(dual):
    if isinstance(dual, Dual):
        val = np.e**dual.val
        der = np.e**dual.val * dual.der
        return Dual(val,der)
    else:
        return np.e**dual

def sin(dual):
    if isinstance(dual, Dual):
        der = np.cos(dual.val)*dual.der
        val = np.sin(dual.val)
        return Dual(val,der)
    else:
        return np.sin(dual)
             
def cos(dual):
    if isinstance(dual, Dual):
        der = -1 * np.sin(dual.val)*dual.der
        val = np.cos(dual.val)
        return Dual(val,der)
    else:
        return np.cos(dual)
        
def tan(dual):
    if isinstance(dual, Dual):
        der = 1/np.cos(dual.val)**2*dual.der
        val = np.tan(dual.val)
        return Dual(val,der)
    else:
        return np.tan(dual)

def log(dual):
    if isinstance(dual,Dual):
        val = np.log(dual.val)
        der = 1/dual.val * dual.der
        return Dual(val,der)
    else:
        return np.log(dual)

from dual_class import *