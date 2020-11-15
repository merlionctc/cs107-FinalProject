import numpy as np

class Dual():
    def __init__(self, val, der, **kwargs):
        self.val = val
        if kwargs:
            self.length = kwargs["length"]
            self.loc = kwargs["loc"]
            self.der = np.zeros(self.length)
            self.der[self.loc] = der
        else:
            self.der = der
        
    ### dunder methods###
    def __repr__(self):
        return "{class_name}(value={value}, derivative={der})".format(class_name=type(self).__name__, value=self.val, der=self.der)
        
    ### dunder method of math operation###
    def __pos__(self):
        return Dual(self.val, self.der)

    def __neg__(self):
        return Dual(-self.val,-self.der)

    def __add__(self, other):
        try:
            val = self.val + other.val
            der = self.der + other.der
        except AttributeError:
            val = self.val + other
            der = self.der
        return Dual(val,der)   

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        try:
            val = self.val - other.val
            der = self.der - other.der
        
        except AttributeError:
            val = self.val - other
            der = self.der
        
        return Dual(val,der)

    def __rsub__(self,other):
        #try:
            #val = -self.val + other.val
            #der = -self.der + other.der
        
        #except AttributeError:
        val = -self.val + other
        der = - self.der
        
        return Dual(val,der)

    def __mul__(self, other):
        try:
            val = self.val * other.val
            der = self.val * other.der + self.der * other.val
        except AttributeError: # real number
            val = self.val * other
            der = self.der * other
        return Dual(val, der)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        try:
            val = self.val/other.val
            der = (self.der*other.val-self.val*other.der)/other.val**2
        except AttributeError:
            val = (self.val*other)/other**2
            der = (self.der*other)/other**2
        return Dual(val, der)

    def __rtruediv__(self, other):
        #try:
            #val = (self.val*other.val)/other.val**2
            #der = (-self.der*other.val )/self.val**2
        #except AttributeError:
        val = (self.val*other)/self.val**2
        der = (-self.der*other)/self.val**2
        return Dual(val, der)

    # exponentiate a dual number by a real or dual number, self is dual, other is dual or real

    def __pow__(self, other):
        try:
            # da^u/dx = ln(a) a^u du/dx
            factor = self.val ** (other.val -1)
            sum_1 = other.val * self.der
            sum_2 = np.log(self.val) * self.val * other.der
            temp = factor * (sum_1 + sum_2)
            return Dual(self.val ** other.val, temp)
        
        except AttributeError:
            # du^n/dx = n * u^(n-1) * du/dx
            temp = other * self.val ** (other-1) * self.der
            return Dual(self.val ** other, temp)
    
    # exponentiate a real by a real or dual
    def __rpow__(self, other):
        temp = np.log(other) * other ** self.val * self.der
        return Dual(other ** self.val, temp)

from elementary import *
