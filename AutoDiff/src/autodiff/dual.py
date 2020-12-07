import numpy as np

class Dual():
    """
	Creates a Dual class supporting custom operations for Automatic Differentiation (AD).
	Attributes
	==========
	val : int or float
		  The value of user defined function(s) f evaluated at x.
	der : int or float
		  The initilized corresponding derivative, gradient, or Jacobian of user defined
		  functions(s) on the variable. 
	"""
    
    def __init__(self, val, der, **kwargs):
        """
		INPUTS
		=======
		val : int, float
			  The value of user defined function(s) f evaluated at x.
		der : int, float, or optional (default=[1])
			  The initilized corresponding derivative, gradient, or Jacobian of user defined
		  functions(s) on the variable

        optional parameters:
        loc : int
              The location/index of this variable when there are multiple input variables for the target function(s).
        length: int
              The length/number of the total variables that will be input when there are multiple input variables for the target function(s).
		
		NOTES
		=====
		PRE: 
			 - val: int, float
			 - der: int, float

             optional:
             - loc: int
             - length: int
    
		POST:
			 - val: float, int
			 - der: float, int, np.array(When the input of the operation has multiple variables, 
                the derivatives of each variable will be listed in the array with the order identified in loc )
            
		EXAMPLES
		=========
		# Single variable for the target function(s)
		>>> Dual(3, 1) 
		Dual(value = 3, derivative = 1)
		# Multiple variables for the target function(s)
		>>> x = Dual(3, 1, loc = 0, length = 3)
		>>> y = Dual(2, 1, loc = 1, length = 3)
		>>> z = x + y
		>>> z
		Dual(value = 5, derivative = [1, 1])
		"""


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
        """ Prints self in the form of Dual(value = [val], derivative = [der])
        
        Parameters
        ----------
        self: Dual object
        
        Returns
        ------- 
        z: Dual object with val and der if z is a scalar
        z: Dual object with val and vector der if z is a combination of vectors
        
        Examples
        -------- 
        >>> z = Dual(2, 1)
        >>> print(z)
        Dual(value=2, derivative=1)
        """
        return "{class_name}(value={value}, derivative={der})".format(class_name=type(self).__name__, value=self.val, der=self.der)
        
    ### dunder method of math operation###
    def __pos__(self):
        """ Returns the positive of self
        
        Parameters
        ----------
        self: Dual object
        
        Returns
        ------- 
        z: Dual object that is the positive of self
        
        Examples
        -------- 
        >>> z = + Dual(2, 1)
        >>> print(z)
        Dual(value=2, derivative=1)
        """
        return Dual(self.val, self.der)

    def __neg__(self):
        """ Returns the negative of self
        
        Parameters
        ----------
        self: Dual object
        
        Returns
        ------- 
        z: Dual object that is the negative of self
        
        Examples
        -------- 
        >>> z = - Dual(2, 1)
        >>> print(z)
        Dual(value=-2, derivative=-1)
        """
    
        return Dual(-self.val, -self.der)
    

    def __add__(self, other):
        """ Returns the addition of self and other
        
        Parameters
        ----------
        self: Dual object
        other: Dual object, float, or int
        
        Returns
        ------- 
        z: Dual object that is the sum of self and other
        
        Examples
        -------- 
        >>> z = Dual(1, 1) + Dual(3, 1)
        >>> print(z)
        Dual(value=4, derivative= [1, 1])
        >>> z =  Dual(1, 2) + 2
        >>> print(z)
        Dual(value=3, derivative=2)
        """

        try:
            val = self.val + other.val
            der = self.der + other.der
        except AttributeError:
            val = self.val + other
            der = self.der
        return Dual(val,der)   

    def __radd__(self, other):
        """ Returns the addition of other and self
        
        Parameters
        ----------
        self: Dual object
        other: Dual object, float, or int
        
        Returns
        ------- 
        z: Dual object that is the sum of other and self
        
        Examples
        -------- 
        >>> z = Dual(1, 1) + Dual(3, 1)
        >>> print(z)
        Dual(value=4, derivative= [1, 1])
        >>> z = 2 + Dual(1, 2)
        >>> print(z)
        Dual(value=3, derivative=2)
        """
        return self.__add__(other)

    def __sub__(self, other):
        """ Returns the subtraction of self and other
        
        Parameters
        ----------
        self: Dual object
        other: Dual object, float, or int
        
        Returns
        ------- 
        z: Dual object that is the subtraction of self and other
        
        Examples
        -------- 
        >>> z = Dual(1, 1) - Dual(3, 1)
        >>> print(z)
        Dual(value=-2, derivative= [1, -1])
        >>> z =  Dual(1, 2) - 2
        >>> print(z)
        Dual(value=-1, derivative=2)
        """
    
        try:
            val = self.val - other.val
            der = self.der - other.der
        
        except AttributeError:
            val = self.val - other
            der = self.der
        
        return Dual(val,der)

    def __rsub__(self,other):
        """ Returns the subtraction of other and self
        
        Parameters
        ----------
        self: Dual object
        other: Dual object, float, or int
        
        Returns
        ------- 
        z: Dual object that is the subtraction of other and self
        
        Examples
        -------- 
        >>> z = Dual(1, 1) - Dual(3, 1)
        >>> print(z)
        Dual(value=-2, derivative= [1, -1])
        >>> z =  2 - Dual(1, 2)
        >>> print(z)
        Dual(value=1, derivative=-2)
        """
    
        #try:
            #val = -self.val + other.val
            #der = -self.der + other.der
        
        #except AttributeError:
        val = -self.val + other
        der = - self.der
        
        return Dual(val,der)

    def __mul__(self, other):
        """ Returns the multiplication of self and other
        
        Parameters
        ----------
        self: Dual object
        other: Dual object, float, or int
        
        Returns
        ------- 
        z: Dual object that is the subtraction of self and other
        
        Examples
        -------- 
        >>> z = Dual(1, 1) * Dual(3, 1)
        >>> print(z)
        Dual(value=3, derivative= [3, 1])
        >>> z =  Dual(2, 1) * 2
        >>> print(z)
        Dual(value=4, derivative=2)
        """
        
        try:
            val = self.val * other.val
            der = self.val * other.der + self.der * other.val
        except AttributeError: # real number
            val = self.val * other
            der = self.der * other
        return Dual(val, der)

    def __rmul__(self, other):
        """ Returns the multiplication of other and self
        
        Parameters
        ----------
        self: Dual object
        other: Dual object, float, or int
        
        Returns
        ------- 
        z: Dual object that is the multiplication of other and self
        
        Examples
        -------- 
        >>> z = Dual(1, 1) * Dual(3, 1)
        >>> print(z)
        Dual(value=3, derivative= [3, 1])
        >>> z =  Dual(2, 1) * 2
        >>> print(z)
        Dual(value=4, derivative=2)
        """
        return self.__mul__(other)

    def __truediv__(self, other):
        """ Returns the devision of self and other
        
        Parameters
        ----------
        self: Dual object
        other: Dual object, float, or int
        
        Returns
        ------- 
        z: Dual object that is the devision of self and other
        
        Examples
        -------- 
        >>> z = Dual(1, 1) / Dual(3, 1)
        >>> print(z)
        Dual(value=1/3, derivative= [1/3, -1])
        >>> z =  Dual(2, 1) / 2
        >>> print(z)
        Dual(value=1, derivative=1/2)
        """
    
        try:
            val = self.val/other.val
            der = (self.der*other.val-self.val*other.der)/other.val**2
        except AttributeError:
            val = (self.val*other)/other**2
            der = (self.der*other)/other**2
        return Dual(val, der)

    def __rtruediv__(self, other):
        """ Returns the devision of other and self
        
        Parameters
        ----------
        self: Dual object
        other: Dual object, float, or int
        
        Returns
        ------- 
        z: Dual object that is the devision of other and self
        
        Examples
        -------- 
        >>> z = Dual(1, 1) / Dual(3, 1)
        >>> print(z)
        Dual(value=1/3, derivative= [1/3, -1])
        >>> z =  2/Dual(2, 1) 
        >>> print(z)
        Dual(value=1, derivative=2)
        """

        #try:
            #val = (self.val*other.val)/other.val**2
            #der = (-self.der*other.val )/self.val**2
        #except AttributeError:
        val = (self.val*other)/self.val**2
        der = (-self.der*other)/self.val**2
        return Dual(val, der)



    def __pow__(self, other):
        """ Returns the power of self raised by other
        
        Parameters
        ----------
        self: Dual object
        other: Dual object, float, or int
        
        Returns
        ------- 
        z: Dual object that is the power of self raised by other
        
        Examples
        -------- 
        >>> z = Dual(2, 1)** Dual(2, 1)
        >>> print(z)
        Dual(value=4, derivative= [4. 2.77258872])
        >>> z =  Dual(2, 1)**2
        >>> print(z)
        Dual(value=4, derivative=4)
        """
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
    

    def __rpow__(self, other):
        """ Returns the power of other raised by self
        
        Parameters
        ----------
        self: Dual object
        other: Dual object, float, or int
        
        Returns
        ------- 
        z: Dual object that is the power of other raised by self
        
        Examples
        -------- 
        >>> z =  2**Dual(2,1)
        >>> print(z)
        Dual(value=4, derivative= 2.772588722239781)
        """
        temp = np.log(other) * other ** self.val * self.der
        return Dual(other ** self.val, temp)


    # equal dunder method
    def __eq__(self, other):
        """Returns boolean if two objects have equal value
        
        Parameters
        ----------
        self: Dual object
        other: Dual object, float, or int
        
        Returns
        ------- 
        Boolean: True or False
        
        Examples
        -------- 
        >>> x =  Dual(1,1)
        >>> y =  Dual(2,1)
        >>> print(x==y)
        False
        """
        try:
            return (self.val == other.val)
        except AttributeError:
            return (self.val == other)

    def __ne__(self, other):
        """Returns boolean if two objects DO NOT have equal value
        
        Parameters
        ----------
        self: Dual object
        other: Dual object, float, or int
        
        Returns
        ------- 
        Boolean: True or False
        
        Examples
        -------- 
        >>> x =  Dual(1,1)
        >>> y =  Dual(2,1)
        >>> print(x!=y)
        True
        """
        try:
            return not (self.val == other.val)
        except AttributeError:
            return not (self.val == other)

    # comparison dunder method
    def __lt__(self, other):
        """Returns boolean if the former object is less than the latter.
        
        Parameters
        ----------
        self: Dual object
        other: Dual object, float, or int
        
        Returns
        ------- 
        Boolean: True or False
        
        Examples
        -------- 
        >>> x =  Dual(2,1)
        >>> y =  Dual(2,1)
        >>> print(x<y)
        False
        """
        try:
            return (self.val < other.val)
        except AttributeError:
            return (self.val < other)

    def __le__(self, other):
        """Returns boolean if the former object is less than or equal to the latter.
        
        Parameters
        ----------
        self: Dual object
        other: Dual object, float, or int
        
        Returns
        ------- 
        Boolean: True or False
        
        Examples
        -------- 
        >>> x =  Dual(1,1)
        >>> y =  Dual(2,1)
        >>> print(x<=y)
        True
        """
        try:
            return (self.val <= other.val)
        except AttributeError:
            return (self.val <= other)

    def __gt__(self, other):
        """Returns boolean if the former object is greater than the latter.
        
        Parameters
        ----------
        self: Dual object
        other: Dual object, float, or int
        
        Returns
        ------- 
        Boolean: True or False
        
        Examples
        -------- 
        >>> x =  Dual(2,1)
        >>> y =  Dual(2,1)
        >>> print(x>y)
        False
        """
        try:
            return (self.val > other.val)
        except AttributeError:
            return (self.val > other)

    def __ge__(self, other):
        """Returns boolean if the former object is greater than or equal to the latter.
        
        Parameters
        ----------
        self: Dual object
        other: Dual object, float, or int
        
        Returns
        ------- 
        Boolean: True or False
        
        Examples
        -------- 
        >>> x =  Dual(2,1)
        >>> y =  Dual(1,1)
        >>> print(x >= y)
        True
        """
        try:
            return (self.val >= other.val)
        except AttributeError:
            return (self.val >= other)



from autodiff.elementary import *
