import numpy as np

class Node():
    """
	Creates a Node class supporting custom operations for Reverse mode in Automatic Differentiation (AD).
	Attributes
	==========
	val : int or float
		  The value of user defined function(s) f evaluated at x.
	der : int or float,default = 1
		  The initilized corresponding derivative, gradient, or Jacobian of user defined
		  functions(s) on the variable. 
	name : str
		  The name of the node, default as "new".    
	inputs : list consists of node object
		  The list of node variables involved in the calculation
	gradients : list consists of int or float
		  The list storing the partial derivative w.r.t. corresponding node object, because the obtaining the gradients is 
          useful in calculating the previous node in the reverse mode
	"""
    def __init__(self, val, name = "new", der=1):
        """
		INPUTS
		=======
		val : int, float
			  The value of user defined function(s) f evaluated at x.
		name: str, or optional (default = "new")
			  The value of user defined function(s) f evaluated at x.  
		der : int, float, or optional (default=1)
			  The initilized corresponding derivative, gradient, or Jacobian of user defined
		      functions(s) on the variable.

		NOTES
		=====
		PRE: 
			 - val: int, float
             - name: str
			 - der: int, float


		POST:
			 - val: float, int, or array of int/float
             - name: str
			 - der: array, When the input of the operation has multiple variables, 
                the derivatives of each variable will be listed in the array with the order, this is not printed

		EXAMPLES
		=========
        # Single variable for the target function(s)
		>>> x = Node(1) 
        >>> x
		Node(value = 1, name=new)
		# Multiple variables for the target function(s)
        >>> x1 = Node(val = 1)
        >>> y1 = Node(val = 2)
        >>> z1 = Node(val = 5)
        >>> f1 = 3 * x1 + 4 * y1 * 2 - z1
        >>> f1
        Node(value = 14, name = new)
		"""
        self.inputs = []
        self.val = val
        self.der = der
        self.gradients = []
        self.name = name
    
    def __hash__(self):
        return hash(self.gradients.append(str(self.val)+str(self.der)+self.name))
    
    def __repr__(self):
        """ Prints self in the form of Node(value = [val], name = "name")
        
        Parameters
        ----------
        self: Node object
        
        Returns
        ------- 
        z: Node object with val and der if z is a scalar
        z: Node object with val and vector der if z is a combination of vectors
        
        Examples
        -------- 
        >>> z = Node(2)
        >>> print(z)
        Node(value=2, name=new)
        """        
        return "{class_name}(value={value}, name={name})\n".format(class_name=type(self).__name__, value=self.val, name = self.name)
    
    
    def __pos__(self):
        """ Returns the positive of self
        
        Parameters
        ----------
        self: Node object
        
        Returns
        ------- 
        z: Node object that is the positive of self
        
        Examples
        -------- 
        >>> z = + Node(2)
        >>> print(z)
        Node(value=2, name=new)
        """


        new_node = Node(0) #initialization
        new_node.val = self.val
        new_node.inputs = [self]
        new_node.gradients = [1]
        return new_node

    def __neg__(self):
        """ Returns the negative of self
        
        Parameters
        ----------
        self: Node object
        
        Returns
        ------- 
        z: Node object that is the negative of self
        
        Examples
        -------- 
        >>> z = - Node(2, 1)
        >>> print(z)
        Node(value=-2, name=new)
        """
    
        new_node = Node(0) #initialization

        new_node.val = -self.val
        new_node.inputs = [self]
        new_node.gradients = [-1]
        return new_node

    # dunder method for elementary operation
    def __add__(self, other):
        """ Returns the addition of self and other
        
        Parameters
        ----------
        self: Node object
        other: Node object, float, or int
        
        Returns
        ------- 
        z: Node object that is the sum of self and other
        
        Examples
        -------- 
        >>> z = Node(1) + Node(3)
        >>> print(z)
        Node(value=4, name=new)
        >>> z =  Node(1, 2) + 2
        >>> print(z)
        Node(value=3, name=new)
        """
        new_node = Node(0) #initialization
        try:
            new_node.val = self.val + other.val
            new_node.inputs = [self, other]
            new_node.gradients = [1, 1]
        except AttributeError:
            new_node.val = self.val + other
            new_node.inputs = [self]
            new_node.gradients = [1]
        return new_node
    
    def __radd__(self,other):
        """ Returns the addition of other and self
        
        Parameters
        ----------
        self: Node object
        other: Node object, float, or int
        
        Returns
        ------- 
        z: Node object that is the sum of other and self
        
        Examples
        -------- 
        >>> z = Node(1) + Node(3)
        >>> print(z)
        Node(value=4, name=new)
        >>> z =  2 + Node(1)
        >>> print(z)
        Node(value=3, name=new)
        """
        return self.__add__(other)

    def __sub__(self, other):
        """ Returns the subtraction of self and other
        
        Parameters
        ----------
        self: Node object
        other: Node object, float, or int
        
        Returns
        ------- 
        z: Node object that is the subtraction of self and other
        
        Examples
        -------- 
        >>> z = Node(1) - Node(3)
        >>> print(z)
        Node(value=-2, name=new)
        >>> z =  Node(1) - 2
        >>> print(z)
        Node(value=-1, name=new)
        """
        new_node = Node(0)
        try:
            new_node.val = self.val - other.val
            new_node.inputs = [self, other]
            new_node.gradients = [1, -1]
        except AttributeError:
            new_node.val = self.val - other
            new_node.inputs = [self]
            new_node.gradients = [1]
        return new_node

    def __rsub__(self,other):
        """ Returns the subtraction of other and self
        
        Parameters
        ----------
        self: Node object
        other: Node object, float, or int
        
        Returns
        ------- 
        z: Dual object that is the subtraction of other and self
        
        Examples
        -------- 
        >>> z = Node(1) - Node(3)
        >>> print(z)
        Node(value=-2, name=new)
        >>> z =  2 - Node(1)
        >>> print(z)
        Node(value=1, name=new)
        """        
        new_node = Node(0)
        new_node.val = -self.val + other
        new_node.inputs = [self]
        new_node.gradients = [-1]
        return new_node

    def __mul__(self, other):
        #这个函数输出__repr__有问题
        """ Returns the multiplication of self and other
        
        Parameters
        ----------
        self: Node object
        other: Node object, float, or int
        
        Returns
        ------- 
        z: Node object that is the subtraction of self and other
        
        Examples
        -------- 
        >>> z = Node(1) * Node(3)
        >>> print(z)
        Node(value=3, name=new)
        >>> z =  Node(1) * 2
        >>> print(z)
        Node(value=2, name=new)
        """
        new_node = Node(0)
        try:
            new_node.val = self.val * other.val
            new_node.inputs = [self, other]
            new_node.gradients = [other.val, self.val]
        except AttributeError:
            new_node.val = self.val * other
            new_node.inputs = [self]
            new_node.gradients = [other]
        return new_node

    def __rmul__(self, other):
        """ Returns the multiplication of other and self
        
        Parameters
        ----------
        self: Node object
        other: Node object, float, or int
        
        Returns
        ------- 
        z: Node object that is the multiplication of other and self
        
        Examples
        -------- 
        >>> z = Node(1) * Node(3)
        >>> print(z)
        Node(value=1, name=new)
        >>> z =  Node(2) * 2
        >>> print(z)
        Node(value=4, name=new)
        """        
        return self.__mul__(other)

    def __truediv__(self,other):
        """ Returns the devision of self and other
        
        Parameters
        ----------
        self: Node object
        other: Node object, float, or int
        
        Returns
        ------- 
        z: Node object that is the devision of self and other
        
        Examples
        -------- 
        >>> z = Node(1) / Dual(4)
        >>> print(z)
        Node(value=0.25, name=new)
        >>> z =  Node(2) / 2
        >>> print(z)
        Node(value=1, name=new)
        """
        new_node = Node(0)
        try:
            new_node.val = self.val/other.val
            new_node.inputs = [self,other]
            p1 = 1/other.val #partial f wrt self
            p2 = -self.val/other.val**2 #partial f wrt other
            new_node.gradients = [p1,p2]
        except AttributeError:
            new_node.val = self.val/other
            new_node.inputs = [self]
            new_node.gradients = [1/other]
        return new_node

    def __rtruediv__(self, other):
        """ Returns the devision of other and self
        
        Parameters
        ----------
        self: Node object
        other: Node object, float, or int
        
        Returns
        ------- 
        z: Node object that is the devision of other and self
        
        Examples
        -------- 
        >>> z = Node(1) / Dual(4)
        >>> print(z)
        Node(value=0.25, name=new)
        >>> z =  2/Node(2) 
        >>> print(z)
        Dual(value=1, name=new)
        """        
        new_node = Node(0)
        # only discuss f = y/x if y is a real number,x->self, y->other
        new_node.val = other/self.val
        new_node.inputs = [self]
        p1 = -other/self.val**2
        new_node.gradients = [p1]
        return new_node
    
    def __pow__(self, other):
        """ Returns the power of self raised by other
        
        Parameters
        ----------
        self: Node object
        other: Node object, float, or int
        
        Returns
        ------- 
        z: Node object that is the power of self raised by other
        
        Examples
        -------- 
        >>> z = Node(2)** Node(2)
        >>> print(z)
        Node(value=4, name=new)
        >>> z =  Dual(2, 1)**2
        >>> print(z)
        Dual(value=4, name=new)
        """
        new_node = Node(0)
        try:
            new_node.val = self.val ** other.val
            new_node.inputs = [self,other]
            p1 = other.val * self.val**(other.val-1)
            p2 = self.val ** other.val * np.log(self.val)
            new_node.gradients = [p1,p2]
        except AttributeError:
            new_node.val = self.val ** other
            new_node.inputs = [self]
            p1 = other * self.val**(other-1)
            new_node.gradients = [p1]
        return new_node

    def __rpow__(self, other):
        """ Returns the power of other raised by self
        
        Parameters
        ----------
        self: Node object
        other: Node object, float, or int
        
        Returns
        ------- 
        z: Node object that is the power of other raised by self
        
        Examples
        -------- 
        >>> z =  2**Node(2)
        >>> print(z)
        Node(value=4, name=new)
        """
        #only discuss f = y^x, y is not node, x->self, y->other
        new_node = Node(0)
        new_node.val = other**self.val
        new_node.inputs = [self]
        p1 = other**self.val * np.log(other)
        new_node.gradients = [p1]
        return new_node


    # dunder method for comparison
    def __eq__(self, other):
        """Returns boolean if two objects have equal value
        
        Parameters
        ----------
        self: Node object
        other: Node object, float, or int
        
        Returns
        ------- 
        Boolean: True or False
        
        Examples
        -------- 
        >>> x =  Node(1)
        >>> y =  Node(2)
        >>> print(x==y)
        False
        """
        # x is node object, y is node or not
        # comparing value is enough
      
        try:
            return (self.val == other.val)  and (self.gradients ==other.gradients) and (self.der ==other.der)
        except AttributeError:
            return (self.val == other)

    def __ne__(self, other):
        """Returns boolean if two objects DO NOT have equal value
        
        Parameters
        ----------
        self: Node object
        other: Node object, float, or int
        
        Returns
        ------- 
        Boolean: True or False
        
        Examples
        -------- 
        >>> x =  Node(1)
        >>> y =  Node(2)
        >>> print(x!=y)
        True
        """        
        try:
            return not (self.val == other.val)  or not (self.gradients == other.gradients) or not (self.der ==other.der)
        except AttributeError:
            return not (self.val == other)
            
    def __lt__(self, other):
        """Returns boolean if the former object is less than the latter.
        
        Parameters
        ----------
        self: Node object
        other: Node object, float, or int
        
        Returns
        ------- 
        Boolean: True or False
        
        Examples
        -------- 
        >>> x =  Node(2)
        >>> y =  Node(2)
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
        self: Node object
        other: Node object, float, or int
        
        Returns
        ------- 
        Boolean: True or False
        
        Examples
        -------- 
        >>> x =  Node(1)
        >>> y =  Node(2)
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
        self: Node object
        other: Node object, float, or int
        
        Returns
        ------- 
        Boolean: True or False
        
        Examples
        -------- 
        >>> x =  Node(2)
        >>> y =  Node(2)
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
        self: Node object
        other: Node object, float, or int
        
        Returns
        ------- 
        Boolean: True or False
        
        Examples
        -------- 
        >>> x =  Node(2)
        >>> y =  Node(1)
        >>> print(x>= y)
        True
        """
        try:
            return (self.val >= other.val)
        except AttributeError:
            return (self.val >= other)

#from autodiff.elementary import *











    

    
    