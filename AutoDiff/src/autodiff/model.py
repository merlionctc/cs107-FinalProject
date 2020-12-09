# base class for autodiff
import numpy as np
class AutoDiff():
    """
	Creates a AutoDiff class as the base class for Automatic Differentiation (AD).
    
	Attributes
	==========
	f : Dual or list
		  target function(s)
	"""
    def __init__(self, f):
        '''function to differentiate'''
        if isinstance(f, list):
            self.f = f
        else:
            self.f = [f]

# child class inherits from autodiff
class Forward(AutoDiff):
    """
	Creates a Forward AutoDiff class for forward mode Automatic Differentiation (AD) .
    
	Attributes 
	==========
	f : Dual or list
		  target function(s)
	"""

    def __init__(self, f):
        """ Initialize a forward AD object
        
        Parameters
        ----------
        self: Forward object
        f: function of variables
        
        Returns
        ------- 
        z: an AutoDiff object that uses Forward Method
        
        Examples
        -------- 
        >>> fwd = Forward(x1**2 + x2)
        """
        super().__init__(f)

    def get_value(self):
        """ Returns the value of f
        
        Parameters
        ----------
        self: Forward object
        
        Returns
        ------- 
        calculate the value of f on var through forward mode on specific value
        
        Examples
        -------- 
        >>> x = Dual(2, 1, loc = 0, length = 2)
        >>> y = Dual(3, 1, loc = 1, length = 2)
        >>> f = 2*x + y
        >>> fwd = Forward(f)
        >>> fwd.get_value()
        [7]
        """
        return [i.val for i in self.f]

    def get_der(self, *args):
        """ Returns the derivative value of f
        
        Parameters
        ----------
        self: Forward object
        **kwargs: x, y etc. Dual Number

        Returns
        ------- 
        calculate the derivative of f on var through forward mode on all variables
        calculate the derivative of f on var through forward mode on specific variables
        
        Examples
        -------- 
        >>> x = Dual(2, 1, loc = 0, length = 2)
        >>> y = Dual(3, 1, loc = 1, length = 2)
        >>> f = 2*x + y
        >>> fwd = Forward(f)
        >>> fwd.get_der()
        [2, 1]
        >>> fwd.get_der(x)
        [2]
        """
        # get jacobian/partial derivatives for all functions from self.f
        result = self.get_jacobian()
        if args:
            selected_result = None
            for i in args:
                if selected_result is None:
                    selected_result = result[:,i.loc:i.loc+1]
                else:
                    selected_result = np.append(selected_result, result[:,i.loc:i.loc+1], axis=1)
            return selected_result
        else:
            return result
     
    def get_jacobian(self):
        """ Returns the Jacobian matrix of f list
        
        Parameters
        ----------
        self: Forward object

        Returns
        ------- 
        calculate the jacobian matrix of f list on all vars through forward mode on all variables
        
        Examples
        -------- 
        >>> x = Dual(2, 1, loc = 0, length = 2)
        >>> y = Dual(3, 1, loc = 1, length = 2)
        >>> f1 = 2*x + y
        >>> f2 = x + 2*y
        >>> fwd = Forward(f)
        >>> fwd.get_jacobian()
        [[2, 1], [1, 2]]
        """
        return np.array([i.der for i in self.f])



