# base class for autodiff
class AutoDiff():
    def __init__(self, f):
        '''function to differentiate'''
        self.f = f

# child class inherits from autodiff
class Forward(AutoDiff):
    def __init__(self, f):
        """ Returns the positive of self
        
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
        >>> fwd.get_value()
        """
        return self.f.val

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
        >>> fwd.get_der()
        >>> fwd.get_der(x1)
        """
        if args:
            result = []
            for i in args:
                result.append(self.f.der[i.loc])
        else:
            result = self.f.der
        return result
     
    def get_jacobian(self, *args):
        ''' jacobian(2*x + 3*y + 4*z, [x, y, z])
            ans =
                    [ 2, 3, 4]'''
        raise NotImplementedError 
     
    def get_expression(self, *args):
        ''' **kwargs: var = 'all','x','y' etc.
        return the list of derivative expression of the parsed formula'''
        raise NotImplementedError 