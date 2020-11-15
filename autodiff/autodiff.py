# base class for autodiff
class AutoDiff():
    def __init__(self, f):
        '''function to differentiate
        '''
        self.f = f

        ''' dictionary with {key: value}
            key: variable name
            value: value of the variable
            e.g. : {"x": 5, "y": 6}
        '''
        #self.var_val_dict = var_val_dict

        ''' initialize a dict of dual number for each variable value '''
        #self.dual = {key: Dual(self.var_val_dict[key],1) for key in self.var_val_dict}


# child class inherits from autodiff
class Forward(AutoDiff):
    def __init__(self, f):
        super().__init__(f)

    def get_value(self, *args):
        # args: newsvalue of function
        # calculate the value of f on var through forward mode to specific value
        #if not args:
        #    var = [self.dual[key].val for key in self.dual]
        #else:
        #    var = [v for v in args]
        #result = functools.reduce(self.f, var)
        return self.f.val

    def get_der(self, *args):
        # **kwargs: x, y etc. Dual Number
        # calculate the derivative of f on var through forward mode to specific value
        if args:
            result = []
            for i in args:
                result.append(self.f.der[i.loc])
        else:
            result = self.f.der
        return result
     
    def get_jacobian(self, *args):
        # **kwargs: var = 'all','x','y' etc.
        # calculate the derivative of f on var through forward mode with jacobian
        raise NotImplementedError 
     
    def get_expression(self, *args):
        # **kwargs: var = 'all','x','y' etc.
        # return the list of derivative expression of the parsed formula.
        raise NotImplementedError 