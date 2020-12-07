# base class for autodiff
import numpy as np
class AutoDiff():
    def __init__(self, f):
        '''function to differentiate'''
        if isinstance(f, list):
            self.f = f
        else:
            self.f = [f]

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
        >>> fwd.get_der()
        >>> fwd.get_der(x1)
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
        >>> fwd.get_jacobian()"""
        return np.array([i.der for i in self.f])


# child class inherits from autodiff
class Reverse(AutoDiff):
    def __init__(self, f, var_lst = None):
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
        self.var_lst = var_lst

    def _node_sort(self, node_lst):
        visited = []
        node_order = []
        for node in node_lst:            
            self._dfs_sort(node, visited, node_order)
        return node_order

    def _dfs_sort(self, node, visited, node_order):
        if node in visited:
            return
        visited.append(node)
        for node_in in node.inputs:
            self._dfs_sort(node_in, visited, node_order)
        node_order.append(node)
    
    def _reverse_value(self, out_node):
        return out_node.val

    def _reverse_grad(self, out_node, nodes_lst):
        reverse_order_nodes = reversed(self._node_sort([out_node]))
        visited_node = [out_node]
        for node in reverse_order_nodes:
            for i in range(len(node.inputs)):
                node_in = node.inputs[i]
                node_in_grad = node.gradients[i]*node.der
                if node_in not in visited_node:
                    node_in.der = node_in_grad
                    visited_node.append(node_in)
                else:
                    node_in.der = node_in_grad + node_in.der
        result_grads = [node.der for node in nodes_lst]
        return result_grads

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
        return [self._reverse_value(i) for i in self.f]

    def get_der(self, nodes_lst = None):
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
        # get jacobian/partial derivatives for all functions from self.f
        if nodes_lst is None:
            nodes_lst = self.var_lst
        return np.array([self._reverse_grad(i, nodes_lst) for i in self.f])
     
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
        >>> fwd.get_jacobian()"""
        return self.get_der(self.var_lst)
