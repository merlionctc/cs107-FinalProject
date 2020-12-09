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
    """
	Creates a Reverse class for reverse mode of Automatic Differentiation (AD).
    
	Attributes
	==========
	var_lst : list
		  The list of the original nodes(variables) for the function(s).
	f : list
		  The list of the targeted functions needed to calculate derivatives
	"""

    def __init__(self, f, var_lst = None):
        """ Returns a Reverse AD object
        
        Parameters
        ----------
        self: Reverse object
        f: list
            function(s) of variables
        var_lst: 
            The list of the original nodes(variables) for the function(s).
        Returns
        ------- 
        z: an AutoDiff object that uses Reverse Method
        
        Examples
        -------- 
        >>> x = Node(1, "x")
        >>> y = Node(2, "y")
        >>> f1 = x + 2*y
        >>> f2 = 2*x + y
        >>> rvs = Reverse([f1, f2], [x, y])
        """
        super().__init__(f)
        self.var_lst = var_lst
        self.grads = np.array([self._reverse_grad(self, i, self.var_lst) for i in self.f])

    def _node_sort(self, node_lst):
        """ Helper function: Returns a topologically sorted list of all the input nodes 
        and traces of the node_lst(output nodes)
        
        Parameters
        ----------
        self: Reverse object
        node_lst: list
            the list of utput nodes
    

        Returns
        ------- 
        node_order: topologically sorted list of all the input nodes 
        and traces of the node_lst(output nodes)
        
        Examples
        -------- 
        >>> x = Node(1, "x")
        >>> y = Node(2, "y")
        >>> f1 = x + 2*y
        >>> rvs = Reverse([f1], [x, y])
        >>> rvs._node_sort([f1])
        [x, y, 2*y, f]
        """
        visited = []
        node_order = []
        for node in node_lst:            
            self._dfs_sort(node, visited, node_order)
        return node_order

    def _dfs_sort(self, node, visited, node_order):
        """ Helper function: Conduct post-order DFS on the current node and its inputs to change the visited and node order
        
        Parameters
        ----------
        self: Reverse object
        node: Node
            current node
        visited: list
            list of visited nodes
        node_order: list
            to be added new nodes in order
    

        Returns
        ------- 
        None: if the current node has been visited
        Change visited and node_order if not visited
        
        Examples
        -------- 
        >>> rvs._dfs_sort(y, [x], [x] )
        """
        if node in visited:
            return
        visited.append(node)
        for node_in in node.inputs:
            self._dfs_sort(node_in, visited, node_order)
        node_order.append(node)
    
    def _reverse_value(self, out_node):
        """ Helper function: Return the value of the final output node
        
        Parameters
        ----------
        out_node: Node
    
        Returns
        ------- 
        the value of the final output node
        
        Examples
        -------- 
        >>> rvs._reverse_value(f1)
        """
        return out_node.val

    def _reverse_grad(self, out_node, nodes_lst):
        """ Helper function: Return the list of gradients of the output node on the given nodes using reverse mode.
        
        Parameters
        ----------
        out_node: Node, the output node of the function
        nodes_lst: list, list of nodes that need calculating gradients
    
        Returns
        ------- 
        result_grads: list
        the gradients of the final output node on the given nodes
        
        Examples
        -------- 
        >>> rvs._reverse_grad(f1, [x,y])
        """
        #get the reversed topological order of all the input nodes and traces of the output node
        reverse_order_nodes = reversed(self._node_sort([out_node]))
        #initialize visited_node as the list of visited nodes
        visited_node = [out_node]
        for node in reverse_order_nodes:
            for i in range(len(node.inputs)):
                node_in = node.inputs[i]
                #for each input node of the current node, get its derivative using chain rule
                node_in_grad = node.gradients[i]*node.der
                if node_in not in visited_node:
                    #if node not visited, directly update the node's derivative
                    node_in.der = node_in_grad
                    visited_node.append(node_in)
                else:
                    #if node visited, sum the gradient distribution
                    node_in.der = node_in_grad + node_in.der
        # get the list of gradients of the given nodes
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
        """ Returns the derivative value of f(s)
        
        Parameters
        ----------
        self: Reverse object
        nodes_lst: the list of nodes that need calculating derivatives

        Returns
        ------- 
        calculate the derivative of f(s) on var through reverse mode on all variables
        calculate the derivative of f(s) on var through reverse mode on specific variables
        
        Examples
        -------- 
        >>> x = Node(1, "x")
        >>> y = Node(2, "y")
        >>> f1 = x + 2*y
        >>> rvs = Reverse([f1], [x, y])
        >>> rvs.get_der()
        [1, 2]
        >>> rvs.get_der([x])
        [1]
        """
        # get jacobian/partial derivatives for all functions from self.f

        if nodes_lst is None:
            #nodes_lst = self.var_lst
            return self.get_jacobian()
        else:
            selected_result = []
            for i in nodes_lst:
                selected_result.append(self.grads[:, self.var_lst.index(i)])
            return selected_result

     
    def get_jacobian(self):
        """ Returns the Jacobian matrix of f list
        
        Parameters
        ----------
        self: Reverse object

        Returns
        ------- 
        calculate the jacobian matrix of f list on all vars through reverse mode on all variables
        
        Examples
        -------- 
        >>> x = Node(1, "x")
        >>> y = Node(2, "y")
        >>> f1 = x + 2*y
        >>> f2 = 2*x + y
        >>> rvs = Reverse([f1, f2], [x, y])
        >>> rvs.get_jacobian()
        [[1, 2], [2, 1]]
        """
        return self.grads
