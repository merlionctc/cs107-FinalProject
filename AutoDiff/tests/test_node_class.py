#import sys
#sys.path.append('AutoDiff/src/autodiff')

import pytest

from autodiff.node import *
import numpy as np
import math

def test_node_class():
    """
    Test suite for node class methods
    including 
    __pos__, __neg__
    __add__, __radd__, __sub__, __rsub__,__mul__, __rmul__,__truediv__,__rtruediv__
    __pow__, __rpow__
    __eq__,__ne__,__lt__,__le__,__gt__,__ge__

    """
    
    def test_pos():
        x = Node(1)
        y = +x
        assert y.val == 1
        assert y.name == 'new' 
        assert y.inputs == [x]
        assert y.gradients == [1]
       
    def test_neg():
        x = Node(1)
        y = -x
        assert y.val == -1
        assert y.name == 'new'
        assert y.inputs == [x]
        assert y.gradients == [-1]

    def test_add():
        #test case of f = x+y, when y is node or not
        x = Node(1)
        y = Node(2)
        f = x+y
        f_real = x+2

        assert f.val == 3
        assert f.name == 'new'
        assert f.inputs == [x,y]
        assert f.gradients == [1,1]

        assert f_real.val == 3
        assert f_real.name == 'new'
        assert f_real.inputs == [x]
        assert f_real.gradients == [1]

    def test_radd():
        # test case of f = y + x, when y is not node
        x = Node(1)
        f_real = 2+x
        assert f_real.val == 3
        assert f_real.name == 'new'
        assert f_real.inputs == [x]
        assert f_real.gradients == [1]
    
    def test_sub():
        #test case of f = x-y, when y is node or not
        x = Node(1)
        y = Node(2)
        f = x-y
        f_real = x-2

        assert f.val == -1
        assert f.name == 'new'
        assert f.inputs == [x,y]
        assert f.gradients == [1,-1]

        assert f_real.val == -1
        assert f_real.name == 'new'
        assert f_real.inputs == [x]
        assert f_real.gradients == [1]

    def test_rsub():
        # test case of f = y + x, when y is not node
        x = Node(1)
        f_real = 2-x
        assert f_real.val == 1
        assert f_real.name == 'new'
        assert f_real.inputs == [x]
        assert f_real.gradients == [-1]       

    def test_mul():
        #test case of f = x*y, when y is node or not
        x = Node(1)
        y = Node(2)
        f = x*y
        f_real = x*2
        
        assert f.val == 2
        assert f.name == 'new'
        assert f.inputs == [x,y]
        assert f.gradients == [2,1]

        assert f_real.val == 2
        assert f_real.name == 'new'
        assert f_real.inputs == [x]
        assert f_real.gradients == [2]
        
    def test_rmul():
        #test case of f = y*x, when y is not node
        x = Node(3)
        f_real = 2*x
        assert f_real.val == 6
        assert f_real.name == 'new'
        assert f_real.inputs == [x]
        assert f_real.gradients == [2]

    def test_div():
        #test case of f = x/y, when y is node or not 
        x = Node(1)
        y = Node(2)
        f = x/y
        f_real = x/2

        assert f.val == 0.5
        assert f.name == 'new'
        assert f.inputs == [x,y]
        assert f.gradients == [0.5,-0.25]

        assert f_real.val == 0.5
        assert f_real.name == 'new'
        assert f_real.inputs == [x]
        assert f_real.gradients == [0.5]


    def test_rdiv():
        #test case of f = y/x, when y is not node
        x = Node(2)
        f_real = 1/x
        assert f_real.val == 0.5
        assert f_real.name == 'new'
        assert f_real.inputs == [x]
        assert f_real.gradients == [-0.25]
                        
    
    def test_pow():
        x = Node(2)
        y = Node(3)
        f = x**y
        f_real = x**2

        assert f.val == 2**3
        assert f.name == 'new'
        assert f.inputs == [x, y]
        assert f.gradients == [12, np.log(2)*2**3]

        assert f_real.val == 2**2
        assert f_real.name == 'new'
        assert f_real.inputs == [x]
        assert f_real.gradients == [4]
        
    def test_rpow():
        x = Node(3)
        f = 2**x
        assert f.val == 2**3
        assert f.name == 'new'
        assert f.inputs == [x]
        assert f.gradients == [np.log(2) * 2 ** 3]
        

    ## comparison test
    def test_eq():
        x = Node(2)
        y = Node(2)
        z = Node(1)
        assert True == (x == y)
        assert False == (x == z)
        assert True == (x == 2)
        assert True == (2 == x)
        assert False == (x == 3)
        assert False == (3 == x)

    def test_ne():
        x = Node(2)
        y = Node(2)
        z = Node(1)
        assert False == (x != y)
        assert True == (x != z)
        assert True == (x != 4)
        assert True == (4 != x)
        assert False == (x != 2)
        assert False == (2 != x)
    
    def test_lt():
        x = Node(2)
        y = Node(2)
        z = Node(1)
        assert False == (x < y)
        assert False == (x < z)
        assert True  == (z < x)
        assert True  == (x < 3)
        assert False == (3 < x)

    def test_le():
        x = Node(2)
        y = Node(2)
        z = Node(1)
        assert True  == (x <= y)
        assert False == (x <= z)
        assert True  == (z <= x)
        assert True  == (x <= 3)
        assert False == (3 <= x)


    def test_gt():
        x = Node(2)
        y = Node(2)
        z = Node(1)
        assert False == (x > y)
        assert True  == (x > z)
        assert False == (z > x)
        assert True  == (x > 1)
        assert False == (1 > x)

    def test_ge():
        x = Node(2)
        y = Node(2)
        z = Node(1)
        assert True  == (x >= y)
        assert True  == (x >= z)
        assert False == (z >= x)
        assert True  == (x >= 1)
        assert False == (1 >= x)

    test_pos()
    test_neg()
    test_add()
    test_radd()
    test_sub()
    test_rsub()
    test_mul()
    test_rmul()
    test_div()
    test_rdiv()
    test_pow()
    test_rpow()

    #comparison 

    test_eq()
    test_ne()
    test_lt()
    test_le()
    test_gt()
    test_ge()
    print("Pass node class!")


test_node_class()