#import sys
#sys.path.append('AutoDiff/src/autodiff')

import pytest

from autodiff.dual import *
from autodiff.model import *
from autodiff.elementary import *
from autodiff.node import *
import numpy as np
import math


def test_forward_autodiff():
    def test_get_value():
        x1 = Dual(val = 1, der=1, loc = 0, length = 3)
        y1 = Dual(val = 2, der=1, loc = 1, length = 3)
        z1 = Dual(val = 5, der=1, loc = 2, length = 3)
        f1 = 3 * x1 + 4 * y1 * 2 - z1
        fwd_test_1 = Forward(f1)

        x2 = Dual(val = np.pi, der=1, loc = 0, length = 3)
        y2 = Dual(val = 2, der=1, loc = 1, length = 3)
        z2 = Dual(val = 5, der=1, loc = 2, length = 3)

        f2 = 3 * sin(x2) + 8 * y2 ** 3 + z2**2
        fwd_test_2 = Forward(f2)

        assert fwd_test_1.get_value() == [14]
        assert fwd_test_2.get_value() == [89]

    def test_get_der():
        # test single func
        x1 = Dual(val = np.pi, der=1, loc = 0, length = 3)
        y1 = Dual(val = np.pi/2, der=1, loc = 1, length = 3)
        z1 = Dual(val = 0, der=1, loc = 2, length = 3)
        f1 = 3 * sin(x1) + 4 * cos(y1) + exp(z1)
        fwd_test_1 = Forward(f1)
        assert (fwd_test_1.get_der() == np.array([-3, -4,  1.])).all()
        assert (fwd_test_1.get_der(z1) == np.array([1])).all()
        assert (fwd_test_1.get_der(x1,y1) == np.array([-3, -4])).all()

        # test mulitple func
        f2 = 6 * sin(x1) + cos(y1) ** 3 + z1**2
        fwd_test_2 = Forward([f1,f2])
        assert np.allclose(fwd_test_2.get_der(), np.array([[-3.00000000e+00,-4.00000000e+00,1.00000000e+00],[-6.00000000e+00,-1.12481984e-32,0.00000000e+00]]))
        assert np.allclose(fwd_test_2.get_der(z1), np.array([[1.],[0.]]))
        assert np.allclose(fwd_test_2.get_der(y1, x1), np.array([[-4.00000000e+00,-3.00000000e+00],[-1.12481984e-32,-6.00000000e+00]]))

    
    def test_jacobian():
        x = Dual(val = np.pi, der=1, loc = 0, length = 3)
        y = Dual(val = 2, der=1, loc = 1, length = 3)
        z = Dual(val = 5, der=1, loc = 2, length = 3)
        f1 = 3 * x + 4 * y * 2 - z
        f2 = 3 * sin(x) + 8 * y ** 3 + z**2
        fwd = Forward([f1, f2])
        assert np.allclose(fwd.get_jacobian(), np.array([[ 3., 8.,-1.],[-3.,96.,10.]]))
        
    #def test_get_expression():
        #try:
           # x = Dual(val = np.pi, der=1, loc = 0, length = 3)
           # y = Dual(val = np.pi/2, der=1, loc = 1, length = 3)
           # z = Dual(val = 0, der=1, loc = 2, length = 3)
           # f3 = 3 * sin(x) + 4 * cos(y) + exp(z)
           # fwd_f3 = Forward(f3)
           # fwd_f3.get_expression()
        #except:
            #pass
        
    test_get_value()
    test_get_der()
    test_jacobian()
    #test_get_expression()
    print("Pass forward auto diff!")

def test_reverse_autodiff():
    def test_get_value():
        x1 = Node(val = 1)
        y1 = Node(val = 2)
        z1 = Node(val = 5)
        f1 = 3 * x1 + 4 * y1 * 2 - z1

        #rvs_test_1 = Reverse([f1], [x1, y1, z1])

        x2 = Node(val = np.pi)
        y2 = Node(val = 2)
        z2 = Node(val = 5)

        f2 = 3 * sin(x2) + 8 * y2 ** 3 + z2**2
        
        rvs_test = Reverse([f1, f2])


        #assert fwd_test_1.get_value() == [14]
        assert rvs_test.get_value() == [14, 89]

    def test_get_der():
        x = Node(val = np.pi)
        y = Node(val = np.pi/2)
        z = Node(val = 0)
        
        f3 = 3 * sin(x) + 4 * cos(y) + exp(z)
        rvs_f3 = Reverse([f3], [x, y, z])
        assert (rvs_f3.get_der() == np.array([-3, -4,  1.])).all()
        assert (rvs_f3.get_der([z]) == np.array([1])).all()
        assert (rvs_f3.get_der([x,y]) == np.array([-3, -4])).all()
    
    def test_jacobian():
        '''
        x = Node(val = np.pi)
        y = Node(val = 2)
        z = Node(val = 5)
        f1 = 3 * x + 4 * y * 2 - z
        f2 = 3 * sin(x) + 8 * y ** 3 + z ** 2
        rvs = Reverse([f1, f2], [x, y, z])
        print(rvs.get_jacobian())
        assert np.allclose(rvs.get_jacobian(), np.array([[ 3., 8.,-1.],[-3.,96.,10.]]))
        '''
        x = Node(val = np.pi)
        y = Node(val = 2)
        z = Node(val = 5)
        f1 = 3 * x + 4 * y * 2 - z
        f2 = 3 * sin(x) + 8 * y ** 3 + z**2
        rvs1 = Reverse([f1, f2], [x, y, z])
        assert np.allclose(rvs1.get_jacobian(), np.array([[ 3., 8.,-1.],[-3.,96.,10.]]))
        f3 = 3 * sin(x) + 8 * y ** 3 + x**2
        rvs2 = Reverse([f3], [x, y])
        print("f3 is ", f3)
        assert np.allclose(rvs2.get_jacobian(), np.array([2*np.pi-3, 96.]))


    test_get_value()
    test_get_der()
    test_jacobian()
    print("Pass reverse auto diff!")



def test_simplify():
    def test_simplify_num():
        x = Dual(2, 1)
        z1 = x*x
        z2 = x**x
        z3 = x/x
        assert z1.val == 4
        assert z1.der == 4
        assert z2.val == 4
        assert z2.der == (np.log(2) + 1)*2**2
        assert z3.val == 1
        assert z3.der == 0
        
    
    def test_simplify_vector():
        x = Dual(2, 1, loc = 0, length = 2)  
        y = Dual(1, 1, loc = 1, length = 2)
        z1 = (x+y)*x
        z2 = x/x + 3*y
        z3 = x**(2*x + 3*y) 
        assert z1.val == 6
        assert (z1.der == np.array([5,2])).all()
        assert z2.val == 4
        assert (z2.der == np.array([0,3])).all()
        assert z3.val == 2**7
        assert (z3.der == np.array([2**7*(2*np.log(2)+ 7/2), 2**7*3*np.log(2)])).all()
        print("z3 is ", z3)
               
    test_simplify_num()
    test_simplify_vector()
    print("Pass simplify tests!")

test_forward_autodiff()
test_simplify()
test_reverse_autodiff()