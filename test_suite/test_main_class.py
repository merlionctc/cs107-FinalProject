import sys
sys.path.append('./autodiff')

from dual_class import *
from autodiff import *
from elementary import *
import numpy as np
import math

def test_dual_class():
    def test_pos():
        x = Dual(1, 1)
        y = +x
        assert y.val == 1
        assert y.der == 1
    
    def test_neg():
        x = Dual(1, 1)
        y = -x
        assert y.val == -1
        assert y.der == -1

    def test_add():
        x = Dual(1, 1)
        z = x+2
        assert z.val == 3
        assert z.der == 1   

    def test_add_dual():
        x = Dual(1,1,loc = 0, length = 2)
        y = Dual(2,1,loc = 1, length = 2)
        z = x+y
        assert z.val == 3
        assert (z.der == np.array([1,1])).all()

    def test_radd():
        x = Dual(1, 1)
        z = 2+x
        assert z.val == 3
        assert z.der == 1
    
    def test_radd_dual():
        x = Dual(1,1,loc = 0, length = 2)
        y = Dual(2,1,loc = 1, length = 2)
        z = y+x
        assert z.val == 3
        assert (z.der == np.array([1,1])).all()

    def test_sub():
        x = Dual(1, 1)
        z = x-2
        assert z.val == -1
        assert z.der == 1
    
    def test_sub_dual():
        x = Dual(1,1,loc = 0, length = 2)
        y = Dual(2,1,loc = 1, length = 2)
        z = x-y
        z2 = x - x
        assert z.val == -1
        assert (z.der == np.array([1,-1])).all()
        assert z2.val == 0
        assert (z2.der == np.array([0,0])).all()


    def test_rsub():
        x = Dual(1, 1)
        z = 2-x
        assert z.val == 1
        assert z.der == -1
    
    def test_rsub_dual():
        x = Dual(1,1,loc = 0, length = 2)
        y = Dual(2,1,loc = 1, length = 2)
        z = y-x
        assert z.val == 1
        assert (z.der == np.array([-1,1])).all()


    def test_mul():
        x = Dual(1, 1)
        z = x*2
        assert z.val == 2
        assert z.der == 2
    
    def test_mul_dual():
        x = Dual(1, 1, loc = 0, length = 2)
        y = Dual(2, 1, loc = 1, length = 2)
        z = x*y
        assert z.val == 2
        assert (z.der == np.array([2, 1])).all()

    def test_rmul():
        x = Dual(1, 1)
        z = 2*x
        assert z.val == 2
        assert z.der == 2
    
    def test_rmul_dual():
        x = Dual(1, 1, loc = 0, length = 2)
        y = Dual(2, 1, loc = 1, length = 2)
        z = x*y
        assert z.val == 2
        assert (z.der == np.array([2, 1])).all()

    def test_div():
        x = Dual(2, 1)
        z = x/2
        assert z.val == 1
        assert z.der == 1/2
    
    def test_div_dual():
        x = Dual(1, 1, loc = 0, length = 2)
        y = Dual(2, 1, loc = 1, length = 2)
        z = x/y
        assert z.val == 1/2
        assert (z.der == np.array([1/2, -1/4])).all()

    def test_rdiv():
        x = Dual(2, 1)
        z = 2/x
        assert z.val == 1
        assert z.der == -1/2
    
    def test_rdiv_dual():
        x = Dual(1, 1, loc = 0, length = 2)
        y = Dual(2, 1, loc = 1, length = 2)
        z = x/y
        assert z.val == 1/2
        assert (z.der == np.array([1/2, -1/4])).all()
    
    def test_pow():
        x = Dual(2,1)
        z = x**2
        assert z.val == 4
        assert z.der == 4
    
    def test_rpow():
        x = Dual(2,1)
        z = 2**x
        assert z.val == 4
        assert z.der == np.log(2) * 2 ** 2
    
    def test_pow_dual():
        x = Dual(2, 1, loc = 0, length = 2)
        y = Dual(3, 1, loc = 1, length = 2)
        z = x**y
        assert z.val == 2**3
        assert (z.der == np.array([12, np.log(2)*2**3])).all()

    test_pos()
    test_neg()
    test_add()
    test_add_dual()
    test_radd()
    test_radd_dual()
    test_sub()
    test_sub_dual()
    test_rsub()
    test_rsub_dual()
    test_mul()
    test_mul_dual()
    test_rmul()
    test_rmul_dual()
    test_div()
    test_div_dual()
    test_rdiv()
    test_rdiv_dual()
    test_pow()
    test_pow_dual()
    test_rpow()
    print("Pass dual class!")

def test_elementary():

    def test_exp():
        x = Dual(2, 1)
        z = exp(x)
        assert z.val == np.e**2
        assert z.der == np.e**2

    def test_sin():
        x = Dual(np.pi, 1)
        z = sin(x)
        assert z.val == np.sin(np.pi)
        assert z.der == np.cos(np.pi)
             
    def test_cos():
        x = Dual(np.pi, 1)
        z = cos(x)
        assert z.val == np.cos(np.pi)
        assert z.der == -np.sin(np.pi)
        
    def test_tan():
        x = Dual(np.pi/4, 1)
        z = tan(x)
        assert z.val == np.tan(np.pi/4)
        assert z.der == 1/(np.cos(np.pi/4))**2

    def test_log():
        x = Dual(1,1)
        z = log(x)
        assert z.val == 0
        assert z.der == 1

    test_exp()
    test_sin()
    test_cos()
    test_tan()
    test_log()
    print("Pass elementary!")
    
    

def test_autodiff():
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

        assert fwd_test_1.get_value() == 14
        assert fwd_test_2.get_value() == 89

    def test_get_der():
        x = Dual(val = np.pi, der=1, loc = 0, length = 3)
        y = Dual(val = np.pi/2, der=1, loc = 1, length = 3)
        z = Dual(val = 0, der=1, loc = 2, length = 3)

        f3 = 3 * sin(x) + 4 * cos(y) + exp(z)

        fwd_f3 = Forward(f3)
        assert (fwd_f3.get_der() == np.array([-3, -4,  1.])).all()
        assert (fwd_f3.get_der(z) == np.array([1])).all()
        assert (fwd_f3.get_der(x,y) == np.array([-3, -4])).all()
    
    test_get_value()
    test_get_der()
    print("Pass auto diff!")

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
               
    test_simplify_num()
    test_simplify_vector()
    print("Pass simplify tests!")

        
test_dual_class()
test_elementary()
test_autodiff()
test_simplify()
print("All tests passed!")