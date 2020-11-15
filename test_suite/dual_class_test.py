import sys
sys.path.append('./autodiff')
from dual_class import *
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


test_dual_class()
