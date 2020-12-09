#import sys
#sys.path.append('AutoDiff/src/autodiff')

import numpy as np


from autodiff.dual import *
from autodiff.elementary import *
from autodiff.model import *
from autodiff.node import *
from autodiff.symbolic import *

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

    test_get_value()
    test_get_der()
    test_jacobian()
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
        x = Node(val = np.pi, name = "x")
        y = Node(val = np.pi/2, name = "y")
        z = Node(val = 0, name = "z")

        f3 = 3 * sin(x) + 4 * cos(y) + exp(z)
        rvs_f3 = Reverse([f3], [x, y, z])
        assert (rvs_f3.get_der() == np.array([-3, -4,  1.])).all()
        assert (rvs_f3.get_der([z]) == np.array([1])).all()
        assert (rvs_f3.get_der([x,y]) == np.array([-3, -4])).all()

        # test mulitple func
        f4 = 6 * sin(x) + cos(y) ** 3 + z**2
        rvs = Reverse([f3,f4], [x, y, z])
        rvs_f4 = Reverse([f4], [x, y, z])
        print(rvs_f4.get_der())
        print(rvs.get_der())
        print(rvs.get_der([z]))
        print(rvs.get_der([y, x]))
        assert np.allclose(rvs.get_der(), np.array([[-3.00000000e+00,-4.00000000e+00,1.00000000e+00],[-6.00000000e+00,-1.12481984e-32,0.00000000e+00]]))
        assert np.allclose(rvs.get_der([z]), np.array([[1.],[0.]]))
        assert np.allclose(rvs.get_der([y, x]), np.array([[-4.00000000e+00,-3.00000000e+00],[-1.12481984e-32,-6.00000000e+00]]))


    def test_jacobian():
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
    
    def test_simplify_vector_node():
        x = Node(2)
        y = Node(1)
        z1 = (x+y)*x
        z2 = x/x + 3*y
        z3 = x**(2*x + 3*y)
        assert z1.val == 6
        rvs1 = Reverse([z1], [x,y])
        assert (rvs1.get_der() == np.array([5,2])).all()
        assert z2.val == 4
        rvs2 = Reverse([z2], [x,y])
        assert (rvs2.get_der() == np.array([0,3])).all()
        assert z3.val == 2**7
        rvs3 = Reverse([z3], [x,y])
        assert (rvs3.get_der() == np.array([2**7*(2*np.log(2)+ 7/2), 2**7*3*np.log(2)])).all()
        print("z3 is ", z3)

    test_simplify_num()
    test_simplify_vector()
    test_simplify_vector_node()
    print("Pass simplify tests!")

def test_fwd_rvs_syb():
    x1 = Dual(2, 1, loc = 0, length = 3)
    y1 = Dual(np.pi, 1, loc = 1, length = 3)
    z1 = Dual(4, 1, loc = 2, length = 3)
    f_fwd1 = (tanh(cos(sin(y1))**z1) + logistic(z1**z1, 2, 3, 4))**(1/x1)
    f_fwd2 = exp(arccos(tan(sin(y1))) + logb(z1**(1/2), 1/5)*sinh(x1))
    fwd = Forward([f_fwd1, f_fwd2])

    x2 = Node(2)
    y2 = Node(np.pi)
    z2 = Node(4)
    f_rvs1 = (tanh(cos(sin(y2))**z2) + logistic(z2**z2, 2, 3, 4))**(1/x2)
    f_rvs2 = exp(arccos(tan(sin(y2))) + logb(z2**(1/2), 1/5)*sinh(x2))
    rvs = Reverse([f_rvs1, f_rvs2], [x2, y2, z2])

    x, y, z = symbols('x y z')
    f1 = (tanh(cos(sin(y))**z) + logistic(z**z, 2, 3, 4))**(1/x)
    f2 = exp(arccos(tan(sin(y))) + logb(z**(1/2), 1/5)*sinh(x))
    values = {x: 2, y: np.pi, z: 4}

    assert np.allclose(get_jacobian_value([f1, f2], [x, y, z], values), fwd.get_jacobian())
    assert np.allclose(rvs.get_jacobian(), fwd.get_jacobian())
    print("Pass fwd & rvs & symbolic !")

    


    


test_forward_autodiff()
test_simplify()
test_reverse_autodiff()
test_fwd_rvs_syb()