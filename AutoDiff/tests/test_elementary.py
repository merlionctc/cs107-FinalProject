#import sys
#sys.path.append('AutoDiff/src/autodiff')

import pytest

from autodiff.dual import *
from autodiff.elementary import *
from autodiff.node import *
import numpy as np
import math


def test_elementary_dual():
    """
    Test suite for functions in elementary module, for Dual class
    including 
    exp, sqrt, log, log_b, 
    logistics
    sin, cos, tan
    arcsin, arccos, arctan
    sinh, cosh, tanh
    """
    def test_exp_dual():
        x = Dual(2, 1)
        z = exp(x)
        x_real = exp(5)
        assert z.val == np.e**2
        assert z.der == np.e**2
        assert x_real == np.e**5

    def test_sin_dual():
        x = Dual(np.pi, 1)
        z = sin(x)
        x_real = sin(np.pi)
        assert z.val == np.sin(np.pi)
        assert z.der == np.cos(np.pi)
        assert x_real == np.sin(np.pi)
             
    def test_cos_dual():
        x = Dual(np.pi, 1)
        z = cos(x)
        x_real = cos(4)
        assert z.val == np.cos(np.pi)
        assert z.der == -np.sin(np.pi)
        assert x_real == np.cos(4)
        
    def test_tan_dual():
        x = Dual(np.pi/4, 1)
        z = tan(x)
        x_real = tan(5)
        assert z.val == np.tan(np.pi/4)
        assert z.der == 1/(np.cos(np.pi/4))**2
        assert x_real == np.tan(5)

    def test_log_dual():
        x = Dual(1,1)
        z = log(x)
        x_real = log(5)
        assert z.val == 0
        assert z.der == 1
        assert x_real == np.log(5)

    def test_sqrt_dual():
        x = Dual(4,1)
        z = sqrt(x)
        x_real = sqrt(4)
        assert z.val == 2
        assert z.der == 0.5*4**(-0.5)
        assert x_real == np.sqrt(4)

    def test_arcsin_dual():
        x = Dual(0.5,1)
        z = arcsin(x)
        x_real = arcsin(0.5)
        assert z.val == np.arcsin(0.5)
        assert z.der == 1/np.sqrt(1-0.5**2)
        assert x_real == np.arcsin(0.5)

    def test_arccos_dual():
        x = Dual(0.5,1)
        z = arccos(x)
        x_real = arccos(0.5)
        assert z.val == np.arccos(0.5)
        assert z.der == -1/np.sqrt(1-0.5**2)
        assert x_real == np.arccos(0.5)

    def test_arctan_dual():
        x = Dual(1,1)
        z = arctan(x)
        x_real = arctan(1)
        assert z.val == np.arctan(1)
        assert z.der == 0.5
        assert x_real == np.arctan(1)

    def test_sinh_dual():
        x = Dual(1,1)
        z = sinh(x)
        x_real = sinh(1)
        assert z.val == np.sinh(1)
        assert z.der == np.cosh(1)
        assert x_real == np.sinh(1)

    def test_cosh_dual():
        x = Dual(1,1)
        z = cosh(x)
        x_real = cosh(1)
        assert z.val == np.cosh(1)
        assert z.der == np.sinh(1)
        assert x_real == np.cosh(1)

    def test_tanh_dual():
        x = Dual(1,1)
        z = tanh(x)
        x_real = tanh(1)
        assert z.val == np.tanh(1)
        assert z.der == (np.cosh(1)**2 - np.sinh(1)**2)/np.cosh(1)**2
        assert x_real == np.tanh(1)

    def test_logb_dual():
        x = Dual(2,1)
        z = logb(x,2) # chossing 2 as base
        x_real = logb(2,2)
        assert z.val == 1
        assert z.der == 1/np.log(4)
        assert x_real == 1

    def test_logistic_dual():
        #L/(1 + np.exp(-k*(x-x0)))
        x = Dual(1,1)
        z = logistic(x)
        x_real = logistic(1)
 
        assert z.val == 1/(1+np.e**(-1))
        assert z.der == z.val*(1- z.val)
        assert x_real == 1/(1+np.e**(-1))

    test_exp_dual()
    test_sin_dual()
    test_cos_dual()
    test_tan_dual()
    test_log_dual()
    test_sqrt_dual()
    test_arcsin_dual()
    test_arccos_dual()
    test_arctan_dual()
    test_sinh_dual()
    test_cosh_dual()
    test_tanh_dual()
    test_logb_dual()
    test_logistic_dual()
    print("Pass dual elementary!")

def test_elementary_node():
    """
    Test suite for functions in elementary module, for Node class
    including 
    exp, sqrt, log, log_b, 
    logistics
    sin, cos, tan
    arcsin, arccos, arctan
    sinh, cosh, tanh
    """
    def test_exp_node():
        x = Node(2)
        z = exp(x)
        assert z.val == np.e**2
        assert z.inputs == [x]
        assert z.gradients == [np.e**2]
        #assert x_real == np.e**5

    def test_sin_node():
        x = Node(np.pi)
        z = sin(x)
        assert z.val == np.sin(np.pi)
        assert z.gradients == [np.cos(np.pi)]
        assert z.inputs == [x]
        #assert x_real == np.sin(np.pi)
             
    def test_cos_node():
        x = Node(np.pi)
        z = cos(x)
        #x_real = cos(4)
        assert z.val == np.cos(np.pi)
        assert z.gradients == [-np.sin(np.pi)]
        assert z.inputs == [x]
        #assert x_real == np.cos(4)
        
    def test_tan_node():
        x = Node(np.pi/4)
        z = tan(x)
        assert z.val == np.tan(np.pi/4)
        assert z.gradients == [1/(np.cos(np.pi/4))**2]
        assert z.inputs == [x]
        #assert x_real == np.tan(5)

    def test_log_node():
        x = Node(1)
        z = log(x)
        #x_real = log(5)
        assert z.val == 0
        assert z.gradients == [1]
        assert z.inputs == [x]
        #assert x_real == np.log(5)

    def test_sqrt_node():
        x = Node(4)
        z = sqrt(x)
        #x_real = sqrt(4)
        assert z.val == 2
        assert z.gradients == [0.5*4**(-0.5)]
        assert z.inputs == [x]
        #assert x_real == np.sqrt(4)

    def test_arcsin_node():
        x = Node(0.5)
        z = arcsin(x)
        #x_real = arcsin(0.5)
        assert z.val == np.arcsin(0.5)
        assert z.gradients == [1/np.sqrt(1-0.5**2)]
        assert z.inputs == [x]
        #assert x_real == np.arcsin(0.5)


    def test_arccos_node():
        x = Node(0.5)
        z = arccos(x)
        #x_real = arccos(0.5)
        assert z.val == np.arccos(0.5)
        assert z.gradients == [-1/np.sqrt(1-0.5**2)]
        assert z.inputs == [x]
        #assert x_real == np.arccos(0.5)

    def test_arctan_node():
        x = Node(1)
        z = arctan(x)
        #x_real = arctan(1)
        assert z.val == np.arctan(1)
        assert z.inputs == [x]
        assert z.gradients == [0.5]
        #assert x_real == np.arctan(1)

    def test_sinh_node():
        x = Node(1,1)
        z = sinh(x)
        #x_real = sinh(1)
        assert z.val == np.sinh(1)
        assert z.gradients == [np.cosh(1)]
        assert z.inputs == [x]
        #assert x_real == np.sinh(1)

    def test_cosh_node():
        x = Node(1,1)
        z = cosh(x)
        #x_real = cosh(1)
        assert z.val == np.cosh(1)
        assert z.gradients == [np.sinh(1)]
        assert z.inputs == [x]
        #assert x_real == np.cosh(1)

    def test_tanh_node():
        x = Node(1)
        z = tanh(x)
        #x_real = tanh(1)
        assert z.val == np.tanh(1)
        assert z.gradients == [(np.cosh(1)**2 - np.sinh(1)**2)/ (np.cosh(1)**2)]
        assert z.inputs == [x]
        #assert x_real == np.tanh(1)

    def test_logb_node():
        x = Node(2,1)
        z = logb(x,2) # chossing 2 as base
        #x_real = logb(2,2)
        assert z.val == 1
        assert z.gradients == [1/np.log(4)]
        assert z.inputs == [x]
        #assert x_real == 1

    def test_logistic_node():
        x = Node(1)
        z = logistic(x)
        #x_real = logistics(1)
        assert z.val == 1/(1+np.e**(-1))
        assert z.gradients == [z.val*(1-z.val)]
        assert z.inputs == [x]
        #assert x_real == 1/(1+np.e)


    test_exp_node()
    test_sin_node()
    test_cos_node()
    test_tan_node()
    test_log_node()
    test_sqrt_node()
    test_arcsin_node()
    test_arccos_node()
    test_arctan_node()
    test_sinh_node()
    test_cosh_node()
    test_tanh_node()
    test_logb_node()
    test_logistic_node()
    print("Pass Node elementary!")

test_elementary_dual()
test_elementary_node()

