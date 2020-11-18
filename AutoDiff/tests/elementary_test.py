import sys
sys.path.append('AutoDiff/src/autodiff')

import pytest

from dual import *
from elementary import *
import numpy as np
import math


def test_elementary():
    def test_exp():
        x = Dual(2, 1)
        z = exp(x)
        x_real = exp(5)
        assert z.val == np.e**2
        assert z.der == np.e**2
        assert x_real == np.e**5

    def test_sin():
        x = Dual(np.pi, 1)
        z = sin(x)
        x_real = sin(np.pi)
        assert z.val == np.sin(np.pi)
        assert z.der == np.cos(np.pi)
        assert x_real == np.sin(np.pi)
             
    def test_cos():
        x = Dual(np.pi, 1)
        z = cos(x)
        x_real = cos(4)
        assert z.val == np.cos(np.pi)
        assert z.der == -np.sin(np.pi)
        assert x_real == np.cos(4)
        
    def test_tan():
        x = Dual(np.pi/4, 1)
        z = tan(x)
        x_real = tan(5)
        assert z.val == np.tan(np.pi/4)
        assert z.der == 1/(np.cos(np.pi/4))**2
        assert x_real == np.tan(5)

    def test_log():
        x = Dual(1,1)
        z = log(x)
        x_real = log(5)
        assert z.val == 0
        assert z.der == 1
        assert x_real == np.log(5)

    def test_sqrt():
        x = Dual(4,1)
        z = sqrt(x)
        x_real = sqrt(4)
        assert z.val == 2
        assert z.der == 0.5*4**(-0.5)
        assert x_real == np.sqrt(4)

    def test_arcsin():
        x = Dual(0.5,1)
        z = arcsin(x)
        x_real = arcsin(0.5)
        assert z.val == np.arcsin(0.5)
        assert z.der == 1/np.sqrt(1-0.5**2)
        assert x_real == np.arcsin(0.5)

    def test_arccos():
        x = Dual(0.5,1)
        z = arccos(x)
        x_real = arccos(0.5)
        assert z.val == np.arccos(0.5)
        assert z.der == -1/np.sqrt(1-0.5**2)
        assert x_real == np.arccos(0.5)

    def test_arctan():
        x = Dual(1,1)
        z = arctan(x)
        x_real = arctan(1)
        assert z.val == np.arctan(1)
        assert z.der == 0.5
        assert x_real == arctan(1)



    test_exp()
    test_sin()
    test_cos()
    test_tan()
    test_log()
    test_sqrt()
    test_arcsin()
    test_arccos()
    test_arctan()
    print("Pass elementary!")

test_elementary()