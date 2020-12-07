from autodiff.symbolic import *
from autodiff.elementary import *


def test_symbolic():
    def test_get_value():
        x1, y1, z1 = symbols('x1 y1 z1')
        f1 = 3 * x1 + 4 * y1 * 2 - z1

        x2, y2, z2 = symbols('x2 y2 z2')
        f2 = 3 * sin(x2) + 8 * y2 ** 3 + z2 ** 2

        assert f1.evaluate({x1: 1, y1: 2, z1: 5}) == 14
        assert f2.evaluate({x2: math.pi, y2: 2, z2: 5}) == 89

    def test_get_der():
        x1, y1, z1 = symbols('x1 y1 z1')
        f1 = 3 * sin(x1) + 4 * cos(y1) + exp(z1)
        values = {x1: math.pi, y1: math.pi / 2, z1: 0}
        assert diff(f1, x1).evaluate(values) == -3
        assert diff(f1, y1).evaluate(values) == -4
        assert diff(f1, z1).evaluate(values) == 1

    def test_jacobian():
        x, y, z = symbols('x y z')
        f1 = 3 * x + 4 * y * 2 - z
        f2 = 3 * sin(x) + 8 * y ** 3 + z ** 2
        values = {x: math.pi, y: 2, z: 5}
        assert get_jacobian_value([f1, f2], [x, y, z], values) == [[3., 8., -1.], [-3., 96., 10.]]

    def test_get_expression():
        x, y, z = symbols('x y z')
        f1 = 2 ** cos(x)
        assert str(f1) == "(2)^(cos(x))"
        assert str(
            diff(f1, x)) == "((2)^(cos(x)))*(((((1)*(sin(x)))*(-1))*(log[2.718281828459045](2)))+(((cos(x))*(0))/(2)))"

    test_get_value()
    test_get_der()
    test_jacobian()
    test_get_expression()
    print("Pass symbolic diff!")


test_symbolic()
