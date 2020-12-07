from autodiff.symbolic import *
from autodiff.elementary import *


def test_symbolic():
    def test_get_value():
        x1, y1, z1 = symbols('x1 y1 z1')
        f1 = 3 * x1 + 4 * y1 * 2 - z1

        x2, y2, z2 = symbols('x2 y2 z2')
        f2 = 3 * sin(x2) + 8 * y2 ** 3 + z2 ** 2

        assert math.isclose(f1.evaluate({x1: 1, y1: 2, z1: 5}), 14)
        assert math.isclose(f2.evaluate({x2: math.pi, y2: 2, z2: 5}), 89)

    def test_get_der():
        x1, y1, z1 = symbols('x1 y1 z1')
        f1 = 3 * sin(x1) + 4 * cos(y1) + exp(z1)
        values = {x1: math.pi, y1: math.pi / 2, z1: 0}
        assert math.isclose(diff(f1, x1).evaluate(values), -3)
        assert math.isclose(diff(f1, y1).evaluate(values), -4)
        assert math.isclose(diff(f1, z1).evaluate(values), 1)

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
            diff(f1, x)) == "((2)^(cos(x)))*(((((1)*(sin(x)))*(-1))*(ln(2)))+(((cos(x))*(0))/(2)))"

    def test_call():
        x = symbols('x')
        f = x ** 2
        assert f({x: 3}) == 9

    def test_str():
        x = symbols('x')
        f = (log(x) + logb(x, x) - x) / x * x ** 2 + sin(cos(tan(x))) + sinh(cosh(tanh(x))) + arcsin(arccos(arctan(x)))
        expected = "(((((((ln(x))+((ln(x))/(ln(x))))+((-1)*(x)))/(x))*((x)^(2)))+(sin(cos(tan(x)))))+(sinh(cosh(tanh(x)))))+(arcsin(arccos(arctan(x))))"
        assert str(f) == expected
        diff_expected = "(((((((((((1)/(x))+(((((1)/(x))*(ln(x)))+((-1)*((ln(x))*((1)/(x)))))/((ln(x))*(ln(x)))))+(((0)*(x))+((-1)*(1))))*(x))+((-1)*((((ln(x))+((ln(x))/(ln(x))))+((-1)*(x)))*(1))))/((x)*(x)))*((x)^(2)))+(((((ln(x))+((ln(x))/(ln(x))))+((-1)*(x)))/(x))*(((x)^(2))*(((0)*(ln(x)))+(((2)*(1))/(x))))))+(((((1)/((cos(x))*(cos(x))))*(sin(tan(x))))*(-1))*(cos(cos(tan(x))))))+((((1)/((cosh(x))*(cosh(x))))*(sinh(tanh(x))))*(cosh(cosh(tanh(x))))))+(((((1)*((1)/(((x)*(x))+(1))))*((1)/(((1)+((-1)*((arctan(x))*(arctan(x)))))^(0.5))))*(-1))*((1)/(((1)+((-1)*((arccos(arctan(x)))*(arccos(arctan(x))))))^(0.5))))"
        assert str(diff(f, x)) == diff_expected

    def test_div():
        x, y = symbols('x y')
        f = x ** 2 / y ** 2
        values = {x: 3, y: 4}
        assert math.isclose(f.evaluate(values), 9 / 16)
        assert math.isclose(diff(f, x).evaluate(values), 3 / 8)
        assert math.isclose(diff(f, y).evaluate(values), -18 / 64)

    def test_mul():
        x, y = symbols('x y')
        f = x ** 2 * y ** 2
        values = {x: 3, y: 4}
        assert math.isclose(f.evaluate(values), 9 * 16)
        assert math.isclose(diff(f, x).evaluate(values), 6 * 16)
        assert math.isclose(diff(f, y).evaluate(values), 8 * 9)

    def test_log():
        x, y = symbols('x y')
        f = log(x ** 2)
        f2 = logb(y ** 2, x ** 2)
        values = {x: 3, y: 4}
        assert math.isclose(make_ln_expression(4), math.log(4))
        assert math.isclose(f.evaluate(values), log(9))
        assert math.isclose(f2.evaluate(values), logb(16, 9))
        assert math.isclose(diff(f, x).evaluate(values), 2 / 3)
        assert math.isclose(diff(f2, x).evaluate(values), -2 * math.log(16) / (3 * (math.log(9)) ** 2))
        assert math.isclose(diff(f2, y).evaluate(values), 2 / (4 * math.log(9)))

    def test_pow():
        x = symbols('x')
        f1 = x ** (x ** 2)  # x^(x^2)
        f2 = 3 ** x
        f3 = (x ** 2) ** (x ** 2)
        values = {x: 3}
        assert math.isclose(f1.evaluate(values), 3 ** 9)
        assert math.isclose(diff(f1, x).evaluate(values), 3 ** 9 * (3 + 2 * 3 * log(3)))
        assert math.isclose(diff(f2, x).evaluate(values), log(3) * 3 ** 3)
        assert math.isclose(diff(f3, x).evaluate(values), 2 * 3 * (9 ** 9) * (log(9) + 1))

    def test_sin():
        x = symbols('x')
        f = sin(x ** 2)
        values = {x: 3}
        assert math.isclose(f.evaluate(values), sin(9))
        assert math.isclose(diff(f, x).evaluate(values), cos(9) * 6)

    def test_cos():
        x = symbols('x')
        f = cos(x ** 2)
        values = {x: 3}
        assert math.isclose(f.evaluate(values), cos(9))
        assert math.isclose(diff(f, x).evaluate(values), -sin(9) * 6)

    def test_tan():
        x = symbols('x')
        f = tan(x ** 2)
        values = {x: 3}
        assert math.isclose(f.evaluate(values), tan(9))
        assert math.isclose(diff(f, x).evaluate(values), 1 / (cos(9) * cos(9)) * 6)

    def test_arcsin():
        x = symbols('x')
        f = arcsin(x ** 2)
        values = {x: 0.1}
        assert math.isclose(f.evaluate(values), arcsin(0.01))
        assert math.isclose(diff(f, x).evaluate(values), 2 * 0.1 / math.sqrt(1 - 0.1 ** 4))

    def test_arccos():
        x = symbols('x')
        f = arccos(x ** 2)
        values = {x: 0.1}
        assert math.isclose(f.evaluate(values), arccos(0.01))
        assert math.isclose(diff(f, x).evaluate(values), -2 * 0.1 / math.sqrt(1 - 0.1 ** 4))

    def test_arctan():
        x = symbols('x')
        f = arctan(x ** 2)
        values = {x: 3}
        assert math.isclose(f.evaluate(values), arctan(9))
        assert math.isclose(diff(f, x).evaluate(values), 6 / (81 + 1))

    def test_sinh():
        x = symbols('x')
        f = sinh(x ** 2)
        values = {x: 3}
        assert math.isclose(f.evaluate(values), sinh(9))
        assert math.isclose(diff(f, x).evaluate(values), 6 * cosh(9))

    def test_cosh():
        x = symbols('x')
        f = cosh(x ** 2)
        values = {x: 3}
        assert math.isclose(f.evaluate(values), cosh(9))
        assert math.isclose(diff(f, x).evaluate(values), 6 * sinh(9))

    def test_tanh():
        x = symbols('x')
        f = tanh(x ** 2)
        values = {x: 3}
        assert math.isclose(f.evaluate(values), tanh(9))
        assert math.isclose(diff(f, x).evaluate(values), 6 / (cosh(9) * cosh(9)))

    test_get_value()
    test_get_der()
    test_jacobian()
    test_get_expression()
    test_call()
    test_str()
    test_div()
    test_mul()
    test_log()
    test_pow()
    test_sin()
    test_cos()
    test_tan()
    test_arcsin()
    test_arccos()
    test_arctan()
    test_sinh()
    test_cosh()
    test_tanh()
    print("Pass symbolic diff!")


test_symbolic()
