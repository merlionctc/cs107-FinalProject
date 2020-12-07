import math


class Expression:
    def evaluate(self, values):
        '''
        Evaluate the value of this Expression with the given values of variables.
        '''
        raise NotImplementedError()

    def _symdiff(self, respect_to):
        raise NotImplementedError()

    def __add__(self, other):
        op = other if isinstance(other, Expression) else Constant(other)
        return SumExpression([self, op])

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        op = other if isinstance(other, Expression) else Constant(other)
        return SumExpression([self, ProductExpression([Constant(-1), op])])

    def __rsub__(self, other):
        op = other if isinstance(other, Expression) else Constant(other)
        return op - self

    def __mul__(self, other):
        op = other if isinstance(other, Expression) else Constant(other)
        return ProductExpression([self, op])

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        op = other if isinstance(other, Expression) else Constant(other)
        return DivisionExpression(self, op)

    def __rtruediv__(self, other):
        op = other if isinstance(other, Expression) else Constant(other)
        return DivisionExpression(op, self)

    def __pow__(self, power, modulo=None):
        op = power if isinstance(power, Expression) else Constant(power)
        return PowerExpression(base=self, exponent=op)

    def __rpow__(self, other):
        op = other if isinstance(other, Expression) else Constant(other)
        return PowerExpression(base=op, exponent=self)

    def __neg__(self):
        return self * -1


class Constant(Expression):
    def __init__(self, value):
        self.value = value

    def evaluate(self, values):
        return self.value

    def _symdiff(self, respect_to):
        return Constant(0)

    def __str__(self):
        return str(self.value)


class Symbol(Expression):
    def __init__(self, name):
        self.name = name

    def evaluate(self, values):
        assert self in values
        return values[self]

    def _symdiff(self, respect_to):
        return Constant(1) if respect_to is self else Constant(0)

    def __str__(self):
        return self.name


class SumExpression(Expression):
    def __init__(self, operands):
        for i in operands:
            assert isinstance(i, Expression)
        self.operands = operands

    def evaluate(self, values):
        return sum([i.evaluate(values) for i in self.operands])

    def _symdiff(self, respect_to):
        return SumExpression([i._symdiff(respect_to) for i in self.operands])

    def __str__(self):
        return '+'.join(['(%s)' % i for i in self.operands])


class ProductExpression(Expression):
    def __init__(self, operands):
        self.operands = operands

    def evaluate(self, values):
        p = 1
        for i in self.operands:
            p *= i.evaluate(values)
        return p

    def _symdiff(self, respect_to):
        diffs = [i._symdiff(respect_to) for i in self.operands]
        expr_operands = []
        for i in range(len(self.operands)):
            this_expr_operands = []
            for j in range(len(self.operands)):
                if i == j:
                    this_expr_operands.append(diffs[j])
                else:
                    this_expr_operands.append(self.operands[j])
            expr_operands.append(ProductExpression(this_expr_operands))
        return SumExpression(expr_operands)

    def __str__(self):
        return '*'.join(['(%s)' % i for i in self.operands])


class DivisionExpression(Expression):
    def __init__(self, num: Expression, denom: Expression):
        self.num = num
        self.denom = denom

    def evaluate(self, values):
        return self.num.evaluate(values) / self.denom.evaluate(values)

    def _symdiff(self, respect_to):
        return (self.num._symdiff(respect_to) * self.denom - self.num * self.denom._symdiff(respect_to)) / (
            self.denom * self.denom)

    def __str__(self):
        return '(%s)/(%s)' % (self.num, self.denom)


class LogExpression(Expression):
    def __init__(self, x: Expression, base: Expression = Constant(math.e)):
        self.base = base
        self.x = x

    def evaluate(self, values):
        return math.log(self.x.evaluate(values), self.base.evaluate(values))

    def _symdiff(self, respect_to):
        # TODO: This does not support the case where base is not Constant.
        return 1 / (LogExpression(self.base) * self.x)

    def __str__(self):
        return 'log[%s](%s)' % (self.base, self.x)


class PowerExpression(Expression):
    def __init__(self, exponent: Expression, base: Expression = Constant(math.e)):
        self.base = base
        self.exponent = exponent

    def evaluate(self, values):
        return self.base.evaluate(values) ** self.exponent.evaluate(values)

    def _symdiff(self, respect_to):
        return self * (
            self.exponent._symdiff(respect_to) * LogExpression(self.base) + self.exponent * self.base._symdiff(
                respect_to) / self.base)

    def __str__(self):
        return '(%s)^(%s)' % (self.base, self.exponent)


class SinExpression(Expression):
    def __init__(self, x: Expression):
        self.x = x

    def evaluate(self, values):
        return math.sin(self.x.evaluate(values))

    def _symdiff(self, respect_to):
        return self.x._symdiff(respect_to) * CosExpression(self.x)

    def __str__(self):
        return 'sin(%s)' % self.x


class CosExpression(Expression):
    def __init__(self, x: Expression):
        self.x = x

    def evaluate(self, values):
        return math.cos(self.x.evaluate(values))

    def _symdiff(self, respect_to):
        return self.x._symdiff(respect_to) * SinExpression(self.x) * -1

    def __str__(self):
        return 'cos(%s)' % self.x


class TanExpression(Expression):
    def __init__(self, x: Expression):
        self.x = x

    def evaluate(self, values):
        return math.tan(self.x.evaluate(values))

    def _symdiff(self, respect_to):
        return self.x._symdiff(respect_to) / (CosExpression(self.x) * CosExpression(self.x))

    def __str__(self):
        return 'tan(%s)' % self.x


class ArcsinExpression(Expression):
    def __init__(self, x: Expression):
        self.x = x

    def evaluate(self, values):
        return math.asin(self.x.evaluate(values))

    def _symdiff(self, respect_to):
        return self.x._symdiff(respect_to) * (1 / (1 - self.x * self.x) ** 0.5)

    def __str__(self):
        return 'arcsin(%s)' % self.x


class ArccosExpression(Expression):
    def __init__(self, x: Expression):
        self.x = x

    def evaluate(self, values):
        return math.acos(self.x.evaluate(values))

    def _symdiff(self, respect_to):
        return self.x._symdiff(respect_to) * (1 / (1 - self.x * self.x) ** 0.5) * -1

    def __str__(self):
        return 'arccos(%s)' % self.x


class ArctanExpression(Expression):
    def __init__(self, x: Expression):
        self.x = x

    def evaluate(self, values):
        return math.atan(self.x.evaluate(values))

    def _symdiff(self, respect_to):
        return self.x._symdiff(respect_to) * (1 / (self.x * self.x + 1))

    def __str__(self):
        return 'arctan(%s)' % self.x


class SinhExpression(Expression):
    def __init__(self, x: Expression):
        self.x = x

    def evaluate(self, values):
        return math.sinh(self.x.evaluate(values))

    def _symdiff(self, respect_to):
        return self.x._symdiff(respect_to) * CoshExpression(self.x)

    def __str__(self):
        return 'sinh(%s)' % self.x


class CoshExpression(Expression):
    def __init__(self, x: Expression):
        self.x = x

    def evaluate(self, values):
        return math.cosh(self.x.evaluate(values))

    def _symdiff(self, respect_to):
        return self.x._symdiff(respect_to) * SinhExpression(self.x)

    def __str__(self):
        return 'cosh(%s)' % self.x


class TanhExpression(Expression):
    def __init__(self, x: Expression):
        self.x = x

    def evaluate(self, values):
        return math.tanh(self.x.evaluate(values))

    def _symdiff(self, respect_to):
        return self.x._symdiff(respect_to) / (TanhExpression(self.x) * TanhExpression(self.x))

    def __str__(self):
        return 'tanh(%s)' % self.x
