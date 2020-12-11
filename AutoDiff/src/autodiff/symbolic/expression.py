from __future__ import annotations

import math

# Symbolic reverse differentiation, an illustration of reverse mode differentiation

class Expression:
    """

    - A parent class of Expression prints out the human-readable expression of function
    - the child classes are mathematical operands

    To sum up, it 
	1. Creates an Expression class supporting custom operations for symbolic differentiation 
    (first, second, and higher order supported)

    2. Contains methods of standard operations: addition, substraction, multiplication, division, exponentiation, and negation

	"""
    def evaluate(self, values: dict[Symbol, float]) -> float:
        '''Evaluate the value of this Expression with the given values of variables.

        will be implemented in different child classes of Expression.

        Parameters
        ----------
        self: Expression
        values: dict: key -> variable symbol (x, y, z, etc); value -> float
        
        Returns
        ------- 
        the numerical value (float) of the expression

        Examples
        -------
        >>> evaluate({x:1})
        NotImplementedError
        '''
        raise NotImplementedError()

    def _symdiff(self, respect_to: Symbol) -> Expression:
        '''Display the symbolic representation of the derivative of this Expression.

        a "private method" that will be implemented in different child classes of Expression.

        Parameters
        ----------
        self: Expression
        respect_to: variable symbol (x, y, z, etc), the partial derivative direction
        
        Returns
        ------- 
        the symbolic representation of the derivative, an expression

        '''        
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        '''Special method enabling Expression instance to use evalute method and returns the derivative value of the instance
        
        '''       
         
        return self.evaluate(args[0])

    def __add__(self, other):
        '''Addition on Expression.

        Parameters
        ----------
        self: Expression
        other: Expression or Constant
        
        Returns
        ------- 
        an sum expression object, the symbolic representation of the of addition(+) for two objects

        '''        
        op = other if isinstance(other, Expression) else Constant(other)
        return SumExpression([self, op])

    def __radd__(self, other):
        '''Addition (commutative) on Expression.

        Parameters
        ----------
        self: Expression
        other: Expression or Constant
        
        Returns
        ------- 
        an sum expression object, the symbolic representation of the of addition(+) for two objects

        '''        
        return self + other

    def __sub__(self, other):
        '''Substraction on Expression.

        Parameters
        ----------
        self: Expression
        other: Expression or Constant
        
        Returns
        ------- 
        an sum expression object, the symbolic representation of the of substraction(-) for two objects

        '''        
        op = other if isinstance(other, Expression) else Constant(other)
        return SumExpression([self, ProductExpression([Constant(-1), op])])

    def __rsub__(self, other):
        '''Substraction (reversal) on Expression.

        Parameters
        ----------
        self: Expression
        other: Expression or Constant
        
        Returns
        ------- 
        an sum expression object, the symbolic representation of the of substraction(-) for two objects

        '''        
        op = other if isinstance(other, Expression) else Constant(other)
        return op - self

    def __mul__(self, other):
        '''Multiplication on Expression.

        Parameters
        ----------
        self: Expression
        other: Expression or Constant
        
        Returns
        ------- 
        an product expression object, the symbolic representation of the of multiplication(*) for two objects

        '''        
        op = other if isinstance(other, Expression) else Constant(other)
        return ProductExpression([self, op])

    def __rmul__(self, other):
        '''Multiplication (commutative) on Expression.

        Parameters
        ----------
        self: Expression
        other: Expression or Constant
        
        Returns
        ------- 
        an product expression object, the symbolic representation of the of multiplication(*) for two objects

        '''        
        return self * other

    def __truediv__(self, other):
        '''Division on Expression.

        Parameters
        ----------
        self: Expression
        other: Expression or Constant
        
        Returns
        ------- 
        an division expression object, the symbolic representation of the of division(/) for two objects

        '''        
        op = other if isinstance(other, Expression) else Constant(other)
        return DivisionExpression(self, op)

    def __rtruediv__(self, other):
        '''Division (reversal, up-side-down) on Expression.

        Parameters
        ----------
        self: Expression
        other: Expression or Constant
        
        Returns
        ------- 
        an division expression object, the symbolic representation of the of division(/) for two objects

        '''        
        op = other if isinstance(other, Expression) else Constant(other)
        return DivisionExpression(op, self)

    def __pow__(self, power, modulo=None):
        '''Exponentiation (self as base) on Expression.

        Parameters
        ----------
        self: Expression
        other: Expression or Constant
        
        Returns
        ------- 
        an power expression object, the symbolic representation of the of exponentiation (**) for two objects

        '''     
        op = power if isinstance(power, Expression) else Constant(power)
        return PowerExpression(base=self, exponent=op)

    def __rpow__(self, other):
        '''Exponentiation (self as exponent) on Expression.

        Parameters
        ----------
        self: Expression
        other: Expression or Constant
        
        Returns
        ------- 
        an power expression object, the symbolic representation of the of exponentiation (**) for two objects

        '''     
        op = other if isinstance(other, Expression) else Constant(other)
        return PowerExpression(base=op, exponent=self)

    def __neg__(self):
        '''Negation on Expression.

        Parameters
        ----------
        self: Expression
        
        Returns
        ------- 
        an symbolic representation of the of object with negative (-) sign

        '''   
        return self * -1

## child classes for Expression ##
class Constant(Expression):
    """
    - The constant class, a child class of Expression 
    
    Attributes
	==========
	value : float
		  The value of the constant.

	NOTES
	=====
    1. the constant is evaluated to be the constant itself
    2. the derivative of constant is zero
	3. printing the class instance will result in a str, which can be concatenated into the Expressio

	"""

    def __init__(self, value: float):
        """
		INPUTS
		=======
		value : float
			  The value of value of the constant

		"""


        self.value = value

    def evaluate(self, values):
        '''Evaluate the value of constant with the given values.

        Parameters
        ----------
        self: Constant 
        values: float
        
        Returns
        ------- 
        the numerical value (float) of the constant
        '''        
        return self.value

    def _symdiff(self, respect_to):
        '''Display the symbolic representation of the derivative of constant.

        Parameters
        ----------
        self: Constant
        respect_to: variable symbol (x, y, z, etc), the partial derivative direction
        
        Returns
        ------- 
        the symbolic representation of the derivative of the constant, which is zero, an expression

        '''        
        return Constant(0)

    def __str__(self):
        '''print out the value of constant.

        Parameters
        ----------
        self: Constant
        
        Returns
        ------- 
        str version of the value of constant

        '''     

        return str(self.value)


class Symbol(Expression):
    """
    - The symbol class, a child class of Expression 
    
    Attributes
	==========
	name : str, the symbol of variable (x,y,z, etc.)

	"""
    def __init__(self, name: str):
        """
		INPUTS
		=======
		name: str, the symbol of variable (x,y,z, etc.)

		"""

        self.name = name

    def evaluate(self, values):
        '''Evaluate the corresponding value of the variable symbol 

        Parameters
        ----------
        self: a variable symbol
        values: a list of float
        
        Returns
        ------- 
        the corresponding numerical value (float) of the variable symbol
        '''  
        assert self in values
        return values[self]

    def _symdiff(self, respect_to):
        '''Display the symbolic representation of the derivative of a variable symbol.

        Parameters
        ----------
        self: str, namely a variable symbol
        respect_to: a variable symbol (x, y, z, etc), the partial derivative direction
        
        Returns
        ------- 
        the symbolic representation of the derivative of the symbol, w.r.t. the chosen direction
        if orthongal: returns a constant object 0
        if the direction matches: returns  a constant object of 1

        '''   
        return Constant(1) if respect_to is self else Constant(0)

    def __str__(self):
        '''print out the variable symbol.

        Parameters
        ----------
        self: a variable symbol
        
        Returns
        ------- 
        str version of a variable symbol, i.e.the name attribute

        '''     
        return self.name


class SumExpression(Expression):
    """
    - The SumExpression class, a child class of Expression 
    
    Attributes
	==========
	operands : list of operand that will be combined by the operator

	NOTES
	=====
    1. the evaluation using addition operator is a summing function 
    2. the derivative of addition is homomorphism, i.e.D(a+b) = D(a) + D(b)
	4. printing the class instance will result in a str, which can be concatenated into the Expression

	"""
    def __init__(self, operands: list[Expression]):
        """
		INPUTS
		=======
		operand : list of operand (Expression instance) that will be combined by the operator
			  
		"""

        self.operands = operands

    def evaluate(self, values):
        '''Evaluate the value of addtion operation with the given values for operands.

        Parameters
        ----------
        self: SumExpression 
        values: list of floats
        
        Returns
        ------- 
        the numerical value (float) of the addtion operation 
        '''        
        return sum([i.evaluate(values) for i in self.operands])

    def _symdiff(self, respect_to):
        '''Display the symbolic representation of the derivative of addition operation.

        Parameters
        ----------
        self: SumExpression 
        respect_to: variable symbol (x, y, z, etc), the partial derivative direction
        
        Returns
        ------- 
        the symbolic representation of the derivative of addition operation, perserving homomorphism

        '''        
        return SumExpression([i._symdiff(respect_to) for i in self.operands])

    def __str__(self):
        '''print out the addition expression.

        Parameters
        ----------
        self: SumExpression
        
        Returns
        ------- 
        str version of the addition expression

        '''     
        return '+'.join(['(%s)' % i for i in self.operands])


class ProductExpression(Expression):
    """
    - The ProductExpression class, a child class of Expression 
    
    Attributes
	==========
	operands : list of operand that will be combined by the operator

	NOTES
	=====
    1. the evaluation using multiplication operator is a product operation
    2. the multiplication is commutative, so the order of appending to list can vary. i.e. (xyz)' = [yz,zx,xy] = [zy,xz,yx]
	4. printing the class instance will result in a str, which can be concatenated into the Expression

	"""
    def __init__(self, operands: list[Expression]):
        """
		INPUTS
		=======
		operand : list of operand (Expression instance) that will be combined by the operator
			  
		"""
 
        self.operands = operands

    def evaluate(self, values):
        '''Evaluate the value of multiplication operation with the given values for operands.

        Parameters
        ----------
        self: ProductExpression 
        values: list of floats
        
        Returns
        ------- 
        the numerical value (float) of the multipication operation 
        '''        
        p = 1
        for i in self.operands:
            p *= i.evaluate(values)
        return p

    def _symdiff(self, respect_to):
        '''Display the symbolic representation of the derivative of multiplication operation.

        Parameters
        ----------
        self: ProductExpression 
        respect_to: variable symbol (x, y, z, etc), the partial derivative direction
        
        Returns
        ------- 
        the symbolic representation of the derivative of multiplication operation

        '''    
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
        '''print out the multiplication expression.

        Parameters
        ----------
        self: ProductExpression
        
        Returns
        ------- 
        str version of the multiplication expression

        '''     
        return '*'.join(['(%s)' % i for i in self.operands])

class DivisionExpression(Expression):
    """
    - The DivisionExpression class, a child class of Expression 
    
    Attributes
	==========
	num : numerator of the divison operation, an Expression
	denom : denominator of the divison operation, an Expression


	NOTES
	=====
    1. the evaluation uses multiplication operator is a division operation, numerator divided by the denomenator
	2. printing the class instance will result in a str, which can be concatenated into the Expression

	"""
    def __init__(self, num: Expression, denom: Expression):
        """
		INPUTS
		=======
		num : numerator of the divison operation, an Expression
		denom : denominator of the divison operation, an Expression

		"""
        self.num = num
        self.denom = denom

    def evaluate(self, values):
        '''Evaluate the value of division operation with the given values for operands.

        Parameters
        ----------
        self: DivisionExpression 
        values: list of floats
        
        Returns
        ------- 
        the numerical value (float) of the division operation 
        '''        
        return self.num.evaluate(values) / self.denom.evaluate(values)

    def _symdiff(self, respect_to):
        '''Display the symbolic representation of the derivative of division operation, an illustration of quotient rule

        Parameters
        ----------
        self: ProductExpression 
        respect_to: variable symbol (x, y, z, etc), the partial derivative direction
        
        Returns
        ------- 
        the symbolic representation of the derivative of division operation

        '''  
        return (self.num._symdiff(respect_to) * self.denom - self.num * self.denom._symdiff(respect_to)) / (
            self.denom * self.denom)

    def __str__(self):
        '''print out the division expression.

        Parameters
        ----------
        self: DivisionExpression
        
        Returns
        ------- 
        str version of the division expression

        '''     
        return '(%s)/(%s)' % (self.num, self.denom)


def make_ln_expression(x):
    '''
    helper function:
    Handle the case where x is not an Expression.
    '''
    if isinstance(x, Expression):
        return LnExpression(x)
    
    # Hopefully it's a number.
    return math.log(x)


class LnExpression(Expression):
    """
    - The LnExpression class, a child class of Expression 
    
    Attributes
	==========
	x : an operand, an Expression instance

	NOTES
	=====
    1. the objective is to work with the natural log operation on an expression, or number singleton
    2. the derivative of natural log is in the funtional form of 1/x
	3. printing the class instance will result in a str, which can be concatenated into the Expression

	"""
    def __init__(self, x: Expression):
        """
		INPUTS
		=======
		x : an operand, an Expression instance that we are 'taking log' on
	
		"""
        self.x = x

    def evaluate(self, values):
        '''Evaluate the value of 'taking log' operation with the given values for operands.

        Parameters
        ----------
        self: LnExpression 
        values: a float
        
        Returns
        ------- 
        the numerical value (float) of the 'taking natural log' operation 
        '''
        return math.log(self.x.evaluate(values))

    def _symdiff(self, respect_to):
        '''Display the symbolic representation of the derivative of 'taking log' operation

        Parameters
        ----------
        self: LnExpression 
        respect_to: variable symbol (x, y, z, etc), the partial derivative direction
        
        Returns
        ------- 
        the symbolic representation of the derivative of taking natural log

        '''  
        return self.x._symdiff(respect_to) / self.x

    def __str__(self):
        '''print out the 'taking natural log' expression.

        Parameters
        ----------
        self: LnExpression
        
        Returns
        ------- 
        str version of the 'taking natural log'  expression

        '''     
        return 'ln(%s)' % self.x


class PowerExpression(Expression):
    """
    - The PowerExpression class, a child class of Expression 
    
    Attributes
	==========
	exponent : an operand, an Expression instance, the power that the base is raised to
    base: an operand, an Expression instance, the base that will be raised 

	NOTES
	=====
    1. This class will handle varies cases, constant raised to constant (a**a),variable raised to constant (x**a),
    constant raised to variable (a**x), variable raised to variable (x**x)
    2. This class can handle the square root function, setting exponent = 0.5
	3. printing the class instance will result in a str, which can be concatenated into the Expression

	"""
    def __init__(self, exponent: Expression, base: Expression = Constant(math.e)):
        """
		INPUTS
		=======
		exponent : an operand, an Expression instance, the power that the base is raised to, default to the natural number
        base: an operand, an Expression instance, the base that will be raised 
	
		"""
        # TODO: do it correctly ;)
        # If base is a constant (defined by it can be evaluated without value), replace it with a Constant instance.
        try:
            self.base = Constant(base.evaluate({}))
        except AssertionError:
            self.base = base
        # Do similar thing for exponent
        try:
            self.exponent = Constant(exponent.evaluate({}))
        except AssertionError:
            self.exponent = exponent

    def evaluate(self, values):
        '''Evaluate the value of exponential operation with the given values for operands.

        Parameters
        ----------
        self: PowerExpression 
        values: a float
        
        Returns
        ------- 
        the numerical value (float) of the exponentiation operation 
        '''
        return self.base.evaluate(values) ** self.exponent.evaluate(values)

    def _symdiff(self, respect_to):
        '''Display the symbolic representation of the derivative of exponentiation operation

        Parameters
        ----------
        self: PowerExpression 
        respect_to: variable symbol (x, y, z, etc), the partial derivative direction
        
        Returns
        ------- 
        the symbolic representation of the derivative of exponentiation

        '''  
        # Shortcut for cases where one of exponent, base is constant.
        if isinstance(self.exponent, Constant) and isinstance(self.base, Constant):
            return Constant(0)
        if isinstance(self.exponent, Constant):
            return self.base._symdiff(respect_to) * self.exponent * self.base ** (self.exponent - 1)
        if isinstance(self.base, Constant):
            return self.exponent._symdiff(respect_to) * LnExpression(self.base) * self
        return self * (
            self.exponent._symdiff(respect_to) * LnExpression(self.base) + self.exponent * self.base._symdiff(
                respect_to) / self.base)
            #PowerExpression(self.exponent * LnExpression(self.base))._symdiff(respect_to)

    def __str__(self):
        '''print out the exponentiation expression.

        Parameters
        ----------
        self: PowerExpression
        
        Returns
        ------- 
        str version of the exponentiation expression

        '''     
        return '(%s)^(%s)' % (self.base, self.exponent)


class SinExpression(Expression):
    """
    - The SinExpression class, a child class of Expression 
    
    Attributes
	==========
	x : an operand, an Expression instance

	NOTES
	=====
    1. This class will handle sine function
	2. printing the class instance will result in a str, which can be concatenated into the Expression

	"""
    def __init__(self, x: Expression):
        """
		INPUTS
		=======
		x : an operand, an Expression instance that we are "taking sine" on
	
		"""
        self.x = x

    def evaluate(self, values):
        '''Evaluate the value of "taking sine" operation with the given values for operands.

        Parameters
        ----------
        self: SinExpression 
        values: a float
        
        Returns
        ------- 
        the numerical value (float) of the "taking sine"  operation 
        '''
        return math.sin(self.x.evaluate(values))

    def _symdiff(self, respect_to):
        '''Display the symbolic representation of the derivative of "taking sine" operation

        Parameters
        ----------
        self: SinExpression 
        respect_to: variable symbol (x, y, z, etc), the partial derivative direction
        
        Returns
        ------- 
        the symbolic representation of the derivative of "taking sine"

        '''  
        return self.x._symdiff(respect_to) * CosExpression(self.x)

    def __str__(self):
        '''print out the "taking sine" expression.

        Parameters
        ----------
        self: SinExpression
        
        Returns
        ------- 
        str version of the "taking sine" expression

        '''     
        return 'sin(%s)' % self.x


class CosExpression(Expression):
    """
    - The CosExpression class, a child class of Expression 
    
    Attributes
	==========
	x : an operand, an Expression instance

	NOTES
	=====
    1. This class will handle cos function
	2. printing the class instance will result in a str, which can be concatenated into the Expression

	"""
    def __init__(self, x: Expression):
        """
		INPUTS
		=======
		x : an operand, an Expression instance that we are "taking cosine" on
	
		"""
        self.x = x

    def evaluate(self, values):
        '''Evaluate the value of "taking cosine" operation with the given values for operands.

        Parameters
        ----------
        self: CosExpression 
        values: a float
        
        Returns
        ------- 
        the numerical value (float) of the "taking cosine"  operation 
        '''
        return math.cos(self.x.evaluate(values))

    def _symdiff(self, respect_to):
        '''Display the symbolic representation of the derivative of "taking cosine" operation

        Parameters
        ----------
        self: CosExpression 
        respect_to: variable symbol (x, y, z, etc), the partial derivative direction
        
        Returns
        ------- 
        the symbolic representation of the derivative of "taking cosine"

        ''' 
        return self.x._symdiff(respect_to) * SinExpression(self.x) * -1

    def __str__(self):
        '''print out the "taking cosine" expression.

        Parameters
        ----------
        self: CosExpression
        
        Returns
        ------- 
        str version of the "taking cosine" expression

        '''    
        return 'cos(%s)' % self.x


class TanExpression(Expression):
    """
    - The TanExpression class, a child class of Expression 
    
    Attributes
	==========
	x : an operand, an Expression instance

	NOTES
	=====
    1. This class will handle tangent function
	2. printing the class instance will result in a str, which can be concatenated into the Expression

	"""
    def __init__(self, x: Expression):
        """
		INPUTS
		=======
		x : an operand, an Expression instance that we are "taking tangent" on
	
		"""
        self.x = x

    def evaluate(self, values):
        '''Evaluate the value of "taking tangent" operation with the given values for operands.

        Parameters
        ----------
        self: TanExpression 
        values: a float
        
        Returns
        ------- 
        the numerical value (float) of the "taking tangent"  operation 
        '''
        return math.tan(self.x.evaluate(values))

    def _symdiff(self, respect_to):
        '''Display the symbolic representation of the derivative of "taking tangent" operation

        Parameters
        ----------
        self: TanExpression 
        respect_to: variable symbol (x, y, z, etc), the partial derivative direction
        
        Returns
        ------- 
        the symbolic representation of the derivative of "taking tangent"

        ''' 
        return self.x._symdiff(respect_to) / (CosExpression(self.x) * CosExpression(self.x))

    def __str__(self):
        '''print out the "taking tangent" expression.

        Parameters
        ----------
        self: TanExpression
        
        Returns
        ------- 
        str version of the "taking tangent" expression

        '''    
        return 'tan(%s)' % self.x


class ArcsinExpression(Expression):
    """
    - The ArcsinExpression class, a child class of Expression 
    
    Attributes
	==========
	x : an operand, an Expression instance

	NOTES
	=====
    1. This class will handle inverse sine function
	2. printing the class instance will result in a str, which can be concatenated into the Expression

	"""
    def __init__(self, x: Expression):
        """
		INPUTS
		=======
		x : an operand, an Expression instance that we are "taking arcsin" on
	
		"""
        self.x = x

    def evaluate(self, values):
        '''Evaluate the value of "taking arcsin" operation with the given values for operands.

        Parameters
        ----------
        self: ArcsinExpression 
        values: a float
        
        Returns
        ------- 
        the numerical value (float) of the "taking arcsin"  operation 
        '''
        return math.asin(self.x.evaluate(values))

    def _symdiff(self, respect_to):
        '''Display the symbolic representation of the derivative of "taking arcsin" operation

        Parameters
        ----------
        self: ArcsinExpression 
        respect_to: variable symbol (x, y, z, etc), the partial derivative direction
        
        Returns
        ------- 
        the symbolic representation of the derivative of "taking arcsin"

        ''' 
        return self.x._symdiff(respect_to) * (1 / (1 - self.x * self.x) ** 0.5)

    def __str__(self):
        '''print out the "taking arcsin" expression.

        Parameters
        ----------
        self: ArcsinExpression
        
        Returns
        ------- 
        str version of the "taking arcsin" expression

        '''    
        return 'arcsin(%s)' % self.x


class ArccosExpression(Expression):
    """
    - The ArccosExpression class, a child class of Expression 
    
    Attributes
	==========
	x : an operand, an Expression instance

	NOTES
	=====
    1. This class will handle inverse cosine function
	2. printing the class instance will result in a str, which can be concatenated into the Expression

	"""
    def __init__(self, x: Expression):
        """
		INPUTS
		=======
		x : an operand, an Expression instance that we are "taking arccos" on
	
		"""
        self.x = x

    def evaluate(self, values):
        '''Evaluate the value of "taking arccos" operation with the given values for operands.

        Parameters
        ----------
        self: ArccosExpression 
        values: a float
        
        Returns
        ------- 
        the numerical value (float) of the "taking arccos"  operation 
        '''
        return math.acos(self.x.evaluate(values))

    def _symdiff(self, respect_to):
        '''Display the symbolic representation of the derivative of "taking arccos" operation

        Parameters
        ----------
        self: ArccosExpression 
        respect_to: variable symbol (x, y, z, etc), the partial derivative direction
        
        Returns
        ------- 
        the symbolic representation of the derivative of "taking arccos"

        ''' 
        return self.x._symdiff(respect_to) * (1 / (1 - self.x * self.x) ** 0.5) * -1

    def __str__(self):
        '''print out the "taking arccos" expression.

        Parameters
        ----------
        self: ArccosExpression
        
        Returns
        ------- 
        str version of the "taking arccos" expression

        '''    
        return 'arccos(%s)' % self.x


class ArctanExpression(Expression):
    """
    - The ArctanExpression class, a child class of Expression 
    
    Attributes
	==========
	x : an operand, an Expression instance

	NOTES
	=====
    1. This class will handle inverse tangent function
	2. printing the class instance will result in a str, which can be concatenated into the Expression

	"""
    def __init__(self, x: Expression):
        """
		INPUTS
		=======
		x : an operand, an Expression instance that we are "taking arctan" on
	
		"""
        self.x = x

    def evaluate(self, values):
        '''Evaluate the value of "taking arctan" operation with the given values for operands.

        Parameters
        ----------
        self: ArctanExpression 
        values: a float
        
        Returns
        ------- 
        the numerical value (float) of the "taking arctan"  operation 
        '''
        return math.atan(self.x.evaluate(values))

    def _symdiff(self, respect_to):
        '''Display the symbolic representation of the derivative of "taking arctan" operation

        Parameters
        ----------
        self: ArctanExpression 
        respect_to: variable symbol (x, y, z, etc), the partial derivative direction
        
        Returns
        ------- 
        the symbolic representation of the derivative of "taking arctan"

        ''' 
        return self.x._symdiff(respect_to) * (1 / (self.x * self.x + 1))

    def __str__(self):
        '''print out the "taking arccos" expression.

        Parameters
        ----------
        self: ArccosExpression
        
        Returns
        ------- 
        str version of the "taking arctan" expression

        '''    
        return 'arctan(%s)' % self.x


class SinhExpression(Expression):
    """
    - The ArccosExpression class, a child class of Expression 
    
    Attributes
	==========
	x : an operand, an Expression instance

	NOTES
	=====
    1. This class will handle hyperbolic sine function
	2. printing the class instance will result in a str, which can be concatenated into the Expression

	"""
    def __init__(self, x: Expression):
        """
		INPUTS
		=======
		x : an operand, an Expression instance that we are "taking sinh" on
	
		"""
        self.x = x

    def evaluate(self, values):
        '''Evaluate the value of "taking sinh" operation with the given values for operands.

        Parameters
        ----------
        self: SinhExpression 
        values: a float
        
        Returns
        ------- 
        the numerical value (float) of the "taking sinh"  operation 
        '''
        return math.sinh(self.x.evaluate(values))

    def _symdiff(self, respect_to):
        '''Display the symbolic representation of the derivative of "taking sinh" operation

        Parameters
        ----------
        self: SinhExpression 
        respect_to: variable symbol (x, y, z, etc), the partial derivative direction
        
        Returns
        ------- 
        the symbolic representation of the derivative of "taking sinh"

        ''' 
        return self.x._symdiff(respect_to) * CoshExpression(self.x)

    def __str__(self):
        '''print out the "taking sinh" expression.

        Parameters
        ----------
        self: SinhExpression
        
        Returns
        ------- 
        str version of the "taking sinh" expression

        '''    
        return 'sinh(%s)' % self.x


class CoshExpression(Expression):
    """
    - The CoshExpression class, a child class of Expression 
    
    Attributes
	==========
	x : an operand, an Expression instance

	NOTES
	=====
    1. This class will handle hyperbolic cosine function
	2. printing the class instance will result in a str, which can be concatenated into the Expression

	"""
    def __init__(self, x: Expression):
        """
		INPUTS
		=======
		x : an operand, an Expression instance that we are "taking cosh" on
	
		"""
        self.x = x

    def evaluate(self, values):
        '''Evaluate the value of "taking cosh" operation with the given values for operands.

        Parameters
        ----------
        self: CoshExpression 
        values: a float
        
        Returns
        ------- 
        the numerical value (float) of the "taking cosh"  operation 
        '''
        return math.cosh(self.x.evaluate(values))

    def _symdiff(self, respect_to):
        '''Display the symbolic representation of the derivative of "taking cosh" operation

        Parameters
        ----------
        self: CoshExpression 
        respect_to: variable symbol (x, y, z, etc), the partial derivative direction
        
        Returns
        ------- 
        the symbolic representation of the derivative of "taking cosh"

        ''' 
        return self.x._symdiff(respect_to) * SinhExpression(self.x)

    def __str__(self):
        '''print out the "taking cosh" expression.

        Parameters
        ----------
        self: CoshExpression
        
        Returns
        ------- 
        str version of the "taking cosh" expression

        '''    
        return 'cosh(%s)' % self.x


class TanhExpression(Expression):
    """
    - The TanhExpression class, a child class of Expression 
    
    Attributes
	==========
	x : an operand, an Expression instance

	NOTES
	=====
    1. This class will handle hyperbolic tangent function
	2. printing the class instance will result in a str, which can be concatenated into the Expression

	"""
    def __init__(self, x: Expression):
        """
		INPUTS
		=======
		x : an operand, an Expression instance that we are "taking tanh" on
	
		"""
        self.x = x

    def evaluate(self, values):
        '''Evaluate the value of "taking tanh" operation with the given values for operands.

        Parameters
        ----------
        self: TanhExpression 
        values: a float
        
        Returns
        ------- 
        the numerical value (float) of the "taking tanh"  operation 
        '''
        return math.tanh(self.x.evaluate(values))

    def _symdiff(self, respect_to):
        '''Display the symbolic representation of the derivative of "taking tanh" operation

        Parameters
        ----------
        self: TanhExpression 
        respect_to: variable symbol (x, y, z, etc), the partial derivative direction
        
        Returns
        ------- 
        the symbolic representation of the derivative of "taking tanh"

        ''' 
        return self.x._symdiff(respect_to) / (CoshExpression(self.x) * CoshExpression(self.x))

    def __str__(self):
        '''print out the "taking tanh" expression.

        Parameters
        ----------
        self: TanhExpression
        
        Returns
        ------- 
        str version of the "taking tanh" expression

        '''    
        return 'tanh(%s)' % self.x
