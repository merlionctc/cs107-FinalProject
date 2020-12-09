#import sys
#sys.path.append('AutoDiff/src/autodiff')

from autodiff.model import *
from autodiff.dual import *
from autodiff.elementary import *
from autodiff.symbolic import *

################### Forward mode implementation ##################
# First Step: User instantiate variables
# val: value of variable that you start with
# der: value of the derivative of variable that you start with, usually starting with 1
# loc: The location/index this variable when there are multiple input variables for the target function(s). 
#      For example, if you initialize x1 first, the loc will be 0; then you initialize y1, the loc will increment to 1
# length: The length/number of the total variables that will be input when there are multiple input variables for the target function(s).
#         For example, if you want to initialize x1,y1 and z1, the length will be 3, for each variable in the initialization process
x1 = Dual(val = 1, der=1, loc = 0, length = 3)
y1 = Dual(val = np.pi, der=1, loc = 1, length = 3)
z1 = Dual(val = 5, der=1, loc = 2, length = 3)

# Second Step: User inputs function, based on above variables
f1 = 3 * x1 + 4 * y1 * 2 - z1

# Third Step: User instantiate AutoDiff.Forward class 
fwd_test = Forward(f1)

# Four Step: User could choose to call instance method get_value() to get value of func
print(fwd_test.get_value())

# Five Step: User could choose to call instance method get_der() to get der of func
# Note: This method will return a derivative vector w.r.t to all variables 
print(fwd_test.get_der())

# Sixth Step: User could choose to call instance method get_der(var) to get der of func
# Note: This method will return a derivative vector w.r.t to specific variables you input
print(fwd_test.get_der(x1))

#Seventh Step: User could get jacobian/derivatives of multiple functions with multiple variables
f2 = (tanh(cos(sin(y1))**z1) + logistic(z1**z1, 2, 3, 4))**(1/x1)
f3 = exp(arccos(tan(sin(y1))) + logb(z1**(1/2), 1/5)*sinh(x1))

# User should use list to combine multiple functions together
fwd_test_multiple = Forward([f1, f2, f3])

# User could choose single/several variables to get derivatives
print(fwd_test_multiple.get_der(x1, y1))

# User could get the jacobian matrix of multiple functions on multiple variables
# Note: the order displayed in the Jacobian Matrix is matched with the order of input functions(as row) and the input variables(as column)
print(fwd_test_multiple.get_jacobian())

################### Symbolic Reverse Mode implementation #####################
# First Step: User instantiate variables
x, y, z = symbols('x y z')

# Second Step: User inputs function, based on above variables
f2 = (tanh(cos(sin(y))**z) + logistic(z**z, 2, 3, 4))**(1/x)

# Third Step: User input the values of the variables
values = {x: 2, y: np.pi, z: 4}

# Fourth Step: User could choose to call instance method evaluate() to get value of func
print(f2.evaluate(values))

# Fifth Step: User could choose to call instance method diff() to get der or higher order derivative of func
# get derivative of f1 with respect to z
print(diff(f2, z).evaluate(values))

# get second order derivative of f2 with respect to z
print(diff(f2, z, z).evaluate(values))

#get partial derivative of f2: df2/dx1dy1
print(diff(f2, x, y).evaluate(values))

#get third derivative of f with respect to x1
print(diff(f2, x, x, x).evaluate(values))

# Sixth Step: User could User could get jacobian/derivatives of multiple functions with multiple variables
f1 = 3 * x + 4 * y * 2 - z
f3 = exp(arccos(tan(sin(y))) + logb(z**(1/2), 1/5)*sinh(x))

# User could get Jacobian Matrix with method get_jacobian_value()
# Note: the order displayed in the Jacobian Matrix is matched with the order of input functions(as row) and the input variables(as column)
print(get_jacobian_value([f1, f2, f3], [x, y, z], values))

#Seventh Step: User could get the expression of the function
print(f1)

# User could also get the expression of (higher order) derivatives
print(diff(f2, x))


################### Root Finding via Newton's Method using Forward Mode #################
val = 1
x2 = Dual(val = val, der=1)
f_grad = Forward(2*sin(x2) + x2**2)

while abs(f_grad.get_value()[0]) > 1e-5:
    x2 = x2 - f_grad.get_value()[0]/f_grad.get_der()[0]
    f_grad = Forward(2*sin(x2) + x2**2)

print("Root found! The root is ", x2.val)

################### Root Finding via Newton's Method using Symbolic Reverse Mode#################
x = symbols('x')
value = {x: 1}
f_grad2 = 2*sin(x) + x**2

while abs(f_grad2.evaluate(value)) > 1e-5:
    val = val - f_grad2.evaluate(value)/diff(f_grad2, x).evaluate(value)
    value = {x: val}
    #f_grad = Forward(2*sin(x) + x**2)

print("Root found! The root is ", val)



    