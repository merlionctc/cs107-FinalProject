from model import *
from dual_class import *
from elementary import *

# First Step: User instantiate variables
# val: value of variable that you start with
# der: value of the derivative of variable that you start with, usually starting with 1
# loc: The location/index this variable when there are multiple input variables for the target function(s). 
#      For example, if you initialize x1 first, the loc will be 0; then you initialize y1, the loc will increment to 1
# length: The length/number of the total variables that will be input when there are multiple input variables for the target function(s).
#         For example, if you want to initialize x1,y1 and z1, the length will be 3, for each variable in the initialization process
x1 = Dual(val = 1, der=1, loc = 0, length = 3)
y1 = Dual(val = 2, der=1, loc = 1, length = 3)
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

################### Root Finding via Newton's Method #################

val = 0
x2 = Dual(val = val, der=1)

def func(x):   
    f = 2*sin(x) + x**2
    return f

f_grad = Forward(2*sin(x2) + x2**2)

while abs(func(val)) > 1e-5:
    val = val + f_grad.get_der()
    x2 = Dual(val = val, der=1)
    f_grad = Forward(2*sin(x2) + x2**2)

print("Root found! The root is ", x2.val)




#################### Interface (TBC) ############################

# def interface():
#     num = input("Enter the number of variables you need: ")
#     vals = []
#     variables = []
#     for i in range(num):
#         val = input("Enter the value of the variables you want in order: ")
#         variables.append(Dual(val = val, der=1, loc = i, length = num))
    