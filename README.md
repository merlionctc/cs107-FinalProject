# cs107-FinalProject
final project

[![Codecov](https://codecov.io/gh/merlionctc/cs107-FinalProject/branch/master/graph/badge.svg?token=OK5DF8VBMO)](undefined)
[![Build Status](https://travis-ci.com/merlionctc/cs107-FinalProject.svg?token=GoqM45ZYtzxFGATTpCrx&branch=master)](https://travis-ci.com/merlionctc/cs107-FinalProject)

final project
Group 9

Group Member: 
* Jiahui Tang (jiahuitang@g.harvard.edu)
* Wenqi Chen (wenqichen@g.harvard.edu)
* Yujie Cai (ycai@g.harvard.edu)

## Get Started


 ### 1 Installation

There are two ways to install the `AutoDiff` package. 

Method 1 directly install the package through pip into your local environment.

Method 2 create a virtual environment and install the package through pip, and also clones github repo for `demo.py`.

The package is distributed through TestPyPI at https://test.pypi.org/project/autodiff-merlionctc/


* **Method 1: install packaged using pip** 

    To install AutoDiff directly using pip, in the terminal, type:

    ```bash
    pip install -i https://test.pypi.org/simple/ autodiff-merlionctc
    ```

    This will install all required modules as dependency.

    Next, user could start use AutoDiff package by following `demo.py` examples in github repository or **3.2 How to Use** section:
    
    ```python
    >>> from autodiff.model import *
    >>> from autodiff.dual import *
    >>> from autodiff.elementary import *
    >>> from autodiff.symbolic import *
    #（begin autodifferentiation)
    >>> quit()
    ```


* **Method 2: install using virtual environment**
    
    For now, to get started, please do git clone on our project and test by using `demo.py`

    First, download the package from github to your folder

    ```bash
    mkdir test_merlionctc
    cd test_merlionctc
    git clone https://github.com/merlionctc/cs107-FinalProject.git
    cd cs107-FinalProject
    ```

    Create a vertual environment and activate it

    ```bash
    # If you do not have virtualenv, install it
    sudo easy_install virtualenv
    # Create virtual environment
    virtualenv ac207
    # Activate your virtual environment
    source ac207/bin/activate
    ```

    To ensure you have your enviroment and all required package setup, 

    ```bash
    pip install -r requirements.txt
    ```

    To install our package while using 
    
    ```bash
    pip install -i https://test.pypi.org/simple/ autodiff-merlionctc
    ```

    To run a demo we provided.

    ```bash
    python3 AutoDiff/demo.py
    ```
    
    If you want to quit the virtual enviornment:

    ```bash
    deactivate
    ```
    

### 2 How to Use (Forward Mode)

Here is an example that serves that a quick start tutorial on Forward Mode.

** Disclaimer: For usage of reverse mode and symbolic, please refer to Section 6 Extension. **

After installing AutoDiff package (see section 3.1)

```python
>>> from autodiff.dual import *
>>> from autodiff.elementary import *
>>> from autodiff.model import *
>>> from autodiff.symbolic import *
>>> import numpy as np
```

First Step: User instantiate variables
* val: value of variable that you start with
* der: value of the derivative of variable that you start with, usually starting with 1
* loc: The location/index this variable when there are multiple input variables for the target function(s). For example, if you initialize x1 first, the loc will be 0; then you initialize y1, the loc will increment to 1
* length: The length/number of the total variables that will be input when there are multiple input variables for the target function(s).For example, if you want to initialize x1,y1 and z1, the length will be 3, for each variable in the initialization process

```python
>>>x1 = Dual(val = 1, der=1, loc = 0, length = 3)
>>>y1 = Dual(val = 2, der=1, loc = 1, length = 3)
>>>z1 = Dual(val = 5, der=1, loc = 2, length = 3)
```

Second Step: User inputs function, based on above variables
```python
>>>f1 = 3 * x1 + 4 * y1 * 2 - cos(z1)
```

Third Step: User instantiate `autodiff.Forward` class 
```python
>>>fwd_test = Forward(f1)
```

Fourth Step: User could choose to call instance method `get_value()` to get value of func
```python
>>>print(fwd_test.get_value())
18.716337814536775
```

Fifth Step: User could choose to call instance method `get_der()` to get derivatives of func

Note: This method will return a derivative vector w.r.t to ALL variables. 


Note 2: If user enters a scalar function, then get_der will return the jacobian
```python
>>>print(fwd_test.get_der())
[ 3.          8.         -0.95892427]
```

Sixth Step: User could choose to call instance method get_der(var) to get derivatives of func

Note: This method will return a derivative vector w.r.t to specific variables you input

```python
>>>print(fwd_test.get_der(x1))
[3.0]
```

Seventh Step: User could also inputs multiple functions with multiple variables and call get_der() and get_jacobian()
```python
f2 = (tanh(cos(sin(y1))**z1) + logistic(z1**z1, 2, 3, 4))**(1/x1)
f3 = exp(arccos(tan(sin(y1))) + logb(z1**(1/2), 1/5)*sinh(x1))

# User should use list to combine multiple functions together
fwd_test_multiple = Forward([f1, f2, f3])

# User could choose single/several variables to get derivatives
print(fwd_test_multiple.get_der(x1, y1))

# User could get the jacobian matrix of multiple functions
# Note: the order displayed in the Jacobian Matrix is matched with the order of input functions(as row) and the input variables(as column)
print(fwd_test_multiple.get_jacobian())
```

quitting
```python
>>>quit()
```

### 3 How to Use (Reverse Symbolic Mode & Higher Order Derivative)

First Step: User instantiate variables.
You can choose to initialize by wrapper method for multiple variables together.
Or you could initialize indivial symbols by Symbol class.

```python
x, y, z = symbols('x y z')
x1 = Symbol('x1')
```

Second Step: User inputs function, based on above variables
```python
f2 = (tanh(cos(sin(y))**z) + logistic(z**z, 2, 3, 4))**(1/x)
```

Third Step: User input the values of the variables
```Python
values = {x: 2, y: np.pi, z: 4}
```

Fourth Step: User could choose to call instance method evaluate() to get value of func
```Python
print(f2.evaluate(values))
```

Fifth Step: User could choose to call instance method diff() to get first order derivative or higher order derivative of func
*get derivative of f1 with respect to z*
```Python
print(diff(f2, z).evaluate(values))
```

*get second order derivative of f2 with respect to z*
```Python
print(diff(f2, z, z).evaluate(values))
```

*get partial derivative of f2: df2/dxdy*
```Python
print(diff(f2, x, y).evaluate(values))
```

*get third derivative of f with respect to x*
```Python
print(diff(f2, x, x, x).evaluate(values))
```

Sixth Step: User could User could get jacobian/derivatives of multiple functions with multiple variables
```Python
f1 = 3 * x + 4 * y * 2 - z
f3 = exp(arccos(tan(sin(y))) + logb(z**(1/2), 1/5)*sinh(x))
```

User could get Jacobian Matrix with method get_jacobian_value()

*Note: the order displayed in the Jacobian Matrix is matched with the order of input functions(as row) and the input variables(as column)*
```Python
print(get_jacobian_value([f1, f2, f3], [x, y, z], values))
```

Seventh Step: User could get the expression of the function
```Python
print(f1)
```

User could also get the expression of (higher order) derivatives
```Python
print(diff(f2, x))
print(diff(f2, x, y))
```


## Broader Impact and Inclusivity Statement 

### 1 Broader Impact

As a open source library, it is important to care about the broader impact on the social community, especially in terms of diversity, ethical and social impact. 

Above all, we, the developers for AutoDiff, spare no effort to encourage the autonomy and freedom of using and contributing to the library from diversified groups, especially for the women, people with disability and working parents. We hope our library can provide great motivation and encouragement for the minority groups in contributors and for other open source library, contributing to the diversity in the whole open source community.

Additionally, the open source Auto Differentiation library may also be misused in some scenarios and cause some ethical issues. For one thing, there may be some students or researchers to overuse this library instead of calculating the derivatives by hand. Although AutoDiff can be efficient and powerful for solving complicated gradient problems, sometimes it is also vital for students to learn how to conduct the derivatives by hand, and for mathematical/physical researchers to discover drawbacks and make breakthroughs. Therefore, it should be notified that AutoDiff is only a tool, but not a bible to entirely depend on. For the other, there may be some people using AutoDiff for business purpose or offering it for sale, which is entirely against our purpose and privacy rules. It should be warned that our library is designed for academical purpose, increasing the efficiency of Machine Learning tasks and other gradient-related problems.



### 2 Software Inclusivity
Software development, like many fields of science, has been prosperous because the contribution of people from a variety of backgrounds. This package welcomes and encourage participation and usage from a global community. 
Just as Python Software's Diversity Statement indicated, *the Python community is based on mutual respect, tolerance, and encouragement, and we are working to help each other live up to these principles.* We, as the developer of this AutoDiff package, also want our user group to be more diverse: whoever you are, and whatever your background, we welcome you to use our package.
This software package is built based upon the diversity perspective on Python broader community. We strongly believe that embrace diverse community to use our package brings new blood and perspective, making our user group stronger and more vibrant. A diverse user group where all users treat each other with respect has more potential contributors and more sources for fresh ideas.
We also welcomes users from all language background. Mathematics has no boundary.

In principle, there should be no barrier whatsoever for other developers to contribute to our code base. 
In practice, these barriers do exist and could be rather subtle. 

Our software project will be mainly published on Github page, and welcomes contributions from opening issues and pull requests from a broad audiences. 
We will also leave our email (email address could be found at **README.md**) for reaching out for closer contact if any users has ideas on contributing but found it difficult to overcome barriers. With a closer communication, we could reach out to our users and accomodate for their needs accordingly if that would help.

Pull requests will be reviewed and approved together by each of our group members. Emails will also be reviewed and approved together.

For underrepresented group, we welcome contributions from their perspective and are willing to receive their comments and feedbacks. We will also remove sensitive content and words in our code base, for example, we would be taking care of code and variable naming to avoid use of words like `blacklist` or `whitelist`, `master` or `slave`, but to use a more generic term.
We will also rename our main branch to `main`.

For working parents, we will accomodate for their time if they would like a closer contact and discuss about our package implementation. 
For people from different countries or non-native English speakers, rural communities, we would be willing to receive their feedback and contribution through all channels. We could be reached via email, phone, letter or any other means plausible.
We will also use translator to help accomodate for language barriers if it is needed.

For people with disabilities, we will be happy to provide accommodations, inluding but not limiting to sign language explanation and hearing assitance. We are happy to go with more detail if you feel needed.

We want to make our biggest effort to create an inclusive learning and communication environment for all the users and members from the science community that supports a diversity of thoughts, perspectives, and experiences, and honors the identities.
To help achieve it:

* If you feel anything such as the name and the written records that disturb you in this software, please let us know.

* If you feel anything improper in our future maintanance and development process, please let us know

* As we welcome everyone to contribute, we shall all strive to honor the diversity of each other



