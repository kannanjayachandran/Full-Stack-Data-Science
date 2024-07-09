# Advanced Python

## Functional Programming using python

It is a programming paradigm that emphasizes the use of `pure functions`, `immutable data`, and `declarative programming`. Functional programming separates the data from the functions that operate on the data, while OOPS combines the data and functions together. The core ideas of functional programming are:

- Pure functions (First class functions)
- Immutability of data (Data should not be modified once created)
- Declarative Programming (Using expressions and declarations instead of statements and control flow)
- Lazy evaluation 
- Recursion

### Pure Functions

Pure functions are functions that have no side effects and return a value that depends only on their arguments and not anything on the outside. They are deterministic, meaning that they will always return the same output for the same input. They do not modify the input arguments or any other data outside the function. 

```python
def add(a, b):
    return a + b

print(add(2, 3)) # 5
```

### Lazy evaluation

Lazy evaluation is an evaluation strategy that delays the evaluation of an expression until its value is actually needed. This can help in improving the performance of the program by avoiding unnecessary computations. 

```python
def fib(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fib(n-1) + fib(n-2)

print(fib(10)) # 55
```

Some of the functional programming constructs in python are:

- `map` - Applies a function to all the items in an input list
- `filter` - Filters out items based on a condition
- `reduce` - Reduces a list of items to a single item
- `lambda` - Anonymous functions
- `list comprehension` - A concise way to create lists
- `generator expression` - A concise way to create generators
- `itertools` - Functions to create iterators for efficient looping
- `functools` - Higher-order functions and operations on callable objects
- `partial` - Partial function application

```python   
# map
def square(x):
    return x*x

numbers = [1, 2, 3, 4, 5]       
squared = list(map(square, numbers))
print(squared) # [1, 4, 9, 16, 25]

# filter
def is_even(x):
    return x % 2 == 0

numbers = [1, 2, 3, 4, 5]
even = list(filter(is_even, numbers))
print(even) # [2, 4]

# reduce
from functools import reduce

def add(x, y):
    return x + y

numbers = [1, 2, 3, 4, 5]
sum = reduce(add, numbers)
print(sum) # 15

# lambda
square = lambda x: x*x
print(square(5)) # 25

# list comprehension
numbers = [1, 2, 3, 4, 5]
squared = [x*x for x in numbers]
print(squared) # [1, 4, 9, 16, 25]

# generator expression
numbers = [1, 2, 3, 4, 5]
squared = (x*x for x in numbers)
print(squared) # <generator object <genexpr> at 0x7f8b1c1b3d60>

# itertools
import itertools

numbers = [1, 2, 3, 4, 5]
even = itertools.filterfalse(lambda x: x % 2, numbers)
print(list(even)) # [1, 3, 5]

# functools
from functools import partial

def add(x, y):
    return x + y

add_5 = partial(add, 5)
print(add_5(3)) # 8
```
Refer to the [Functional Programming In python](./01_Functional_programming.ipynb) notebook for more examples.

2. [Decorators](./02_Python_Decorators.ipynb)
3. [Generators](./03_Python_Generator.ipynb)
4. [Iterators](./04_Python_Iterators.ipynb)
5. [Context managers](./05_Context_Managers.ipynb)
6. [Regular Expressions](./06_RegEx_in_Python.ipynb)
7. [Multi-Processing](./07_MultiProcessing_in_Python.ipynb)
8. [Multi-Threading](./07_Multithreading_in_Python.ipynb)
9. [Testing](./Testing/)
10. [Networking](./08_Python_Networking.ipynb)
11. [Advanced OOPS](./09_Advanced_OOP.ipynb)
12. [Interpreter-CPython](./10_Interpreter_CPYTHON_GIL.ipynb)