<h1 align="center" > Python Programming </h1>

**Python** is an **open-source**, **high-level** and **general-purpose** programming language. It is **dynamically type-checked** (type safety of a program is verified at runtime) and **garbage-collected**.

> Note: We are using Python 3

## Table of Contents

1. **Python Fundamentals**
    - [Python Syntax and Indentation](#Python-Syntax)
    - [Variables and Data types](#variables-and-data-types)
    - [Type conversion and Casting](#type-conversion-and-casting)
    - [Basic Input/Output](#basic-inputoutput)
    - [Operators](#operators)

2. **Control Flow**
    - [Conditional Statements](#conditional-statements)
    - [Looping](#loops)
    - [Loop control statements](#loop-control-statements)

3. **Functions**
    - [Defining and Calling functions](#defining-and-calling-functions)
    - [Key Points about Python Functions](#key-points-about-python-functions)
    - [Lambda functions](#lambda-functions)
    - [Scope and namespace](#scope-and-namespace)

4. **Python Memory Model and Variable Behavior**
    - [Variables and Object References](#variables-and-object-references)
    - [Memory Optimization and Object Reuse](#memory-optimization-and-object-reuse)
    - [Automatic Memory Management and Garbage Collection](#automatic-memory-management-and-garbage-collection)
    - [Python’s Argument Passing Model](#pythons-argument-passing-model)
    - [Boxing and Unboxing](#boxing-and-unboxing)
    - [Integer Caching Optimization](#integer-caching-optimization)
    - [Handling Large Integers (No Overflow!)](#handling-large-integers-no-overflow)
    - [Cython: Bypassing Python’s Overhead](#cython-bypassing-pythons-overhead)

4. **Built-in Data Structures**
    - [Strings](#string)
    - [Lists](#list)
    - [Tuples](#tuple) 
    - [Sets](#set)
    - [Dictionaries](#dictionary)

5. **File Handling**
    - [Common File Modes](#common-file-modes)
    - [Reading and Writing Text Files](#reading-and-writing-text-files)
    - [Working with Binary Files](#working-with-binary-files)
    - [Using Context Managers for file handling](#using-context-managers-with-open)
    - [Working with File Paths](#working-with-file-paths)
    - [File Iteration and Buffering](#file-iteration-and-buffering)
    - [Advanced File Operations](#advanced-file-operations)
    - [Performance Optimization](#performance-optimization)
    - [Exception Handling in File Operations](#exception-handling-in-file-operations)

6. **Error Handling**
    - [Error vs Exception](#errors-vs-exceptions)
    - [Types of Errors](#types-of-errors)
    -[Exception Handling](#basic-exception-handling-try-except-else-finally)
    -[Handling multiple exceptions](#handling-multiple-exceptions)
    -[Built-in exceptions](#common-built-in-exceptions)
    - [Nested and re-raised exceptions](#nested-and-re-raised-exceptions)
    - [Custom exceptions](#custom-exceptions)
    - [Exception chaining](#exception-chaining-from-keyword)
    - [Exception logging](#logging-exceptions)

7. **Modules and Packages**
    - [Modules](#modules)
    - [Importing modules](#importing-modules)
    - [Module search path](#module-search-path-syspath)
    - [Packages](#packages)
    - [Importing from packages](#importing-from-packages)
    - [Standard Library](#standard-library)

8. Object-Oriented Programming (OOP)

* Classes and Objects
* `__init__`, `__str__`, `__repr__`
* Instance vs Class vs Static methods
* Inheritance and method overriding
* Encapsulation and abstraction
* Magic/Dunder methods (e.g., `__add__`, `__len__`, `__eq__`, `__getitem__`)
* `@property` decorator

9. Advanced OOP

* MRO (Method Resolution Order) and `super()`
* Abstract Base Classes (`abc` module)
* Multiple inheritance
* Metaclasses and dynamic class creation

10. Functional Programming Concepts

* Higher-order functions
* `map()`, `filter()`, `reduce()`
* `zip()`, `enumerate()`
* Generator functions and expressions
* Iterators and the `__iter__`, `__next__` protocol
* Decorators (function decorators, class decorators)
* Closures

11. Asynchronous Programming

* `async` and `await`
* `asyncio` module
* Event loop, tasks, coroutines
* Async generators and context managers

12. Python-Specific Features

* Type hinting and annotations (`PEP 484`, `mypy`)
* `dataclasses`
* `NamedTuple`, `TypedDict`
* Context managers (`__enter__`, `__exit__`)
* Descriptors
* `__slots__`
* Dynamic code execution (`eval`, `exec`)
* Introspection and Reflection (`getattr`, `hasattr`, `dir`, `vars`)

13. Performance and Optimization

* Profiling tools (`cProfile`, `timeit`, `line_profiler`)
* Caching (`functools.lru_cache`)
* Memory optimization tips (`__slots__`, generators)
* Multithreading vs Multiprocessing (`threading`, `multiprocessing`, `concurrent.futures`)
* GIL (Global Interpreter Lock)

14. Development Tools & Practices

* Working with virtual environments (`venv`, `conda`, `pipenv`)
* Package management (`pip`, `conda`, `requirements.txt`)
* Writing testable code
* Unit Testing (`unittest`, `pytest`)
* Debugging (`pdb`, `breakpoint()`)
* Structuring Python projects
* `setup.py`, `pyproject.toml`
* Creating your own packages
* Publishing to PyPI

15. Web Development Tools

* Flask or FastAPI basics
* REST API creation and deployment
* Data science APIs with FastAPI + Swagger docs
* Streamlit

--- 

## Python Syntax

Python emphasizes code readability and relies on indentation to define code blocks instead of using braces `{}` like many other programming languages.

```python
print("Hello, World!")

if 5 > 2:
    print("Five is greater than two!")
```

We can write a single line comment using `#` and multi-line comments using triple quotes `'''` or `"""`.

```python
# This is a comment

'''
This is a multi-line comment
'''

"""
This is also a multi-line comment
"""
```

---

## Variables and Data types

Variables are containers for storing data values. Python is dynamically typed, meaning you don’t need to declare a variable’s type; it is determined automatically based on the value assigned.

```python
x = 5
y = "Hello, World!"
```

Python provides a variety of built-in data types, categorized as follows:

- **Text Type**: 
    - `str`

- **Numeric Types**: 
    - `int`
    - `float`
    - `complex`

- **Sequence Types**: 
    - `list`  (mutable sequence of items) 
    - `tuple` (immutable sequence of items)
    - `range` (sequence of numbers generated on demand)

- **Mapping Type**: 
    - `dict` (key-value pairs)

- **Set Types**: 
    - `set` (unordered collection of unique items)
    - `frozenset` (immutable version of `set`)

- **Boolean Type**: 
    - `bool` (represents `True` or `False`)

- **Binary Types**: 
    - `bytes`  (immutable sequence of bytes)
    - `bytearray` (mutable sequence of bytes)
    - `memoryview` (views over memory buffers)

- **None Type**:
    - `None` (represents the absence of a value)

---

## Type Conversion and Casting

Type conversion, also known as type casting, involves changing the data type of a value. In Python, this can occur implicitly or explicitly

### Implicit Type Conversion (Coercion)

Python automatically converts one data type to another without the programmer's intervention. This usually happens when performing operations between different data types, ensuring no data loss occurs. For instance, when adding an integer and a float, Python will convert the integer to a float before performing the addition.

```python
num_int = 123
num_float = 1.23
num_new = num_int + num_float
print("datatype of num_int:",type(num_int))
print("datatype of num_float:",type(num_float))
print("Value of num_new:",num_new)
print("datatype of num_new:",type(num_new))
```

### Explicit Type Conversion (Casting)

Programmers manually convert the data type using built-in functions. This is necessary when specific data types are required for certain operations or when data loss is acceptable.

- `int()`: Converts to an integer.

- `float()`: Converts to a floating-point number.

- `str()`: Converts to a string.

- `list()`, `tuple()`, `set()`: Converts to list, tuple and set, respectively.

```python
x = 5
y = float(x)
z = str(x)
```

### `type()` function

Python provides the `type()` function to check the type of an object.

```python
x = 5
print(type(x))
# Output: <class 'int'>
```

---

## Basic Input/Output

We can use the `input()` function to take user inputs and use `print()` function to display text, variables and expressions on the console. By default `input` function takes user input as string.

```py
name = input("Enter your name: ")
print("Hello ", name)

x, y, z = input("Enter the x-y-z coordinates: ").split()
```

> There are other ways to get input like `stdin` and `stdout` in `sys` module.

---

## Operators

Python provides a wide range of operators to perform various operations. These are categorized as follows:

### Arithmetic Operators

Used to perform basic mathematical operations.

| Operator | Description | Example | Output |
| --- | --- | --- | --- |
| `+` | Addition | 5 + 3 | 8 |
| `-` | Subtraction | 5 - 3 | 2 |
| `*` | Multiplication | 5 * 3 | 15 |
| `/` | Division | 10 / 2 | 5 |
| `%` | Modulus | 10 % 3 | 1 |
| `//` | Floor Division | 10 // 3 | 3 |
| `**` | Exponentiation | 5 ** 3 | 125 |

### Comparison (Relational) Operators

Used to compare values and return a Boolean (`True` or `False`).

| Operator | Description | Example | Output |
| --- | --- | --- | --- |
| `==` | Equal to | `5 == 3` | `False` |
| `!=` | Not Equal to | `5 != 3` | `True` |
| `>` | Greater Than | `5 > 3` | `True` |
| `<` | Less Than | `5 < 3` | `False` |
| `>=` | Greater Than or Equal To | `5 >= 3` | `True` |
| `<=` | Less Than or Equal To | `5 <= 3` | `False` |

### Assignment Operators

Used to assign values to variables and perform shorthand operations.

| Operator | Description | Example | Equivalent |
| --- | --- | --- | --- |
| `=` | Assign | `x = 5` | `x = 5` |
| `+=` | Add and Assign | `x += 3` | `x = x + 3` |
| `-=` | Subtract and Assign | `x -= 3` | `x = x - 3` |
| `*=` | Multiply and Assign | `x *= 3` | `x = x * 3` |
| `/=` | Divide and Assign | `x /= 3` | `x = x / 3` |
| `%=` | Modulus and Assign | `x %= 3` | `x = x % 3` |
| `//=` | Floor Division and Assign | `x //= 3` | `x = x // 3` |
| `**=` | Exponentiation and Assign | `x **= 3` | `x = x ** 3` |

### Logical Operators

Used to combine conditional statements.

| Operator | Description | Example | Output |
| --- | --- | --- | --- |
| `and` | Logical AND | `True and False` | `False` |
| `or` | Logical OR | `True or False` | `True` |
| `not` | Logical NOT | `not True` | `False` |

### Bitwise Operators

Operate on binary representations of integers.

| Operator | Description | Example | Output |
| --- | --- | --- | --- |
| `&` | Bitwise AND | `5 & 3` | `1` |
| `\|` | Bitwise OR | `5 \| 3` | `7` |
| `^` | Bitwise XOR | `5 ^ 3` | `6` |
| `~` | Bitwise NOT | `~5` | `-6` |
| `<<` | Left Shift | `5 << 1` | `10` |
| `>>` | Right Shift | `5 >> 1` | `2` |

- Bitwise XOR (`^`) operator returns true if the bits are different, otherwise returns false.

- Bitwise NOT operator flips the bits of the number. It can also change the sign of the number.

    - In most programming languages negative numbers are represented using the 2's complement system. To find the decimal value of a 2's complement binary number:
        * If the leading bit is zero; The number is positive and you can directly convert the binary to decimal.
        * If the leading bit is one; The number is negative. To find its magnitude:
            1. Invert all bits.
            2. Add one to the result.
            3. The decimal value is the negative of this result.

- Left shift operator shifts the bits to the left by the specified number of positions, filling the rightmost bits with zeros. Mathematically, this is equivalent to multiplying the number by 2 raised to the power of the number of positions shifted.

- Right shift operator shifts the bits to the right by the specified number of positions, discarding the rightmost bits. Mathematically, this is equivalent to dividing the number by 2 raised to the power of the number of positions shifted.

### Membership Operators

Used to check if a value is part of a sequence (e.g., string, list, tuple, etc.).

| Operator | Description | Example | Output |
| --- | --- | --- | --- |
| `in` | Present in | `5 in [1, 2, 3, 4, 5]` | `True` |
| `not in` | Not Present in | `5 not in [1, 2, 3, 4, 5]` | `False` |

### Identity Operators

Used to compare the memory locations of two objects.

| Operator | Description | Example | Output |
| --- | --- | --- | --- |
| `is` | Same Object | `x is y` | `True` |
| `is not` | Different Object | `x is not y` | `False` |

---

## Conditional Statements

Conditional statements in Python are used to execute specific blocks of code based on logical conditions. Python supports the following conditional statements:

### `if` statement

Executes a block of code if the condition evaluates to `True`.

```python
x = 10
if x > 5:
    print("x is greater than 5")
# Output: x is greater than 5
```

### `elif` Statement

Allows checking multiple conditions. It is short for "else if."

```python
x = 10
if x > 15:
    print("x is greater than 15")
elif x > 5:
    print("x is greater than 5 but less than or equal to 15")
# Output: x is greater than 5 but less than or equal to 15
```

### `else` Statement

Executes a block of code if none of the preceding conditions are `True`.

```python
x = 2
if x > 5:
    print("x is greater than 5")
else:
    print("x is 5 or less")
# Output: x is 5 or less
```

### Nested `if` Statement

Allows placing an if statement inside another if statement to check multiple conditions hierarchically.

```python
x = 10
if x > 5:
    if x % 2 == 0:
        print("x is greater than 5 and even")
    else:
        print("x is greater than 5 and odd")

# Output: x is greater than 5 and even
```

### Ternary Operator

```python
result = a if condition else b
```

> If condition is True, the value of `a` is assigned to result. Otherwise, the value of `b` is assigned.

```python
x = 10
y = 20
max_value = x if x > y else y
print(max_value)  # Output: 20
```

---

## Loops

Loops in Python are used to execute a block of code repeatedly as long as a condition is met or for each item in a sequence. Python supports the following loop constructs:

### while loops

A while loop runs as long as its condition evaluates to True.

````python
i = 1
while i <= 5:
    print(i)
    i += 1
# Output: 
# 1
# 2
# 3
# 4
# 5
````

### for loops

A `for` loop iterates over a sequence (like a list, tuple, string, or range).

````python
for i in range(1, 6):
    print(i)
# Output: 
# 1
# 2
# 3
# 4
# 5
````
## Loop control statements

We can use the `break` statement to stop the loop before it has looped through all the items, and the `continue` statement to stop the current iteration of the loop, and continue with the next.

Python loops also have something like $for → else$ and $while → else$ which is executed when the loop is finished without a `break` statement.

````python
# for else
for i in range(1, 6):
    if i == 7:
        break
else:
    print("Loop completed without a break")
# Output: Loop completed without a break

# while else
i = 1
while i <= 5:
    i += 1
    if i == 2:
        continue
    print(i)
else:
    print("Loop completed without a break")
# 3
# 4
# 5
# 6
# Loop completed without a break
````

---

## Functions

A function is a reusable block of code designed to perform a specific task. Functions allow modularity, code reuse, and better organization of programs. In Python, functions are defined using the `def` keyword.

### Defining and Calling functions

The syntax for defining a function:

````python
def function_name(parameters):
    # Function body
    return value

def greet(name):
    return f"Hello, {name}"
````

The syntax for calling a function is:

```py
function_name(arguments)

print(greet("Doe"))
```

> `Parameters` are what the function expects to receive (the variables listed inside the parentheses in the function definition). `Arguments` are what you actually give to the function (the actual values that are passed to the function when you call it).

```python
def add(a, b):  # 'a' and 'b' are parameters
    return a + b

print(add(5, 10))  # '5' and '10' are arguments
```

### Key Points about Python Functions

1. **Multiple Return Values**:

Python functions can return multiple values as a tuple.

````python
def calculate(a, b):
    return a + b, a - b

sum_result, diff_result = calculate(10, 5)
print(sum_result)  # Output: 15
print(diff_result) # Output: 5
````

2. **Default Parameters**:

Functions can have default values for parameters.

```python
def greet(name="Guest"):
    return f"Hello, {name}!"

print(greet())          # Output: Hello, Guest!
print(greet("Alice"))   # Output: Hello, Alice!
```

3. **Variable-length Arguments**:

- ***args**: Allows passing a variable number of positional arguments.

- ****kwargs**: Allows passing a variable number of keyword arguments.

```python
def sum_numbers(*args):
    return sum(args)

print(sum_numbers(1, 2, 3, 4))  # Output: 10

def display_info(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

display_info(name="Alice", age=25)
# Output:
# name: Alice
# age: 25
```

4. **Stack and Heap Memory**:

Stack and Heap here refers to two memory allocation concepts. Stack is used for static memory allocation; for storing function calls and local variables, while heap is used for dynamic memory allocation; storing objects and data that persist beyond a single function call.

```py
def example_function(arg1, arg2, *args, kw_only_arg, **kwargs):
    print(f"arg1: {arg1}")
    print(f"arg2: {arg2}")
    print(f"args: {args}")
    print(f"kw_only_arg: {kw_only_arg}")
    print(f"kwargs: {kwargs}")

# Calling the function
example_function(1, 2, 3, 4, 5, kw_only_arg="hello", name="Alice", age=30)
```

### Lambda Functions

Lambda functions are small, anonymous functions defined using the `lambda` keyword. They are limited to a single expression and are often used for short-term tasks.

```python
lambda arguments: expression
```
- Lambda functions are particularly useful with higher-order functions like `map()`, `filter()`, and `reduce()`.

```python
nums = [1, 2, 3, 4]
squared = list(map(lambda x: x ** 2, nums))
print(squared)  # Output: [1, 4, 9, 16]
```

> Refer **[_Function Notebook_](./Notebooks/8_Functions.ipynb)** for additional content

### Scope and Namespace

Namespace and scope are fundamental concepts in every programming languages. In python they govern how variables and names are organized and accessed within a program.

- A `namespace` is a system that ensures all names in a program are unique and avoids naming conflicts. It's a collection of names (identifiers) mapped to their corresponding objects. Namespaces are implemented as dictionaries in Python.

- `Scope` refers to the region of a program where a particular namespace is directly accessible. It determines the visibility and lifetime of names within that region.

**Python has several scopes**:

- **Local (Function)**: Variables defined inside a function have local scope. They are only accessible within that function.

- **Enclosing (Nonlocal)**: If a function is defined inside another function (nested function), the inner function can access variables from the outer function's scope. This is the enclosing scope.

- **Global (Module)**: Variables defined at the top level of a module (outside any function or class) have global scope. They can be accessed from anywhere within the module.

- **Built-in**: This scope contains pre-defined functions and constants that are always available in Python.

### LEGB Rule

When you refer to a name in your Python code, the interpreter searches for that name in a specific order through different scopes. This order is known as the LEGB rule:

- L: Local: Search the local scope first.

- E: Enclosing: If the name is not found locally, search the enclosing function's scope.

- G: Global: If not found in the enclosing scope, search the global scope.

- B: Built-in: Finally, if the name is not found in any of the previous scopes, search the built-in scope.

> If a name is not found in any of these scopes, Python raises a `NameError`.

### Global and Nonlocal

- **`global`**: Declares a variable inside a function as referring to the global scope.

- **`nonlocal`**: Used in nested functions to refer to variables in the enclosing (non-global) scope.

```python
x = 10  # global

def outer():
    y = 5  # enclosing
    def inner():
        nonlocal y
        global x
        y += 1
        x += 1
        print("Inner y:", y, "Global x:", x)
    inner()
    print("Outer y:", y)

outer()
print("Final x:", x)
```

---

## Python Memory Model and Variable Behavior

Before diving into **Object-Oriented Programming (OOP)** in Python, it's essential to grasp how the language manages **variables**, **references**, **objects**, and **memory**. Understanding these fundamental concepts is also crucial for discussing **complex data structures** or **performance optimization**. This knowledge will clarify some of Python's unique or "*quirky*" behaviors.

In Python, everything is an **object** — from integers and strings to lists and functions. Variables act as **references** (**bindings**) to these objects, not as direct storage locations for data.

### Variables and Object References

When you assign a value to a variable, Python creates an object in memory and binds the variable name to it.

```py
n = 300
```

Here:

- An **integer object** with value **300** is created.

- The variable `n` references this object (it holds the memory address, not the value itself).

Every Python object has a unique identifier, typically its memory address, which can be checked using `id()`.

```py
n = 300
print(id(n))
```

When you reassign a variable, it simply points to a new object.

```py
n = 300
n = "foo"
```

Now, `n` no longer refers to the integer `300`; it refers to the string "`foo`".

<p align="center"><img src="./img/Pointer_animation.gif" alt="Pointer visualization"></p>

### Memory Optimization and Object Reuse

Python optimizes memory usage by reusing immutable objects with the same value.

```py
a = 10
b = 10
print(a is b)  # True (same object)
```

Both `a` and `b` reference the same integer object in memory.

- Always use `is` to compare **object identity** (same object in memory), and `==` for **value equality**.

<p align="center"><img src="./img/Pointer_animation_2.gif" alt="Shared reference"></p>

---

### Automatic Memory Management and Garbage Collection

Python’s memory manager handles:

1. **Object allocation** on the heap

1. **Reference counting**

1. **Garbage collection** for unreferenced objects

Each object keeps track of how many variables reference it. When the reference count drops to zero, Python automatically frees that memory. You can manually delete a reference using `del`:

```py
x = [1, 2, 3]
del x  # reduces reference count
```

Garbage collection is handled by Python’s built-in **gc module**, which also removes cyclic references (e.g., objects referencing each other).

- Understand reference counting is crucial for debugging memory leaks

- Use tools like `sys.getrefcount()` and the gc module to analyze **memory usage**.

---

### Python’s Argument Passing Model

A common confusion in Python:

> “Is Python **pass-by-value** or **pass-by-reference**?”

The answer is **neither**.

Python uses **Parameter passing mechanism** or (**Pass by Object Reference** (or **Pass by Assignment**)).

This means:

- The function receives a reference to the object, not the object itself.

- The behavior depends on mutability of the object.

#### Immutable Objects (Pass-by-Value-like Behavior)

When you pass an immutable object (int, str, tuple), reassigning it inside a function creates a new object.

```py
def modify_value(x):
    x += 10
    print("Inside function:", x)

a = 5
modify_value(a)
print("Outside function:", a)

# Inside function: 15
# Outside function: 5
```

- The original variable `a` remains unchanged.

#### Mutable Objects (Pass-by-Reference-like Behavior)

When you pass a **mutable object** (list, dict, set), modifications inside the function affect the original.

```py
def modify_list(lst):
    lst.append(10)
    print("Inside function:", lst)

nums = [1, 2, 3]
modify_list(nums)
print("Outside function:", nums)

# Inside function: [1, 2, 3, 10]
# Outside function: [1, 2, 3, 10]
```

- The function directly modifies the list object referenced by `nums`.

---

### Boxing and Unboxing

Since everything in Python is an object, even primitive-looking types (like `int` and `float`) are actually **boxed objects**.

- **Boxing** : Wrapping a raw value inside an object

- **Unboxing** : Extracting the underlying value from an object for operations

```py
a = 10
b = 20
c = a + b
```

Behind the scenes:

1. Python verifies the operand types (`int`).

1. Calls the appropriate magic method (`__add__`).

1. Unboxes the integer values (`10`, `20`).

1. Performs the addition.

1. Boxes the result (`30`) back into a new integer object.

Although flexible, this process adds overhead, making Python slower than low-level languages like **C**.

---

### Integer Caching Optimization

For performance, Python preallocates and caches small integers in the range [**-5, 256**].

```py
a = 100
b = 100
print(a is b)  # True (cached)

x = 1000
y = 1000
print(x is y)  # False (new objects)
```

This optimization reduces object creation overhead, as these values are frequently used in most programs.

---

### Handling Large Integers (No Overflow!)

Unlike many languages, Python’s integers are arbitrary-precision. They can grow as large as memory allows — no overflow errors.

```py
huge = 10**100
print(huge)
```

Internally, Python dynamically allocates additional memory to store large numbers, using a variable-length representation.

---

### Cython: Bypassing Python’s Overhead

For performance-critical sections, tools like **Cython** allow compiling Python code to **C** for near-native speed.

```py
cpdef int add(int x, int y):
    cdef int result
    result = x + y
    return result
```

By declaring types, **Cython** avoids Python’s dynamic type checks and boxing/unboxing overhead, leading to **significant performance gains**.

---

## Built-in Data Structures

### String

A string is an immutable sequence of Unicode characters enclosed in single quotes (`'`), double quotes (`"`), or triple quotes (`''' """`). 

- Strings are immutable (cannot be modified once created). Any operation that alters a string creates a new string object.

- Strings are ordered, indexed (starting at `0` and also support negative indexing), iterable and can contain duplicate elements.

- Strings can be sliced (`[start : stop : step]`) and supports membership operations.

#### String Operations

- Concatenation: `+` operator

- Repetition: `*` operator

- Membership Testing: `in`, `not in`

- Comparison: Lexicographic (based on Unicode code points)

```py
a, b = "Hello", "World"
print(a + " " + b)   # Hello World
print(a * 3)         # HelloHelloHello
print("H" in a)      # True
print("hello" < "world")  # True (lexicographic order)
```

#### Common String Methods

1. **Case Conversion** 
    - `.upper()`, `.lower()`, `.capitalize()`, `.title()`, `.swapcase()`

2. **Searching and Checking**
    - `.find()`, `.rfind()`, `.index()`, `.startswith()`, `.endswith()`

    - `.count()`, `.isalnum()`, `.isalpha()`, `.isdigit()`, `.isspace()`, `.istitle()`

3. **Modification**
    - `.replace()`, `.strip()`, `.lstrip()`, `.rstrip()`

    - `.center()`, `.ljust()`, `.rjust()`

    - `.zfill()`

4. **Splitting and Joining**
    - `.split()`, `.rsplit()`, `.splitlines()`

    - `.join(iterable)`

5. **Unicode Encodings**
    - `ord(char)` → Returns Unicode code point of a character

    - `chr(code)` → Returns character for a Unicode code point

    - `encode()` → Converts string to bytes

    - `decode()` → Converts bytes to string

    ```py
    print(ord("A"))     # 65
    print(chr(65))      # A

    s = "Python 🐍"
    encoded = s.encode("utf-8")
    decoded = encoded.decode("utf-8")
    print(encoded)  # b'Python \xf0\x9f\x90\x8d'
    print(decoded)  # Python 🐍
    ```

#### f-Strings

> Python 3.6+

Efficient and easier way to do string formatting.

```py
print(f"Name: {name}, Age: {age}")
print(f"Next year: {age + 1}")
```

#### Performance Considerations

- String concatenation with `+` inside loops is inefficient (creates new objects each time). Use `str.join()` or `io.StringIO` instead.

```py
# Inefficient
res = ""
for i in range(1000):
    res += str(i)

# Efficient
res = "".join(str(i) for i in range(1000))
```

- Time complexity of string operations
    - Indexing : $O(1)$
    - Slicing : $O(k)$ where `k` is the length of the slice
    - Concatenation : $O(n)$ for strings of length `n`

- Python caches small strings, commonly short identifiers, called string interning.

```py
a = "hello"
b = "hello"
print(a is b)  # True (due to interning)

a = "Strings in Python are powerful and versatile."
b = "Strings in Python are powerful and versatile."
print(a is b)  # False
```

**[String Notebook](./Notebooks/7_String.ipynb)**

---

### List

Python lists are **ordered, mutable collections** of objects. They can store heterogenous types.

- Lists are indexed, dynamic and allows duplicate elements. 

- Defined using square brackets `[]` or the `list()` constructor.

```python
nums = [1, 2, 3, 4]
mixed = [1, "hello", 3.14, [5, 6]]

print(nums[0])     # 1
print(nums[-1])    # 4
print(nums[1:3])   # [2, 3]
```

#### List Operations

- Concatenation: `+`

- Repetition: `*`

- Membership Testing: `in`, `not in`

- Unpacking: Direct assignment of elements

```py
a = [1, 2, 3]
b = [4, 5]
print(a + b)     # [1, 2, 3, 4, 5]
print(a * 2)     # [1, 2, 3, 1, 2, 3]

x, y, z = a
print(x, y, z)   # 1 2 3
```

#### Common List Methods

- **Adding / Removing Elements**

    - `.append(x)` → Add element at end

    - `.extend(iterable)` → Add all elements from iterable

    - `.insert(i, x)` → Insert at position i

    - `.remove(x)` → Remove first occurrence

    - `.pop([i])` → Remove and return element at index i (last by default)

    - `.clear()`→ Remove all elements

- **Searching and Counting**

    - `.index(x, [start], [end])` → Find index of first occurrence

    - `.count(x)` → Count occurrences

- **Sorting and Reversing**

    - `.sort(key=None, reverse=False)` → In-place sort

    - `sorted(iterable, key=None, reverse=False)` → Returns a new sorted list

    - `.reverse()` → Reverse in-place

    - `reversed(list)` → Returns an iterator

- **Copying**

    - `.copy()` or `[:]` or `list()` : Shallow copy (It creates a new list container but simply copies the references to the items within the original list. Both lists therefore point to the same internal objects.) Therefore modifying a mutable object within the shallow-copied list will also change the original list

    - `copy.deepcopy()` : A deep copy creates a completely independent new list. It recursively duplicates all objects it encounters, from the list itself to all the objects contained within it, and all the objects within those objects, and so on. Changes made to the deep-copied list will not affect the original list, and vice versa.

    ```py
    import copy
    a = [[1, 2], [3, 4]]
    b = a.copy()
    c = copy.deepcopy(a)

    a[0][0] = 99
    print(b)  # [[99, 2], [3, 4]]  (affected)
    print(c)  # [[1, 2], [3, 4]]   (independent)
    ```

#### List comprehension

```py
squares = [x**2 for x in range(5)]
print(squares)  # [0, 1, 4, 9, 16]

evens = [x for x in range(10) if x % 2 == 0]
print(evens)    # [0, 2, 4, 6, 8]
```

#### Nested List

```py
matrix = [[1, 2], [3, 4], [5, 6]]
print(matrix[1][0])  # 3
```

#### Useful functions with lists

- `enumerate(list)` → Returns index + value pairs

- `zip(list1, list2)` → Combines lists element-wise

- `*args` unpacking in function calls

#### Performance considerations

Lists are implemented as dynamic arrays in CPython. Common time complexities are;

| Operation                     | Complexity                      |
| ----------------------------- | ------------------------------- |
| Indexing                      | `O(1)`                          |
| Append                        | `O(1)` (amortized)              |
| Pop (end)                     | `O(1)`                          |
| Insert / Pop (anywhere else)  | `O(n)`                          |
| Membership test (`x in list`) | `O(n)`                          |
| Slicing                       | `O(k)` where `k` = slice length |
| Sort                          | `O(n log n)`                    |

- For large numeric arrays, prefer `array.array` or `numpy.ndarray` for efficiency.

**[List Notebook](./Notebooks/3_List.ipynb)**

---

### Tuple

Tuples are an immutable, ordered sequence type in Python. They are widely used for grouping related data, ensuring data integrity, and improving performance when immutability is desirable.

- They can be defined using parenthesis (**(** **)**) or built-in **tuple()** constructor.

- They allow duplicate elements and can store heterogenous data types.

```py
# Creating tuples
t1 = (1, 2, 3)
t2 = ("apple", "banana", "cherry")
t3 = (1, "hello", 3.14, True)

# Single element tuple (needs trailing comma!)
t_single = (5,)  # not (5)
```

#### Tuple Operations

- Indexing and slicing are same as lists

```py
t = (10, 20, 30, 40, 50)
print(t[0])    # 10
print(t[-1])   # 50
print(t[1:4])  # (20, 30, 40)
```

- Concatenation and repetition

```py
a = (1, 2)
b = (3, 4)
print(a + b)   # (1, 2, 3, 4)
print(a * 3)   # (1, 2, 1, 2, 1, 2)
```

- Membership test

```py
print(2 in a)   # True
print(5 not in b)  # True
```

#### Tuple methods

- Tuples support only two built-in methods (since they are immutable):
    1. `.count(value)` : Returns occurrences of a value
    2. `.index(value)` : Returns first index of the value

    ```py
    t = (1, 2, 2, 3, 4)
    print(t.count(2))   # 2
    print(t.index(3))   # 3
    ```

#### Packing and Unpacking

- **Tuple Packing**: Assign multiple values at once.

- **Tuple Unpacking**: Extract values into variables.

```py
# Packing
point = (3, 4)

# Unpacking
x, y = point
print(x, y)  # 3 4

# Extended unpacking
a, *b, c = (1, 2, 3, 4, 5)
print(a, b, c)  # 1 [2, 3, 4] 5
```

#### Tuple immutability in detail

Tuples can contain other tuples or even mutable objects. Therefore the tuples itself are immutable, but if they hold mutable object, that object can still be changed. If a given tuple only contains immutable data, they would be hashable and hence can be used as dictionary keys.

```py
coords = {}
coords[(10, 20)] = "A"
coords[(15, 25)] = "B"
print(coords)  # {(10, 20): 'A', (15, 25): 'B'}
```

#### Named Tuples (Collections)

`Collections` module provide a factory function called `namedtuple`, that creates tuple subclasses with named fields. It offers a way to combine the *immutability* and *memory efficiency* of regular tuples with enhanced readability of accessing elements by name instead of numerical index.

```py
from collections import namedtuple

# Define a named tuple type for a Point
Point = namedtuple('Point', ['x', 'y'])

# Create an instance of the Point named tuple
p = Point(10, 20)

# Access elements by name
print(f"X coordinate: {p.x}")
print(f"Y coordinate: {p.y}")

# Access elements by index (still works)
print(f"First element by index: {p[0]}")
```

#### Performance notes

- **Memory efficient** : Tuples uses less memory than lists

- **Faster Iteration** : Due to their immutability, Python can optimize their usage.

- **Tuple Interning** : Small immutable tuples may be cached by python, similar to *string interning*

**[Tuple Notebook](./Notebooks/4_Tuple.ipynb)**

---

### Set

A set is an unordered, mutable, and un-indexed collection of unique elements in Python. Sets are useful for membership testing, removing duplicates, and performing mathematical set operations and allows faster insertion, deletion and searching. They are implemented using hash tables.

- Elements of set must be hashable (immutable).

```py
# Creating sets
s1 = {1, 2, 3, 4}
s2 = set([3, 4, 5, 6])  # converting list to set

print(s1)   # {1, 2, 3, 4}
print(s2)   # {3, 4, 5, 6}

# Empty set
empty = set()   # not {}
```

#### Set Operations

- **Membership Test** (Very fast, O(1) average time)

```py
s = {10, 20, 30}
print(10 in s)   # True
print(40 not in s)  # True
```

- **Union**

```py
a = {1, 2, 3}
b = {3, 4, 5}
print(a | b)       # {1, 2, 3, 4, 5}
print(a.union(b))  # {1, 2, 3, 4, 5}
```

- **Intersection**

```py
print(a & b)                # {3}
print(a.intersection(b))    # {3}
```

- **Difference**

```py
print(a - b)  # {1, 2}
print(b - a)  # {4, 5}
```

- Symmetric Difference

```py
print(a ^ b)  # {1, 2, 4, 5}
```

#### Set Methods

- **Adding / Removing elements**

```py
s = {1, 2}
s.add(3)         # {1, 2, 3}
s.update([4, 5]) # {1, 2, 3, 4, 5}

s.remove(2)      # removes element, raises KeyError if not present
s.discard(10)    # safe remove (no error if not found)
s.pop()          # removes a random element
s.clear()        # empties the set
```

- **Copy**

```py
s1 = {1, 2, 3}
s2 = s1.copy()
```

#### Frozen set (Immutable set)

It is the immutable version of set. Since frozen set itself is hashable, it can be used as dictionary keys or elements of another set.

```py
fs = frozenset([1, 2, 3])
print(fs)          # frozenset({1, 2, 3})
# fs.add(4)           # Error: 'frozenset' object has no attribute 'add'
```

#### Advanced usage

- **Set Comprehension**

```py
s = {x**2 for x in range(5)}
print(s)  # {0, 1, 4, 9, 16, 25}
```

- **Subset / Superset checks**

```py
a = {1, 2}
b = {1, 2, 3}
print(a.issubset(b))    # True
print(b.issuperset(a))  # True
```

- Suitable for **large-scale lookups** and **deduplication**.

- Sets are extremely helpful in situations like doing a membership test for blacklist/whitelist or deduplicating user ID's/logs. 

- We use sets for tracking visited nodes in graph algorithms.

**[Set Notebook](./Notebooks/6_Set.ipynb)**

---

### Dictionary

A dictionary is an *unordered* (From Python 3.7 onwards, Dictionaries are officially insertion ordered), **mutable**, and **key-value pair** data structure in Python. It allows for fast lookups, insertions, and deletions, with average O(1) time complexity.

- Defined using curly braces `{}` or the `dict()` constructor.

- Keys must be unique and immutable (e.g., strings, numbers, tuples).

- Values can be any object (mutable or immutable).

```py
# Creating dictionaries
person = {"name": "Alice", "age": 25, "city": "London"}

# Using dict() constructor
info = dict(language="Python", version=3.12)

# Empty dictionary
empty = {}
```

#### Accessing and Modifying Values

```py
person = {"name": "Alice", "age": 25, "city": "London"}

# Accessing values
print(person["name"])       # Alice
print(person.get("age"))    # 25
print(person.get("salary", "Not Found"))  # Default value

# Modifying values
person["age"] = 26
person["country"] = "UK"   # Adding new key-value pair
print(person)
```

- `get()` is preferred for safe access to avoid KeyError.

#### Dictionary Operations

- **Membership tests**

```py
print("name" in person)      # True
print("Alice" in person)     # False
```

- **Deleting entries**

```py
del person["city"]        # Removes key
removed = person.pop("age")   # Removes key and returns its value
person.clear()             # Empties dictionary
```

#### Dictionary methods

| Method                      | Description                                   | Example                        |
| --------------------------- | --------------------------------------------- | ------------------------------ |
| `.keys()`                   | Returns view of keys                          | `dict.keys()`                  |
| `.values()`                 | Returns view of values                        | `dict.values()`                |
| `.items()`                  | Returns key-value pairs as tuples             | `dict.items()`                 |
| `.update(other)`            | Merges another dictionary                     | `d1.update(d2)`                |
| `.popitem()`                | Removes and returns last inserted pair        | `d.popitem()`                  |
| `.copy()`                   | Returns a shallow copy                        | `d2 = d.copy()`                |
| `.setdefault(key, default)` | Inserts key with default value if not present | `d.setdefault('role', 'user')` |

```py
student = {"name": "Bob", "grade": "A"}

# Using items()
for key, value in student.items():
    print(f"{key}: {value}")
```

#### Dictionary Comprehensions

It is similar to list comprehension, but produce key-value pairs.

```py
squares = {x: x**2 for x in range(5)}
print(squares)  # {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}

# filtering example 
nums = {x: x**2 for x in range(10) if x % 2 == 0}
print(nums)  # {0: 0, 2: 4, 4: 16, 6: 36, 8: 64}
```

#### Nested Dictionaries

Dictionaries can hold other dictionaries as values, allowing for hierarchical data representation.

```py
employees = {
    "E001": {"name": "Alice", "dept": "HR"},
    "E002": {"name": "Bob", "dept": "IT"}
}

print(employees["E002"]["name"])  # Bob
```

#### Merging Dictionaries (Python 3.9+)

```py
d1 = {"a": 1, "b": 2}
d2 = {"b": 3, "c": 4}
merged = d1 | d2   # {'a': 1, 'b': 3, 'c': 4}
```

#### Dictionary Unpacking

```py
d3 = {**d1, **d2}
```

#### Defaultdict (from collections)

```py
from collections import defaultdict

grades = defaultdict(list)
grades["Alice"].append(90)
grades["Bob"].append(85)
print(grades)  # defaultdict(<class 'list'>, {'Alice': [90], 'Bob': [85]})
```

- Dictionaries are implemented using hash tables, average complexity is $O(1)$ for lookup, insertion and deletion and $O(n)$ for iteration.

**[Dictionary Notebook](./Notebooks/5_Dictionary.ipynb)**

---

## File Handling

File handling allows Python programs to interact with files stored on disk — reading, writing, and manipulating data persistently.

In Python, files are opened using the built-in `open()` function:

```python
file = open("example.txt", "r")  # 'r' = read mode
content = file.read()
print(content)
file.close()
```

### Common File Modes

| Mode   | Description        | Notes                                  |
| ------ | ------------------ | -------------------------------------- |
| `'r'`  | Read (default)     | File must exist                        |
| `'w'`  | Write              | Overwrites existing content            |
| `'a'`  | Append             | Adds new data at the end               |
| `'x'`  | Exclusive creation | Fails if file exists                   |
| `'r+'` | Read and write     | File must exist                        |
| `'w+'` | Write and read     | Overwrites file                        |
| `'a+'` | Append and read    | Creates file if missing                |
| `'b'`  | Binary mode        | Used with other modes (`'rb'`, `'wb'`) |

### Reading and Writing Text Files

- **Reading a File**

```py
with open("notes.txt", "r") as f:
    data = f.read()  # Reads entire file
    print(data)

# other read methods
f.readline()   # Reads one line
f.readlines()  # Reads all lines as a list
```

- **Writing to a file**

```py
with open("output.txt", "w") as f:
    f.write("Hello, World!\n")
    f.write("This overwrites existing content.")
```

- **Appending to a file**

```py
with open("output.txt", "a") as f:
    f.write("\nNew line appended.")
```

### Working with Binary Files

Binary files (e.g., images, audio, executables) require reading/writing in binary mode ('`b`').

```py
# Copying an image file
with open("photo.jpg", "rb") as src:
    data = src.read()

with open("photo_copy.jpg", "wb") as dest:
    dest.write(data)
```

- Python does not interpret binary data — it simply reads and writes bytes (`b'\x...'`).

### Using Context Managers (`with open(...)`)

Using `with` ensures that files are automatically closed after the code block is executed, even if an exception occurs.

> **Context managers** in Python are objects that define a runtime context for use with the `with` statement. They provide a mechanism for **automatically managing resources**, ensuring that setup and cleanup operations are performed correctly, even if errors occur within the code block. Once we cover OOP, we will come back to Context managers in depth.


### Working with File Paths

Python provides two main modules for handling file paths:

- **Using `os` module**

```py
import os

print(os.getcwd())           # Get current directory
print(os.listdir("."))       # List files
print(os.path.exists("file.txt"))  # Check if file exists
print(os.path.join("folder", "file.txt"))  # Join paths safely
```

- **Using `pathlib` (Modern Way)**

```py
from pathlib import Path

p = Path("example.txt")
if p.exists():
    print(p.read_text())   # Directly read file content
p.write_text("This is new text")  # Write to file

# Directory operations
folder = Path("data")
folder.mkdir(exist_ok=True)
```

- `pathlib` is **object-oriented**, more readable, and recommended for new Python code.

### File Iteration and Buffering

When working with large files, reading the entire content into memory (`read()`) is inefficient.

- **Line-by-line Iteration**

```py
with open("bigfile.txt", "r") as f:
    for line in f:
        print(line.strip())
```

- This approach reads one line at a time, conserving memory.

- **Buffered Reading (Binary Data)**

```py
with open("video.mp4", "rb") as f:
    chunk_size = 4096
    while chunk := f.read(chunk_size):
        process(chunk)
```

- This pattern is common in **data streaming** or **network applications**.

### Advanced File Operations

- **File Positioning**

You can control the read/write pointer using `seek()` and `tell()`:

```py
f = open("sample.txt", "r")
print(f.tell())      # Current position
f.seek(0)            # Move to start
print(f.readline())  # Read first line
f.close()
```

- **File Metadata**

```py
import os
info = os.stat("sample.txt")
print(f"Size: {info.st_size} bytes")
print(f"Modified: {info.st_mtime}")
```

- **File Deletion and Renaming**

```py
import os
os.rename("old.txt", "new.txt")
os.remove("new.txt")
```

### Performance Optimization

- Use **Buffered I/O** (`io.BufferedReader`, `io.BufferedWriter`) for large data streams.

- Prefer **context managers** over manual `open()`/`close()` for safety.

- Avoid frequent small writes — instead, accumulate data and write in chunks.

- Use binary mode for non-text data to prevent encoding overhead.

- Use memory mapping (`mmap`) for high-speed file access:

```py
import mmap

with open("largefile.txt", "r+") as f:
    with mmap.mmap(f.fileno(), 0) as mm:
        print(mm.readline().decode())
```

- Memory mapping treats file content as a **byte array** in memory — ideal for large file manipulation.

### Exception Handling in File Operations

Always handle exceptions — file operations are prone to runtime errors (missing file, permissions, I/O failures).

```py
try:
    with open("config.yaml", "r") as file:
        data = file.read()
except FileNotFoundError:
    print("File not found.")
except PermissionError:
    print("Permission denied.")
except Exception as e:
    print(f"Unexpected error: {e}")
```

---

## Error Handling

Error handling in Python allows developers to gracefully detect, respond to, and recover from unexpected events that occur during program execution.

### Errors vs Exceptions

- An **error** is an issue in a program that causes abnormal termination. 

- An **exception** is a special event raised during execution when an error occurs. We can **catch** and **handle** these exceptions dynamically.

### Types of Errors

| Type | Description | Example |
|------|--------------|----------|
| **Syntax Errors** | Detected during parsing (before execution). | `if True print("Hi")` |
| **Runtime Errors (Exceptions)** | Detected during execution; can be caught and handled. | `1 / 0` (ZeroDivisionError) |



### Basic Exception Handling (`try`, `except`, `else`, `finally`)

The `try` statement allows testing a block of code for errors, while `except` handles them.

```python
try:
    x = int(input("Enter a number: "))
    result = 10 / x
except ZeroDivisionError:
    print("❌ Cannot divide by zero.")
except ValueError:
    print("❌ Please enter a valid number.")
else:
    print("✅ Division successful:", result)
finally:
    print("🧹 Execution completed.")
```

- `try` : code that might raise an exception
- `exception` : Code that runs if an exception occurs
- `else` : Code that runs only if no exception occurs
- `finally` : Code that always runs (cleanup, closing files, releasing resources)

### Handling Multiple Exceptions

We can catch multiple exceptions in one block using a tuple:

```py
try:
    val = int("abc")
except (ValueError, TypeError) as e:
    print(f"Error occurred: {e}")
```

### Common Built-in Exceptions

| Exception           | Description                                                                          |
| ------------------- | ------------------------------------------------------------------------------------ |
| **ValueError**        | Raised when an operation receives an argument of right type but inappropriate value. |
| **TypeError**         | Raised when an operation or function is applied to an object of inappropriate type.  |
| **KeyError**          | Raised when a dictionary key is not found.                                           |
| **IndexError**        | Raised when accessing an invalid list or tuple index.                                |
| **ZeroDivisionError** | Raised when dividing by zero.                                                        |
| **FileNotFoundError** | Raised when a file or directory is missing.                                          |
| **AttributeError**    | Raised when an invalid attribute reference occurs.                                   |
| **ImportError**       | Raised when an import fails.                                                         |
| **RuntimeError**      | Raised for generic runtime errors.                                                   |
| **ArithmeticError** | Raised when an error occurs in numeric calculations |
| **AssertionError** | Raised when an `assert` statement fails |
| **Exception** | Base class for all exceptions |
| **EOFError** | Raised when the `input()` method hits an "end of file" condition (EOF) |
| **FloatingPointError** | Raised when a floating-point calculation fails |
| **GeneratorExit** | Raised when a generator is closed (with the `close()` method) |
| **IndentationError** | Raised when indentation is not correct |
| **KeyboardInterrupt** | Raised when the user presses Ctrl+c, Ctrl+z or Delete |
| **LookupError** | Raised when errors raised can't be found |
| **MemoryError** | Raised when a program runs out of memory |
| **NameError** | Raised when a variable does not exist |
| **NotImplementedError** | Raised when an abstract method requires an inherited class to override the method |
| **OSError** | Raised when a system related operation causes an error |
| **OverflowError** | Raised when the result of a numeric calculation is too large |
| **ReferenceError** | Raised when a weak reference object does not exist |
| **StopIteration** | Raised when the `next()` method of an iterator has no further values |
| **SyntaxError** | Raised when a syntax error occurs |
| **TabError** | Raised when indentation consists of tabs or spaces |
| **SystemError** | Raised when a system error occurs |
| **SystemExit** | Raised when the `sys.exit()` function is called |
| **UnboundLocalError** | Raised when a local variable is referenced before assignment |
| **UnicodeError** | Raised when a unicode problem occurs |
| **UnicodeEncodeError** | Raised when a unicode encoding problem occurs |
| **UnicodeDecodeError** | Raised when a unicode decoding problem occurs |
| **UnicodeTranslateError** | Raised when a unicode translation problem occurs |

### Nested and Re-raised Exceptions

You can handle exceptions at multiple levels or re-raise them for higher-level handling.


```py
def divide(a, b):
    try:
        return a / b
    except ZeroDivisionError as e:
        print("Caught inside function:", e)
        raise  # Re-raise exception

try:
    divide(10, 0)
except ZeroDivisionError:
    print("Handled again at caller level.")
```

### Custom Exceptions

Custom exceptions are user-defined classes derived from Python’s `Exception` base class.

```py
class NegativeValueError(Exception):
    """Raised when input is negative."""
    pass

def square_root(x):
    if x < 0:
        raise NegativeValueError("Negative value not allowed.")
    return x ** 0.5

try:
    print(square_root(-9))
except NegativeValueError as e:
    print(f"Custom Exception: {e}")
```

- Always inherit from `Exception`, not from `BaseException`, to avoid interfering with system-level exceptions like `KeyboardInterrupt`.

### Exception Chaining (from Keyword)

Python allows linking related exceptions using `raise ...` `from ....`

```py
try:
    int("abc")
except ValueError as e:
    raise RuntimeError("Conversion failed.") from e
```

- This helps preserve traceback context across multiple layers of failure.

### Cleanup and Resource Management

Use the `finally` block or **context managers** (`with` statement) for safe cleanup.

```py
try:
    f = open("data.txt")
    data = f.read()
finally:
    f.close()  # Ensures closure even on exception

# Pythonic code 
with open("data.txt") as f:
    data = f.read()
```

- `with` automatically closes the resource using `__enter__` and `__exit__` methods, ensuring no resource leaks.

### Logging Exceptions

Instead of printing, log exceptions.

```py
import logging

logging.basicConfig(level=logging.ERROR)

try:
    1 / 0
except ZeroDivisionError as e:
    logging.exception("An error occurred")
```

- This records stack traces with timestamps — crucial for debugging in deployed systems.

### Performance and Memory Insights

| Aspect                    | Details                                                                            |
| ------------------------- | ---------------------------------------------------------------------------------- |
| **Raising Exceptions**    | Slightly expensive due to stack unwinding; avoid in performance-critical loops.    |
| **Control Flow**          | Don’t use exceptions for regular control logic — prefer condition checks.          |
| **Garbage Collection**    | Exception tracebacks hold references to frames; large exception logs can delay GC. |
| **Optimized Alternative** | For repeated validation, use `if` checks instead of raising exceptions repeatedly. |

- Use `try` blocks only around risky code segments, not entire functions.
- When debugging, inspect the exception object with `repr(e)` or `traceback` module.
- Use contextlib.suppress to ignore specific exceptions intentionally:

```py
from contextlib import suppress

with suppress(FileNotFoundError):
    os.remove("nonexistent.txt")
```

---

## Modules and Packages

Python’s modular architecture allows code to be divided into **modules** (single files) and **packages** (collections of modules). This promotes reusability, organization, and maintainability.

### Modules

A **module** is simply a Python file (`.py`) containing variables, functions, or classes that can be imported into other programs.

```py
# math_utils.py
def add(a, b):
    return a + b

def multiply(a, b):
    return a * b
```

```py
import math_utils

print(math_utils.add(5, 3))
print(math_utils.multiply(4, 2))
```

#### Importing modules

- **Basic import**

```py
import math
print(math.sqrt(16))
```

- **Import with Alias**

```py
import numpy as np
np.array([1, 2, 3])
```

- **Import Specific Items**

```py
from math import sqrt, pi
print(sqrt(25), pi)
```

- **Import All** (⚠️ Not Recommended)

```py
from math import *
```

- Avoid `import *` to prevent namespace pollution and ambiguity.

#### Module Search Path (`sys.path`)

When importing, Python searches modules in a specific order defined by `sys.path`.

```py
import sys
print(sys.path)
```

Typical search order:

1. Current directory.

1. `PYTHONPATH` environment variable.

1. Standard library directories.

1. Installed site-packages.

You can add custom paths dynamically:

```py
import sys
sys.path.append("/path/to/your/modules")
```

### Packages

A package is a directory containing a special file named `__init__.py`, which marks it as a package.

- **Example structure**

```sh
project/
│
├── main.py
└── mypackage/
    ├── __init__.py
    ├── module1.py
    └── module2.py
```

```py
# mypackage/module1.py
def greet():
    print("Hello from module1!")
```

```py
# main.py
from mypackage import module1
module1.greet()
```

- The `__init__.py` file can also include initialization code and can controls imports when using `from package import *`.

```py
# mypackage/__init__.py
from .module1 import greet
```

```py
# now we can directly do 
from mypackage import greet
greet()
```

#### Importing from Packages

You can import deeply nested modules using dot notation.

```py
from mypackage.subpackage.module import function
```

Or import the entire package and access modules hierarchically.

```py
import mypackage.subpackage.module
mypackage.subpackage.module.function()
```

#### Reloading a Module

If you modify a module during runtime, use `importlib.reload()` to reload it.

```py
import importlib, mymodule
importlib.reload(mymodule)
```

#### Relative vs Absolute Imports

```py
# Absolute path
from mypackage.module1 import greet
```

```py
# Relative import
from .module1 import greet
from ..subpackage.module2 import func
```
> Relative imports only work in packages, not standalone scripts.

#### Packaging for Distribution

You can distribute Python packages using `setuptools`.

```py
from setuptools import setup, find_packages

setup(
    name="mypackage",
    version="1.0.0",
    packages=find_packages(),
)
```

Build and install

```sh
python setup.py sdist bdist_wheel
pip install .
```

### Standard Library

Python comes with a rich **standard library** covering file handling, math, system operations, networking, and more.

| Module        | Common Uses                                   |
| ------------- | --------------------------------------------- |
| `os`          | Interacting with operating system, file paths |
| `sys`         | System-specific parameters and environment    |
| `math`        | Mathematical operations                       |
| `random`      | Random number generation                      |
| `datetime`    | Working with dates and times                  |
| `json`        | Parsing and serializing JSON data             |
| `re`          | Regular expressions                           |
| `collections` | Specialized container datatypes               |
| `itertools`   | Efficient looping utilities                   |

### Performance and Design Notes

- Python caches imported modules in `sys.modules`. Re-importing uses the cache for performance (import caching).
- For heavy modules, consider on-demand imports to optimize **startup time** (Lazy loading).
- Circular import occur when two modules depend on each other; can be fixed by restructuring or local imports.

---

## Object-Oriented Programming (OOP) in Python

Object-Oriented Programming (OOP) is a programming paradigm based on the concept of *objects*, which encapsulate data (attributes) and behavior (methods).

In Python, **everything is an object** — this includes integers, strings, functions, classes, and even modules. Each object is an instance of a class and has:

- **Identity**: Unique ID (obtained via `id()`)
- Type: The class it belongs to (obtained via `type()`)
- Value: The data it holds

```py
x = 42
print(type(x))      # <class 'int'>
print(isinstance(x, object))    # True

def my_func():
    pass

print(type(my_func))     # <class 'function'>
print(isinstance(my_func, object))  # True

# Even classes are objects (instances of 'type')
class MyClass:
    pass

print(type(MyClass))     # <class 'type'>
print(isinstance(MyClass, object))  # True
```

### Core OOP Principles

- **Encapsulation**: Bundling data and methods that operate on that data within a single unit (class)
- **Abstraction**: Hiding complex implementation details and exposing only essential features
- **Inheritance**: Creating new classes from existing ones to promote code reuse
- **Polymorphism**: Ability to use a common interface for different underlying data types

### Class and Object

- **Class:** A blueprint or template that defines the structure and behavior of objects.
- **Object:** An instance of a class containing real data.

```python
class Dog:
    """A simple Dog class demonstrating basic OOP concepts"""
    
    # Class attribute (shared by all instances)
    species = "Canis familiaris"
    
    def __init__(self, name, age):
        """Instance initializer (constructor)"""
        # Instance attributes (unique to each instance)
        self.name = name
        self.age = age
    
    def bark(self):
        """Instance method"""
        return f"{self.name} says Woof!"

# Creating objects (instances)
dog1 = Dog("Buddy", 3)
dog2 = Dog("Lucy", 5)

print(dog1.name)          # Buddy
print(dog1.species)       # Canis familiaris
print(dog1.bark())        # Buddy says Woof!

# Class attributes are shared
print(dog1.species is dog2.species)  # True (same object in memory)
```

#### Instance vs Class Attributes

- Class attributes are shared across all instances
- Instance attributes are unique to each object
- Accessing a class attribute through an instance creates a lookup chain: `instance → class`
- Assigning to an attribute through an instance creates an instance attribute (doesn't modify class attribute)

Functions defined inside a class are known as **methods**. 

- *classes* in Python are **callable objects** that return instances.

### Methods

Methods are Functions defined inside a class. They can access and modify the data associated with the object.

#### Types of Methods in Python OOP

- **Instance Methods**

Operate on instance data and have access to instance data (`self`).

```py
class BankAccount:
    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance
    
    def deposit(self, amount):
        """Instance method - modifies instance state"""
        if amount > 0:
            self.balance += amount
            return f"Deposited ${amount}. New balance: ${self.balance}"
        return "Invalid amount"
    
    def withdraw(self, amount):
        """Instance method with validation"""
        if 0 < amount <= self.balance:
            self.balance -= amount
            return f"Withdrew ${amount}. New balance: ${self.balance}"
        return "Insufficient funds or invalid amount"

account = BankAccount("Alice", 1000)
print(account.deposit(500))   # Deposited $500. New balance: $1500
print(account.withdraw(200))  # Withdrew $200. New balance: $1300
```

- Instance method can acutally access class data via `self.__class__`

- **Class Methods**

Operate on the class data, not instances. Use the `@classmethod` decorator and `cls` as the first parameter. Often used as **factory methods** to create objects in alternative ways, or to access and modify class variables.

```py
class Employee:
    company = "TechCorp"
    employee_count = 0
    
    def __init__(self, name, salary):
        self.name = name
        self.salary = salary
        Employee.employee_count += 1
    
    @classmethod
    def get_employee_count(cls):
        """Class method - accesses class attributes"""
        return f"{cls.company} has {cls.employee_count} employees"
    
    @classmethod
    def from_string(cls, emp_string):
        """Alternative constructor pattern"""
        name, salary = emp_string.split('-')
        return cls(name, int(salary))
    
    @classmethod
    def set_company_name(cls, name):
        """Modify class attribute"""
        cls.company = name

# Using class methods
emp1 = Employee("Alice", 50000)
emp2 = Employee("Bob", 60000)

print(Employee.get_employee_count())  # TechCorp has 2 employees

# Alternative constructor
emp3 = Employee.from_string("Charlie-55000")
print(emp3.name)  # Charlie

# Modifying class attribute
Employee.set_company_name("NewTech")
print(Employee.company)  # NewTech
```

- **Static Methods**

Behave like normal functions but belong to the class’s namespace. Defined with `@staticmethod`.

```py
class MathOperations:
    """Collection of mathematical utilities"""
    
    @staticmethod
    def add(x, y):
        """Static method - no access to instance or class"""
        return x + y
    
    @staticmethod
    def is_even(num):
        """Pure utility function"""
        return num % 2 == 0
    
    @staticmethod
    def validate_positive(value):
        """Validation helper"""
        if value <= 0:
            raise ValueError("Value must be positive")
        return True

# Using static methods (no instance needed)
print(MathOperations.add(5, 3))        # 8
print(MathOperations.is_even(10))      # True

# Can also call through instance (but not recommended)
math = MathOperations()
print(math.add(2, 3))                  # 5
```

- Use **class methods for factory patterns** and **static methods for utility/helper functions**.

### Magic Methods or Special Methods

Magic Methods or Special Methods or Dunder Methods have predefined meanings and are invoked implicitly by Python in specific situations.

- **`__init__`** :  It is called immediately after object creation. It's used to initialize instance attributes
    - `__init__` is not a constructor, it's an initializer
    - `self` refers to the instance being initialized
    - `__new__` is the constructor and its rarely overridden

```py
class Person:
    def __init__(self, name, age=0):
        """Initialize person with validation"""
        if not isinstance(name, str):
            raise TypeError("Name must be a string")
        if age < 0:
            raise ValueError("Age cannot be negative")
        
        self.name = name
        self.age = age

# Valid initialization
person1 = Person("Alice", 30)

# Using default argument
person2 = Person("Bob")

# Invalid initialization
try:
    person3 = Person(123)  # TypeError
except TypeError as e:
    print(f"Error: {e}")
```


- **`__str__`** and **`__repr__`** (String representations) : `__str__` is the human-readable representation for end users, while, `__repr__` is the unambiguous representation for developers (ideally, evaluating it should recreate the object)
    - Always implement `__repr__`; implement `__str__` only if a different user-facing representation is needed
    - Use `!r` in f-strings within `__repr__` to get `repr()` of attributes

```py
class Book:
    def __init__(self, title, author, year):
        self.title = title
        self.author = author
        self.year = year
    
    def __str__(self):
        """User-friendly string representation"""
        return f'"{self.title}" by {self.author}'
    
    def __repr__(self):
        """Developer-friendly representation (should be unambiguous)"""
        return f'Book(title={self.title!r}, author={self.author!r}, year={self.year})'

book = Book("1984", "George Orwell", 1949)

# __str__ is called by print() and str()
print(str(book))          # "1984" by George Orwell
print(book)               # "1984" by George Orwell

# __repr__ is called by repr() and in interactive shell
print(repr(book))         # Book(title='1984', author='George Orwell', year=1949)

# If __str__ is not defined, __repr__ is used as fallback
```

- Python doesn't support traditional **constructor overloading**, but provides alternatives:

```py
class Rectangle:
    def __init__(self, width=None, height=None, square_side=None):
        """Flexible initialization"""
        if square_side is not None:
            self.width = self.height = square_side
        elif width is not None and height is not None:
            self.width = width
            self.height = height
        else:
            raise ValueError("Provide either width and height, or square_side")
    
    @classmethod
    def from_square(cls, side):
        """Alternative constructor for squares"""
        return cls(width=side, height=side)
    
    @classmethod
    def from_dimensions(cls, dimensions):
        """Create from tuple or list"""
        width, height = dimensions
        return cls(width=width, height=height)
    
    def area(self):
        return self.width * self.height

# Different ways to create rectangles
rect1 = Rectangle(10, 20)
rect2 = Rectangle(square_side=15)
rect3 = Rectangle.from_square(15)
rect4 = Rectangle.from_dimensions((10, 20))

print(rect1.area())  # 200
print(rect2.area())  # 225
```

- Use `@classmethod` for alternative constructors
- Provides clear, named entry points for different initialization scenarios

### Inheritance and Method Overriding

Inheritance allows a class (child) to acquire properties and methods of another class (parent).

```py
class Animal:
    """Base class"""
    
    def __init__(self, name, species):
        self.name = name
        self.species = species
    
    def make_sound(self):
        """Method to be overridden"""
        return "Some generic sound"
    
    def info(self):
        """Inherited method"""
        return f"{self.name} is a {self.species}"

class Dog(Animal):
    """Derived class"""
    
    def __init__(self, name, breed):
        # Call parent constructor
        super().__init__(name, species="Dog")
        self.breed = breed
    
    def make_sound(self):
        """Method overriding"""
        return "Woof!"
    
    def fetch(self):
        """New method specific to Dog"""
        return f"{self.name} is fetching the ball"

class Cat(Animal):
    """Another derived class"""
    
    def __init__(self, name, indoor=True):
        super().__init__(name, species="Cat")
        self.indoor = indoor
    
    def make_sound(self):
        """Method overriding"""
        return "Meow!"

# Using inheritance
dog = Dog("Buddy", "Golden Retriever")
cat = Cat("Whiskers", indoor=True)

print(dog.info())         # Buddy is a Dog (inherited method)
print(dog.make_sound())   # Woof! (overridden method)
print(dog.fetch())        # Buddy is fetching the ball (new method)

print(cat.info())         # Whiskers is a Cat
print(cat.make_sound())   # Meow!

# Type checking
print(isinstance(dog, Dog))     # True
print(isinstance(dog, Animal))  # True (Dog is subclass of Animal)
print(isinstance(dog, Cat))     # False
print(issubclass(Dog, Animal))  # True
```

#### Method overriding rules

```py
class Parent:
    def method(self):
        return "Parent method"
    
    def another_method(self):
        return "Parent another_method"

class Child(Parent):
    def method(self):
        """Complete override"""
        return "Child method"
    
    def another_method(self):
        """Extending parent behavior"""
        parent_result = super().another_method()
        return f"{parent_result} + Child extension"

child = Child()
print(child.method())          # Child method
print(child.another_method())  # Parent another_method + Child extension
```

#### `super()` function

`super()` provides access to methods in parent classes, respecting the Method Resolution Order (MRO).

```py
class Shape:
    def __init__(self, color):
        self.color = color
        print(f"Shape.__init__ called with color={color}")
    
    def area(self):
        raise NotImplementedError("Subclass must implement area()")

class Rectangle(Shape):
    def __init__(self, color, width, height):
        print(f"Rectangle.__init__ called")
        super().__init__(color)  # Call parent initializer
        self.width = width
        self.height = height
    
    def area(self):
        return self.width * self.height

class Square(Rectangle):
    def __init__(self, color, side):
        print(f"Square.__init__ called")
        super().__init__(color, side, side)  # Call Rectangle's initializer

# Tracing initialization chain
square = Square("red", 5)
# Output:
# Square.__init__ called
# Rectangle.__init__ called
# Shape.__init__ called with color=red

print(square.area())  # 25
```

- In **Python 3**, `super()` without arguments is equivalent to `super(CurrentClass, self)`
- Follows **MRO**, not just immediate parent
- Essential for **cooperative multiple inheritance**
- Can be called anywhere in a method, not just at the beginning
- Python does not support traditional *overloading*, but provides alternatives:

```py
class Calculator:
    def add(self, a, b=None, c=None):
        """Using default arguments"""
        if b is None:
            return a
        if c is None:
            return a + b
        return a + b + c
    
    def multiply(self, *args):
        """Using variable arguments"""
        result = 1
        for num in args:
            result *= num
        return result
    
    def process(self, data):
        """Using type checking (discouraged in Python)"""
        if isinstance(data, int):
            return data * 2
        elif isinstance(data, str):
            return data.upper()
        elif isinstance(data, list):
            return sum(data)
        else:
            raise TypeError(f"Unsupported type: {type(data)}")

calc = Calculator()
print(calc.add(5))           # 5
print(calc.add(5, 3))        # 8
print(calc.add(5, 3, 2))     # 10

print(calc.multiply(2, 3, 4))     # 24
print(calc.process(5))            # 10
print(calc.process("hello"))      # HELLO
print(calc.process([1, 2, 3]))    # 6
```

- **Pythonic Alternatives**:
    - Default arguments
    - Variable-length arguments (`*args`, `**kwargs`)
    - Single dispatch (from `functools.singledispatch` for function-based approach)
    - Duck typing (accept any object with required methods)

#### Multiple Inheritance

Multiple inheritance allows a class to inherit from more than one parent class.

```py
class Flyable:
    """Mixin for flying capability"""
    
    def fly(self):
        return f"{self.name} is flying"

class Swimmable:
    """Mixin for swimming capability"""
    
    def swim(self):
        return f"{self.name} is swimming"

class Animal:
    """Base animal class"""
    
    def __init__(self, name):
        self.name = name
    
    def eat(self):
        return f"{self.name} is eating"

class Duck(Animal, Flyable, Swimmable):
    """Duck inherits from three classes"""
    
    def __init__(self, name):
        super().__init__(name)
    
    def quack(self):
        return f"{self.name} says Quack!"

# Using multiple inheritance
duck = Duck("Donald")
print(duck.eat())     # Donald is eating (from Animal)
print(duck.fly())     # Donald is flying (from Flyable)
print(duck.swim())    # Donald is swimming (from Swimmable)
print(duck.quack())   # Donald says Quack! (from Duck)
```

#### Method Resolution Order (MRO)

MRO defines the order in which Python searches for methods in the inheritance hierarchy. Python uses **C3 Linearization** algorithm to compute MRO.

```py
class A:
    def process(self):
        return "A"

class B(A):
    def process(self):
        return "B"

class C(A):
    def process(self):
        return "C"

class D(B, C):
    """D inherits from both B and C"""
    pass

# Checking MRO
print(D.__mro__)
# (<class 'D'>, <class 'B'>, <class 'C'>, <class 'A'>, <class 'object'>)

print(D.mro())  # Same as above, but as a list
# [<class 'D'>, <class 'B'>, <class 'C'>, <class 'A'>, <class 'object'>]

# Method resolution follows MRO
d = D()
print(d.process())  # "B" (found in B first, not C)
```

- **MRO Visualization**:

```mermaid
graph TD
    object --> A
    A --> B
    A --> C
    B --> D
    C --> D
    
    style D fill:#f9f,stroke:#333
```

<!-- Incomplete -->

### Encapsulation

Encapsulation is the process of bundling data and methods together. Python uses naming conventions rather than true access modifiers (like `private` in Java/C++).

- `_single_underscore` → **protected** (convention only)

- `__double_underscore` → name mangling for **private** attributes

```py
class Account:
    def __init__(self, owner, balance):
        self.owner = owner           # Public attribute
        self._balance = balance      # Protected (convention: internal use)
        self.__pin = "1234"          # Private (name mangling applied)
    
    def get_balance(self):
        """Public method to access protected attribute"""
        return self._balance
    
    def __validate_pin(self, pin):
        """Private method (name mangling)"""
        return pin == self.__pin
    
    def withdraw(self, amount, pin):
        """Public method using private validation"""
        if self.__validate_pin(pin):
            if amount <= self._balance:
                self._balance -= amount
                return f"Withdrew ${amount}"
            return "Insufficient funds"
        return "Invalid PIN"

account = Account("Alice", 1000)

# Public access
print(account.owner)  # Alice

# Protected access (nothing prevents it, just convention)
print(account._balance)  # 1000 (not recommended, but possible)

# Private access (name mangling)
try:
    print(account.__pin)  # AttributeError
except AttributeError as e:
    print(f"Error: {e}")

# Private attributes are mangled to _ClassName__attribute
print(account._Account__pin)  # 1234 (not truly private, but obfuscated)

# Using public interface
print(account.withdraw(200, "1234"))  # Withdrew $200
print(account.get_balance())           # 800
```

- **Naming Conventions Summary**

```py
class Example:
    def __init__(self):
        self.public = "I'm public"
        self._protected = "I'm protected by convention"
        self.__private = "I'm name-mangled"
    
    def public_method(self):
        return "Anyone can call me"
    
    def _protected_method(self):
        return "Internal use suggested"
    
    def __private_method(self):
        return "Name mangled method"
    
    def access_private(self):
        """Public interface to private members"""
        return self.__private_method()

obj = Example()

# All are technically accessible
print(obj.public)                    # Works
print(obj._protected)                # Works (not recommended)
print(obj._Example__private)         # Works (name mangling exposed)

# Methods
print(obj.public_method())           # Works
print(obj._protected_method())       # Works (not recommended)
print(obj.access_private())          # Works (proper way)
```

- **Best Practices for Encapsulation**

```py
class BankAccount:
    """Demonstrating proper encapsulation"""
    
    def __init__(self, account_number, initial_balance):
        self.__account_number = account_number  # Private
        self.__balance = initial_balance        # Private
        self._transaction_history = []          # Protected
    
    @property
    def account_number(self):
        """Read-only access to account number"""
        return f"****{self.__account_number[-4:]}"  # Masked
    
    @property
    def balance(self):
        """Read-only access to balance"""
        return self.__balance
    
    def deposit(self, amount):
        """Public method with validation"""
        if amount > 0:
            self.__balance += amount
            self._transaction_history.append(f"Deposit: +${amount}")
            return True
        raise ValueError("Deposit amount must be positive")
    
    def withdraw(self, amount):
        """Public method with business logic"""
        if amount > 0 and amount <= self.__balance:
            self.__balance -= amount
            self._transaction_history.append(f"Withdrawal: -${amount}")
            return True
        raise ValueError("Invalid withdrawal amount")
    
    def _calculate_interest(self, rate):
        """Protected helper method"""
        return self.__balance * rate
    
    def __str__(self):
        return f"Account {self.account_number}: ${self.balance}"

# Usage
account = BankAccount("123456789", 1000)
print(account)                    # Account ****6789: $1000
account.deposit(500)
print(account.balance)            # 1500

# Cannot directly modify balance
# account.balance = 5000  # AttributeError (if using @property without setter)
```

#### Properties and Descriptors

- **`@property` Decorator**

Properties provide a Pythonic way to implement **getters**, **setters**, and **deleters** while maintaining attribute-like syntax.

```py
class Temperature:
    def __init__(self, celsius):
        self._celsius = celsius  # Private attribute
    
    @property
    def celsius(self):
        """Getter for celsius"""
        print("Getting celsius")
        return self._celsius
    
    @celsius.setter
    def celsius(self, value):
        """Setter with validation"""
        print(f"Setting celsius to {value}")
        if value < -273.15:
            raise ValueError("Temperature below absolute zero!")
        self._celsius = value
    
    @celsius.deleter
    def celsius(self):
        """Deleter"""
        print("Deleting celsius")
        del self._celsius
    
    @property
    def fahrenheit(self):
        """Computed property (read-only)"""
        return (self.celsius * 9/5) + 32
    
    @fahrenheit.setter
    def fahrenheit(self, value):
        """Setting fahrenheit updates celsius"""
        self.celsius = (value - 32) * 5/9

# Using properties
temp = Temperature(25)
print(temp.celsius)        # Getting celsius → 25
print(temp.fahrenheit)     # Getting celsius → 77.0

temp.celsius = 30          # Setting celsius to 30
print(temp.celsius)        # Getting celsius → 30

temp.fahrenheit = 86       # Setting celsius to 30.0 (via fahrenheit setter)
print(temp.celsius)        # Getting celsius → 30.0

# Validation works
try:
    temp.celsius = -300    # ValueError
except ValueError as e:
    print(e)

# Deletion
del temp.celsius           # Deleting celsius
```

- Property vs Direct Attribute Access

```py
# Without property (not recommended for complex logic)
class CircleBasic:
    def __init__(self, radius):
        self.radius = radius
        self.diameter = radius * 2  # Problem: can become inconsistent
    
    def area(self):
        return 3.14159 * self.radius ** 2

circle1 = CircleBasic(5)
print(circle1.diameter)  # 10
circle1.radius = 10      # Changing radius
print(circle1.diameter)  # Still 10 (inconsistent!)

# With property (recommended)
class CircleProperty:
    def __init__(self, radius):
        self._radius = radius
    
    @property
    def radius(self):
        return self._radius
    
    @radius.setter
    def radius(self, value):
        if value <= 0:
            raise ValueError("Radius must be positive")
        self._radius = value
    
    @property
    def diameter(self):
        """Computed property - always consistent"""
        return self._radius * 2
    
    @diameter.setter
    def diameter(self, value):
        self._radius = value / 2
    
    @property
    def area(self):
        """Read-only computed property"""
        return 3.14159 * self._radius ** 2

circle2 = CircleProperty(5)
print(circle2.diameter)  # 10
circle2.radius = 10      # Changing radius
print(circle2.diameter)  # 20 (consistent!)
print(circle2.area)      # 314.159

circle2.diameter = 30    # Setting via diameter
print(circle2.radius)    # 15.0
```

- **Custom Descriptors**

Descriptors are the mechanism behind properties. They allow you to customize attribute access across multiple classes.

```py
class ValidatedString:
    """Descriptor for validated string attributes"""
    
    def __init__(self, minlen=0, maxlen=100):
        self.minlen = minlen
        self.maxlen = maxlen
    
    def __set_name__(self, owner, name):
        """Called when descriptor is assigned to a class attribute"""
        self.name = f"_{name}"
    
    def __get__(self, instance, owner):
        """Called when attribute is accessed"""
        if instance is None:
            return self  # Accessing from class, not instance
        return getattr(instance, self.name, None)
    
    def __set__(self, instance, value):
        """Called when attribute is set"""
        if not isinstance(value, str):
            raise TypeError(f"{self.name} must be a string")
        if len(value) < self.minlen:
            raise ValueError(f"{self.name} must be at least {self.minlen} characters")
        if len(value) > self.maxlen:
            raise ValueError(f"{self.name} must be at most {self.maxlen} characters")
        setattr(instance, self.name, value)

class PositiveNumber:
    """Descriptor for positive numbers"""
    
    def __set_name__(self, owner, name):
        self.name = f"_{name}"
    
    def __get__(self, instance, owner):
        if instance is None:
            return self
        return getattr(instance, self.name, 0)
    
    def __set__(self, instance, value):
        if not isinstance(value, (int, float)):
            raise TypeError(f"{self.name} must be a number")
        if value <= 0:
            raise ValueError(f"{self.name} must be positive")
        setattr(instance, self.name, value)

class Product:
    """Using descriptors for validation"""
    name = ValidatedString(minlen=3, maxlen=50)
    price = PositiveNumber()
    
    def __init__(self, name, price):
        self.name = name    # Calls ValidatedString.__set__
        self.price = price  # Calls PositiveNumber.__set__

# Using the descriptor-based class
product = Product("Laptop", 999.99)
print(product.name)   # Laptop
print(product.price)  # 999.99

# Validation works automatically
try:
    product.name = "AB"  # Too short
except ValueError as e:
    print(e)

try:
    product.price = -10  # Negative
except ValueError as e:
    print(e)

try:
    product.price = "expensive"  # Wrong type
except TypeError as e:
    print(e)
```

- **When to use descriptors**:
    - Reusable validation logic across multiple classes
    - Complex attribute management (caching, type checking, etc.)
    - Framework/library development
    - Note: For simple cases, @property is usually sufficient

#### Benefits of Encapsulation

- **Data validation**: Control how attributes are modified
- **Internal representation freedom**: Change implementation without affecting external code
- **Prevents accidental modification**: Reduces bugs from unintended state changes
- **Clear interface**: Public methods define how objects should be used

<!-- ### Abstraction

Hiding complex implementation details and exposing only necessary interfaces.

 -->







[OOPS Notebook](./Notebooks/9_Object_Oriented_Programming.ipynb)

---
