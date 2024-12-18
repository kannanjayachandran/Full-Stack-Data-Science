<h1 align="center" > Python Programming </h1>

**[Python](https://en.wikipedia.org/wiki/Python_(programming_language))** is an **[open-source](https://en.wikipedia.org/wiki/Open_source)**, **[high-level](https://en.wikipedia.org/wiki/High-level_programming_language)** and **[general-purpose](https://en.wikipedia.org/wiki/General-purpose_programming_language)** programming language. It is `dynamically type-checked` (type safety of a program is verified at runtime) and `garbage-collected`. 

> Note: These notes use Python 3 for all examples and explanations.

## Table of Contents

- [Python Syntax](#Python-Syntax)
- [Variables in python](#variables-in-python)	


## Python Syntax

After installing Python, you can start coding by writing code in a text editor and saving it with a .py extension. The saved file can be executed using the Python interpreter. Python emphasizes code readability and relies on indentation to define code blocks instead of using braces `{}` like many other programming languages.

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

## Variables in Python

Variables are containers for storing data values. Python is dynamically typed, meaning you don’t need to declare a variable’s type; it is determined automatically based on the value assigned.

```python
x = 5
y = "Hello, World!"
```

In Python, variables serve as references (or bindings) to [objects](https://en.wikipedia.org/wiki/Object-oriented_programming). For example:

```python
n = 300
```

Here, an integer object with the value `300` is created, and the variable `n` is a reference to this object (means `n` is storing the memory address). Every object in Python has a unique identifier (typically its memory address), which can be checked using the id() function.

> In Python, everything is an object; this includes variables, functions, classes, and even modules. Objects are instances of classes, and each class has its own attributes and methods.

Consider the following code:

```python
n = 300
n = "foo"
```

We can visualize the memory allocation as follows

<p align="center"><img src="./img/Pointer_animation.gif" alt="Logo"></p>

Once we reassign the variable `n` to a new value, the reference is updated to point to the new object.

### Memory Optimization

Python is a high-level language and it manages memory automatically. Python optimizes memory usage by binding multiple variables with the same value to a single object instead of creating duplicate objects. For example:

```python
a = 10
b = 10
```

In this case, `a` and `b` both refer to the same object in memory. If you reassign one of these variables, say `a = "foo"`, Python creates a new string object `"foo"` and binds `a` to it. The integer object `10` remains bound to `b`.

If no variables are referencing an object, it becomes orphaned and eligible for garbage collection. An object is eligible for garbage collection if the number of references to it becomes zero. We can manually reduce the reference count of an object using the `del` keyword.

### Pass by What?

Programming languages typically share objects or data between functions using one of two approaches:

- **Pass by value**: A copy of the actual argument is passed to the function. Modifications inside the function do not affect the original object.

- **Pass by reference**: The actual argument (its memory address) is passed, allowing modifications inside the function to directly affect the original object.

Python, however, uses a different mechanism known as **Pass by Object Reference** (or **Pass by Assignment**). In Python, everything is an object, including primitive types (e.g., integers, floats, and strings) and complex data structures (e.g., lists, dictionaries). When passing arguments to functions, Python passes a reference to the object (essentially a pointer to its memory address), not the actual object.

The behavior depends on whether the object is **mutable** or **immutable**:

- **Immutable Objects** (e.g., integers, strings, tuples):
When an immutable object is passed to a function and modified, Python creates a new object rather than altering the existing one.

```python
def modify_value(x):
    x += 10
    print("Inside function:", x)

a = 5
modify_value(a)
# Output: Inside function: 15
print("Outside function:", a)
# Output: Outside function: 5
```

- **Mutable Objects **(e.g., lists, dictionaries, sets):
When a mutable object is passed to a function and modified, the changes affect the original object.

```python
def modify_list(lst):
    lst.append(10)
    print("Inside function:", lst)

my_list = [1, 2, 3]
modify_list(my_list)
# Output: Inside function: [1, 2, 3, 10]
print("Outside function:", my_list)
# Output: Outside function: [1, 2, 3, 10]
```

### Boxing and Unboxing

Since everything in Python is an object, operations on variables involve processes called **boxing** and **unboxing**.

- **Boxing**: Wrapping primitive values (e.g., integers, floats) into Python objects so they can be managed as objects.

- **Unboxing**: Extracting the actual value from an object to perform operations.

For example, consider the addition of two integers:

```python
a = 10
b = 20
c = a + b
```

Behind the scenes, Python performs the following steps:

- Check the types of both operands (`a` and `b`).

- Verify support for the `+` operation for these types.

- Extract the function responsible for performing the `+` operation.

- Unbox the values from the objects (`10` and `20`).

- Perform the addition (`10 + 20`).

- Box the result into a new integer object (`30`).

These steps ensure Python’s flexibility but can introduce overhead in performance.

### A peak into CPython Optimization

For performance-critical tasks, tools like **Cython** can optimize Python code by reducing the overhead of boxing and unboxing. Here’s an example of a Cython function for adding two integers:

```python
cpdef int add(int x, int y):
    cdef int result
    result = x + y
    return result
```

By specifying types (`int`), this code bypasses Python's usual boxing and unboxing processes, significantly improving performance.

### Automatic Memory Management

Python handles memory allocation and deallocation automatically through its memory manager, which manages the **Python heap**. The memory manager:

1. Allocates memory for objects when needed.

1. Tracks references to objects.

1. Deallocates memory when objects are no longer referenced (via garbage collection).

1. This ensures efficient memory usage without requiring explicit memory management from the programmer.

## Data Types

Python has the following data types built-in by default, in these categories:

- **Text Type**: `str`
- **Numeric Types**: `int`, `float`, `complex`
- **Sequence Types**: `list`, `tuple`, `range`
- **Mapping Type**: `dict`
- **Set Types**: `set`, `frozenset`
- **Boolean Type**: `bool`
- **Binary Types**: `bytes`, `bytearray`, `memoryview`

- For purposes of optimization, the interpreter creates objects for the integers in the range $\color{FEC260}[-5, \;256]$ at startup, and then reuses them during program execution. Thus, when you assign separate variables to an integer value in this range, they will actually reference the same object created earlier. (The numbers from -5 to 256 are found to be used the most). 

We generally won't face integer overflow in python as it can handle arbitrarily large numbers. Python internally uses a data structure called `long` to store large numbers.

## Operators

Python language supports the following types of operators.

- **Arithmetic Operators**

- **Comparison (Relational) Operators**: `==`, `!=`, `>`, `<`, `>=`, `<=`

- **Assignment Operators**: `=`, `+=`, `-=`, `*=`, `/=`, `%=`, `//=`, `**=`

- **Logical Operators**: `and`, `or`, `not`

- **Bitwise Operators**: `&`, `|`, `^`, `~`, `<<`, `>>`

- **Membership Operators**: `in`, `not in`

- **Identity Operators**: `is`, `is not`

- **Ternary Operator**: `a if condition else b`
> Not really ternary operator, but a small hack to achieve the same.

## Python Keywords

![Python keywords](./img/Keywords.png)

## Conditional Statements

Python supports the usual logical conditions:

- **if** statement

- **elif** statement

- **else** statement

- **Nested if** statement

## Loops

Python has two primitive loop commands:

- **while** loops

- **for** loops

We can use the `break` statement to stop the loop before it has looped through all the items, and the `continue` statement to stop the current iteration of the loop, and continue with the next.

Python loops also have something like $\color{FEC260}for→else$ and $\color{FEC260}while→else$ which is executed when the loop is finished without a `break` statement.

## Functions

Like other programming languages, Python also supports functions. A function is a block of code that only runs when it is called. We use the `def` keyword to define functions in Python.

- In python all the function calls resides inside the stack memory. For storing all the objects, we have the heap memory. Heap is larger in size than the stack. The actual data is stored inside heap while stack stores the references. 

- Python functions can return multiple values. It is done by returning a tuple.

- A `parameter` is a variable in a method definition. When a method is called, the `arguments` are the data you pass into the method's parameters.

**[Functions Notebook](./Notebooks/8_Functions.ipynb)**

### Lambda Functions

Lambda functions are small anonymous functions. They can have any number of arguments but only one expression. The expression is evaluated and returned. Lambda functions can be used wherever function objects are required.

## String

Strings in Python are immutable sequences of Unicode code-points/ arrays or bytes representing unicode characters. Strings are ordered, indexed, and can contain duplicate elements. They are also iterable. We can slice and use the membership operator in strings.

**[String Notebook](./Notebooks/7_String.ipynb)**

## Built-in Containers

### List

Python lists are heterogeneous, mutable built-in data structures. Lists are indexed, dynamic and allows duplicate elements. Most of the operations on a list takes $\color{FEC260}O(1)$ or $\color{FEC260}O(n)$ time complexity. Lists has a powerful mechanism to create new lists from other iterable (list, tuple, strings, etc.) called List comprehension. Lists are ordered also. Lists are superset of arrays.

- During list slicing, we are creating a new list object. We can use the slicing technique in other python data structures like tuple, string, etc.

- List comprehension is an elegant way to generate new lists form existing ones. Internally python would run a loop to do this, hence the complexity of comprehension would be similar to the iterative implementation. 

**[List Notebook](./Notebooks/3_List.ipynb)**

### Tuple

Tuples are basically immutable or read-only lists. The tuple stores items as reference, hence the tuple is immutable not the items; i.e. we can change the contents of a list inside a tuple but we cannot change the item in the tuple. 

- Tuples are generally faster than lists because they are immutable. 

**[Tuple Notebook](./Notebooks/4_Tuple.ipynb)**

### Set

Set is an unordered data structure in python, where every item is unique and immutable, however the set itself is mutable. Set allows fast insertion, deletion and searching. 

- We can insert, delete, and search a list in amortized  $\color{gold}O(1)$ time complexity. 

- Set uses hashing under the hood and that’s how it achieves that $\color{gold}O(1)$ time complexity. 

- Set also supports different set (Mathematical set) operations like Union, Intersection, Difference, Symmetric difference, etc. 

- Sets are also highly optimized for membership tests, it can do membership test in $\color{gold}O(1)$ (amortized).

**[Set Notebook](./Notebooks/6_Set.ipynb)**

### Dictionary

A dictionary is a collection of key-value pairs. They are generally unordered, indexed, and mutable. The key is used to access the value and the key must be unique (immutable objects). The value can be any data type. Under the hood dictionaries also use hashing like sets.

- Dictionaries are highly optimized for searching, insertion, and deletion.

- Dictionaries are also highly optimized for membership tests, it can do membership test in $\color{gold}O(1)$ (amortized).

**Shallow Copy and Deep Copy**

Shallow copy creates a new dictionary and copy references to the objects found in the original dictionary. Hence changes to mutable objects within the copy affect the original dictionary.

Deep copy creates a new dictionary and recursively copies the objects found in the original dictionary. Hence changes to mutable objects within the copy do not affect the original dictionary.

- Dictionaries also support comprehension like lists.

**[Dictionary Notebook](./Notebooks/5_Dictionary.ipynb)**

## Object Oriented Programming in Python

OOPS properties in Python are similar to other major object oriented languages. What python does differently are:

### Multiple Inheritance

- A class can be derived from more than one base classes in Python. Useful while having [Mixins](https://en.wikipedia.org/wiki/Mixin).

- the C3 linearization (MRO - Method Resolution Order) to resolve method calls in a consistent and predictable manner, which avoids the diamond problem by defining a specific order in which base classes are searched.

### Abstract Base Classes and Duck Typing

- Abstract Base Classes are classes that contain one or more abstract methods. An abstract method is a method that is declared, but contains no implementation. Python provides the `abc` module to use the Abstract Base Classes.

- Duck typing is a concept related to dynamic typing, where the type or class of an object is less important than the methods it defines. When you use duck typing, you do not check types at all. Instead, you check for the presence of a given method or attribute. 

> If it looks like a duck and quacks like a duck, it must be a duck. 🦆

### Method Overloading

- Python does not support method overloading like C++ or Java. We can use default arguments or variable length arguments to achieve the same.

- Due to the dynamic nature of Python, it can already handle different types of arguments.

### Encapsulation and Access Modifiers

Encapsulation in OOP refers to two concepts:

- A language construct that bind data with methods that operate on that data.

- A language mechanism that restricts direct access to some of an object's components.

Python simply uses naming conventions to indicate private(__) and protected(_) members of a class.

### Constructor Overloading

- Python does not support constructor overloading the same way as C++ or Java.  The `__init__` method is used for initialization, and multiple ways of initializing an object can be handled with default arguments or conditional logic within `__init__`.

### Dunder Methods or Magic Methods

- `Dunder` methods or magic methods are special methods that have double underscores at the beginning and end of their names. They are used to create functionality that can't be represented as a normal method.

[OOPS Notebook](./Notebooks/9_Object_Oriented_Programming.ipynb)

## Python Documentation

[Official Python documentation](https://docs.python.org/3/)
