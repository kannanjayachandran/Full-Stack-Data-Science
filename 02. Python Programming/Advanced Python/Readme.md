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
Refer to the [Functional Programming In python](./Notebooks/01_Functional_programming.ipynb) notebook for more examples.

## Decorators in Python

Decorators are a design pattern in Python that allows a user to add new functionality to an existing object without modifying its structure. 

```python
def decorator(func):
    def wrapper():
        print("Before function call")
        func()
        print("After function call")
    return wrapper

@decorator
def say_hello():
    print("Hello")

say_hello()
```

- A decorator is a design pattern in Python.

## HOF (Higher Order Functions)

A higher-order function is a function that takes a function as an argument, returns a function, or does both. 

```python
def decorator(func):
    def wrapper():
        print("Before function call")
        func()
        print("After function call")
    return wrapper

def say_hello():

print("Hello")

decorated = decorator(say_hello)

decorated()
```

- A decorator is actually a syntactic sugar for elegantly composing higher-order functions.

## Closures in Python

A closure is a function object that has access to variables in its enclosing scope even if the function is defined outside the scope. 

```python
def outer_function(message):
    def inner_function():
        print(message)
    return inner_function

my_func = outer_function("Hello")
my_func()
```

Refer to the [Decorators In python](./Notebooks/02_Python_Decorators.ipynb) notebook for more examples.

## Generators in Python

Generators are a simple way to create iterators using functions. A generator is a function that returns an iterator object (a generator object) which can be used to iterate over the elements produced by the generator.

```python
def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

for num in fibonacci(10):
    print(num)
```

- Generators are crucial in when we want to optimize memory usage and performance while working with large streams of data.

Refer to the [Generators In python](./Notebooks/03_Python_Generator.ipynb) notebook for more examples.

## Iterators in Python

An iterator is an object that implements the iterator protocol, which consists of the methods `__iter__()` and `__next__()`. On a higher level an Iterator is an object that can be iterated upon. An object which will return data, one element at a time { looped over }. 

An object is called iterable if we can get an iterator from it. Most built-in containers in Python like: list, tuple, string etc. are iterables. But they itself are not iterators. An iterator can sort of remember its state, hence knows where it is during iteration.

```python
class MyRange:
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __iter__(self):
        return self

    def __next__(self):
        if self.start >= self.end:
            raise StopIteration
        value = self.start
        self.start += 1
        return value

numbers = MyRange(1, 5)
for num in numbers:
    print(num)
```

Refer to the [Iterators In python](./Notebooks/04_Python_Iterators.ipynb) notebook for more examples.

## Context Managers in Python

A context manager is an object that defines the runtime context to be established when executing a `with` statement. The context manager handles the entry into, and the exit from, the desired runtime context for the execution of the block of code.

```python

class OpenFile:
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode

    def __enter__(self):
        self.file = open(self.filename, self.mode)
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

with OpenFile('sample.txt', 'w') as f:

    f.write('Hello, World!')
```

- Context managers are used to manage resources like files, network connections, database connections, etc.

Refer to the [Context Managers In python](./Notebooks/05_Context_Managers.ipynb) notebook for more examples.

## Regular Expressions in Python

Regular expressions are a powerful tool for pattern matching and searching in strings. They are used to search for patterns in text data. For people with less experience in regular expressions, they can be hard to understand, but they are very powerful and now a days we can use them easily with the help of large language models like chat GPT and so on.

[Python Regular Expressions Re module Documentation](https://docs.python.org/3/library/re.html#)

**Regular expression syntax**

- `.` : Matches any character except a newline.
- `*` : Matches zero or more occurrences of the previous character.
- `+` : Matches one or more occurrences of the previous character.
- `?` : Matches zero or one occurrence of the previous character.
- `\d` : Matches any digit (0-9).
- `\D` : Matches any non-digit character.
- `\w` : Matches any word character (alphanumeric and underscore).
- `\W` : Matches any non-word character.
- `\s` : Matches any whitespace character.
- `\S` : Matches any non-whitespace character.

**Anchors**
* `\b` - word boundary
* `\B` - not a word boundary
* `^` - beginning of a string
* `$` - end of a string

**Character Classes**
* `[]` - matches characters in brackets
* `[^ ]` - matches characters NOT in brackets
* `|` - either or
* `()` - group
* `[1-4]` - range of numbers (minimum, maximum)

```python
import re

text = "The rain in Spain"

# Search for a pattern
pattern = 'rain'
result = re.search(pattern, text)
print(result) # <re.Match object; span=(4, 8), match='rain'>

# Find all occurrences of a pattern
pattern = 'ain'
result = re.findall(pattern, text)
print(result) # ['ain', 'ain']

# Replace a pattern
pattern = 'ain'
result = re.sub(pattern, 'an', text)
print(result) # The ran in Spain
```

Refer to the [Regular Expressions In python](./Notebooks/06_RegEx_in_Python.ipynb) notebook for more examples.

## Multi-Threading and Multi-Processing in Python

- **Program** : An executable file containing a series of instructions that the computer can execute in sequence and is usually stored on the disk.

- **Process** : A process is what we call a program that has been loaded into the memory along with all the resources it needs to operate. A process has its own memory space.

- **Thread** : A thread is the smallest unit of execution within a process. A process can have multiple threads. Threads share the same memory space.

**Multi-threading** is a technique in which multiple threads are created within a process to execute multiple tasks concurrently (well sort of). It gives the illusion of threads running in parallel, but in reality, they are running in a concurrent manner on a single core; that is the CPU switches between threads to give the appearance of parallelism.

The **Global Interpreter Lock** (GIL) in Python is a _mutex_ that protects access to Python objects, preventing multiple native threads from executing Python bytecode at once. This means that in a standard `CPython` interpreter, only one thread can execute Python code at a time, even if multiple threads are active. The GIL is necessary because CPython's memory management is not thread-safe. What that means is; 

- The GIL ensures that only one thread runs in the interpreter at any given moment.

- Multiple threads can exist and context switching is also possible; but only one thread can execute Python code at a time.

- However in the case of I/O bound tasks and some C extensions, the GIL is released and multiple threads can run in parallel.

Hence the GIL is a performance bottleneck in CPU-bound tasks with threading, but not in I/O-bound tasks.

**Multi-processing** is a technique in which multiple processes are created across multiple CPU cores to execute multiple tasks concurrently. Each process has its own memory space, so they do not share memory.

During multi-processing in Python, each process has its own Python interpreter and memory space, so the GIL is not a problem. This means that each process can execute Python code concurrently on different CPU cores and we can achieve true parallelism. This is particularly useful for CPU-bound tasks that require significant processing power. There are some complications that arises with multi-processing, like the overhead of creating and managing processes, extra memory usage, and the need to share data between processes and synchronize their execution. Python does a pretty good job in smoothening them over most of the time.

`Threading` and `Asyncio` are two Python libraries that helps us achieve concurrency. The way the threads or tasks take turns is the big difference between the two. 

In `Threading`, the OS actually knows about each thread and can interrupt it at any time to start running a different thread. This is called **pre-emptive multitasking** since the operating system can `pre-empt` your thread to make the switch. This is quite handy, as the code in the thread doesn't need to do anything special to make the switch happen. But it also means that the OS can interrupt your thread at any time, even in the middle of a critical operation.

`Asyncio` on the other hand uses __Cooperative multitasking__ (Non OS initiated context switching). Therefore the code in the task has to change slightly to make the switching happen. `Asyncio` uses an event loop. It is aware of each tasks and their respective states. The event loop selects one of the ready tasks and runs it. The task would be in complete control until it cooperatively hands the control back to the event loop. Then the event loop would place that task in either the ready or waiting state (or list) and goes through each of the tasks in the waiting list to see if they are ready to run (or continue). The tasks in the ready list are already ready because they haven't run yet.


Refer to [Multi-Threading In python](./Notebooks/09_Multithreading_in_Python.ipynb), [Multi-Processing In python](./Notebooks/10_MultiProcessing_in_Python.ipynb) notebooks for more examples.

Also more examples can be found [here](./Scripts/Multithreading_Multiprocessing/)

## Networking in Python

Python provides several modules to work with network programming. The `socket` module is the most important module for network programming in Python. It provides a low-level interface to create client-server applications. 

Typically python is not a first choice for networking, due to its interpreted nature. Also GIL (Global Interpreter Lock) prevents python from using multiple cores. But nonetheless, python can handle networking tasks.

```python
import socket

# create a socket object
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# get local machine name
host = socket.gethostname()

port = 12345

# bind to the port
server.bind((host, port))

# queue up to 5 requests
server.listen(5)

while True:
    # establish a connection
    client, addr = server.accept()
    print(f"Got a connection from {addr}")
    client.send("Thank you for connecting".encode())
    client.close()
```

Refer to the [Networking In python](./Notebooks/08_Python_Networking.ipynb) notebook for more examples.

## Advanced OOPS in Python

```python
# Abstract Base Class
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return 3.14 * self.radius * self.radius

circle = Circle(5)
print(circle.area()) # 78.5
```

Refer to the [Advanced OOPS In python](./Notebooks/09_Advanced_OOP.ipynb) notebook for more examples.

## Interpreter - CPython - GIL

CPython is the reference implementation of the Python programming language. It is written in C and is the most widely used Python interpreter. CPython uses the Global Interpreter Lock (GIL) to ensure that only one thread executes Python bytecode at a time. This can limit the performance of multi-threaded Python programs.

Refer to the [Interpreter - CPython - GIL](./Notebooks/10_Interpreter_CPYTHON_GIL.ipynb) notebook for more examples.

## Testing in Python

Testing is an important part of software development. Python provides several modules to write and run tests. The `unittest` module is the most popular module for writing tests in Python. It provides a way to write test cases, test suites, and test fixtures.

```python
import unittest

def add(a, b):
    return a + b

class TestAdd(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(2, 3), 5)

if __name__ == '__main__':
    unittest.main()
```

Refer to the [Testing](./Scripts/Readme.md) notebook for more examples.
