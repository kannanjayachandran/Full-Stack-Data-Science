<h1 align="center"> Recursion </h1>

Recursion is a programming technique where a function calls itself to solve a problem. Recursion is based on the principle of _self-similarity_, where a problem can be broken down into smaller instances or sub-problems of the same problem.

- Every iterative solution can be converted into a recursive solution and vice-versa.

- Many standard algorithms and algorithmic techniques are based on recursion.

- We employ recursion when we can spare some extra space for efficient computation and better looking code.

In order to better understand recursion, let's look at the following concepts:

**1. Stack data structure and how function calls are stored in stack**

A `stack` is a linear data structure with push and pop operations, following the _FILO_ (First In, Last Out) or _LIFO_ (Last In, First Out) order.

![Stack and stack operations](./img/Stack.png)

Each time a function is called, a new stack frame is created and pushed onto the stack. When the function returns, the stack frame is popped off the stack.

**2. Base case and Recursive case**

A base case is a condition that stops the recursion. It represents the simplest computation that can't be broken down further. 

A recursive case is a condition that calls the function recursively, moving towards the base case by modifying the function's parameters (not necessarily).

```python
def fib(n: int) -> int:
    # base case
    if n <= 1:
        return n
    else: 
        # recursive call
        return fib(n-1) + fib(n-2)
```

The key to understand recursion, is to think of the problem in terms of smaller sub-problems and how to build up the solution from these sub-problems back to the original problem.




**Divide and Conquer**

Divide and Conquer is a problem-solving technique that breaks down complex problems into smaller sub-problems. It involves solving these sub-problems and combining their solutions to obtain the solution to the original problem.

![Divide and conquer algorithm](./img/Divide_&_Conquer.png)

## Problem set

1. Fibonacci Series

2. Factorial of a number

3. Tower of Hanoi

4. Recursive Binary Search

5. Given a string remove the argument letter

6. Josephus Problem
