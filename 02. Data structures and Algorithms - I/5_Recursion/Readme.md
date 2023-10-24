<h1 align="center" style="color: orange"> Recursion </h1>

> ##  A function calling itself is called recursion.

Recursion is a core topic by itself and for various other problem solving strategies. Fundamentally, it is way to solve complex problems by dividing it into small compossible sub-problems and joining the solutions of the sub-problems to get the solution of the original problem. In order to understand recursion, first we should know about the following terms:

### Stack data structure and how function calls are stored in stack

_Stack_ is a linear data structure which follows FILO (First In Last Out) or LIFO (Last In First Out) order. It has two main operations: **push** and **pop**. Push operation adds an element to the top of the stack and pop operation removes an element from the top of the stack.

![Alt text](image.png)

Each time a function is invoked, a new stack frame is created and pushed to the top of the stack. When the function returns, the stack frame is popped out of the stack. The stack frame contains the return address of the function, the parameters passed to the function and the local variables declared in the function.

### Divide and Conquer

It is a problem solving techniques which involves breaking down a complex problem into smaller sub-problems, solving the sub-problems and combining the solutions of the sub-problems to get the solution of the original problem.

![Alt text](image-1.png)

```python
def fib(n: int) -> int:
    # base case
    if n <= 1:
        return n
    else: 
        # recursive call
        return fib(n-1) + fib(n-2)
```

### Base case and Recursive case

We know recursion is in essence a function calling itself, but we do need to call the function a fixed or limited number of times. So a **base case** is a condition which stops the recursion. We can say that it is the bare minimum computation that needs to be done; ie. the sub-problem which cannot be broken down any further. Think of it like the condition we write inside the loop to stop the loop.

A **recursive case** is a condition which calls the function recursively. In the recursive call we need to make sure that we are moving towards the base case. We can do this by modifying the parameters passed to the function in the recursive call. While approaching a problem, the ability to modify the parameter to recursively call the function is a key indicator of a problem that can be solved using recursion. 

> One thing to notice is that we are essentially trying to avoid a loop by replacing it with recursion. So, if a problem can be solved using a loop, it could also be solved using recursion, not that it should be solved using recursion always. Recursion is a tool in the tool box of a programmer, and it should be used when it is the right tool for the job.

There are different types of recursion , patterns of recursion and problems that can be solved using recursion, which could be learn from any standard algorithm book or website. I want to lay down the fundamental concepts of recursion.

## Questions

1. Factorial of a number
2. Fibonacci series
3. Tower of Hanoi
