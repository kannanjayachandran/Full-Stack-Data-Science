<h1 align="center"> Mathematics For Data Structures and Algorithms </h1>

### $\color{gold}\large\texttt{Variables}$

***In mathematics, variables are letters (like `a`, `b`, `x` …) representing numerical quantities. They allows us to formulate general rules that work for any values. It also help us to write equations and inequalities.***

#### $\color{gold}\large\texttt{Dependent Variables and Independent Variables}$

***Dependent variables are the variable that depends on other variables. While Independent variables that does not dependent on other variables.***

> ***Let’s say that we are analyzing the temperature difference of a particular region over a time. Now `time` here is an independent variable, while the `temperature` is a dependent variable as it depends on the variable time.***

### $\color{gold}\large\texttt{Equations and Inequalities}$

***We can use variables to construct expressions like equations and inequalities. Equations can be used to find the value of an unknown and to express relation between entities. Equation express relation through equality. We can have univariate equations (Equation with only one variable) or Multivariate equations (Equation with multiple variables).***

***Some basic transformations we can do on the equations to solve them are;***

- ***We can multiply/divide both sides with the same value.***
- ***We can move a quantity from one side to another by changing the sign(+ or -).***

> ***While dividing something, we need to be careful not to divide something by 0, as division by 0 is not defined. While we divide an equation’s both side with any $\color{orange}x$ we are automatically considering that $\color{orange}x$ is $\color{orange}\ne 0$.***

***Inequalities refer to relation between numerical values that are different i.e., to express numerical quantities relative to each other.***

#### $\color{gold}\large\texttt{Mathematical Proofs}$

***A mathematical proof is an argument given logically to validate a mathematical statement. Examples are not mathematical proofs. To get more intuition and practical knowledge of mathematics, examples are preferred here over proofs.***

### $\color{gold}\large\texttt{Functions}$

***Both in Computer Science and in Mathematics, functions refers to a set of rules that is used to transform inputs into outputs. They are similar to equations as both express a relation between variables. A function can have multiple inputs but only one output per inputs. Or it is equivalent to say that multiple inputs can give same out output, but it is not possible for a single input to give multiple output. There are different types of functions, like;*** 

**$\color{ffd700}\text{Constant Function (One variable)}$: *The function involving a single variable only, other being equal to 0.***

$$\large\color{orange}f(x) = c$$

***$\color{ffd700}\text{Two variables}$ : An equation containing two variables can be considered as a function with one input and one output.***

$$\large\color{orange}f(x) = x^{2}$$

***This is an equation, but also a function definition. There are two variables: $\color{orange}x$ and $\color{orange}f(x)$. Here $\color{orange}x$ is the input,  $\color{orange}f$ is the name of the function and $\color{orange}x^{2}$ is the function itself.***

***Now functions can have more than two variables also, for instance $\color{orange}y = x_{0}+x_{1}+c$   is an equation with 3 variables, two independent $\color{orange} x_{0}$ , $\color{orange} x_{1}$ and a dependent variable $\color{orange} y$.  We can write the same equation in different ways and thus get different functions corresponding to the equations.***

### $\color{gold}\large\texttt{Factors and Multiples}$

- ***$\color{orange}\text{Factors}$ are numbers that divide the given number without leaving a remainder.***

- ***$\color{orange}\text{Multiples}$ are the numbers obtained as a product of a given number and integers.***

- ***Factor of a number is less than or equal  to itself. Multiples are always greater than or equal to the given number.***

### $\color{gold}\large\texttt{Greatest Common Divisor (GCD | HCF)}$

***Greatest common divisor of two numbers is the highest common factor of those two numbers. GCD of any two numbers should be less than or equal to the smallest number among them.*** 

$\large\color{orange}GCD(5, 15)\;\;\;\;\;\;\;$ $\large\color{orange}5 \rarr1, 5\;\;\;\;\;\;\;$ $\large\color{orange}15\rarr1,3,5,15\;\;\;\;\;\;\;$ $\large\color{orange}GCD(5, 15) = 5$

```python
def gcd_normal(a, b):
    small = min(a, b)
    for i in range(1, small+1):
        if a%i == 0 and b%i == 0:
            gcd = i
    return gcd
```

***The above code is a normal way to find the GCD of two numbers. The time complexity of this code is $\color{orange}O(n)$, where $\color{orange}n$ is the smallest number among the two.***

***We can use the $\color{orange}Euclidean$ algorithm to find the GCD of two numbers.***

```python
def gcd_euclidean(a, b):
    if b == 0:
        return a
    return gcd_euclidean(b, a%b)
```

```python
def gcd_euclidean(a, b):
    while b:
        a, b = b, a%b
    return a
```

***The time complexity of the above code is $\color{orange}O(log(min(a, b)))$.***


### $\color{gold}\large\texttt{Least Common Multiple (LCM)}$

LCM or least common multiple of two numbers $\color{orange}x$ and $\color{orange}y$  is the lowest number $\color{orange}p$ so that both the numbers could divide $\color{orange}p$ completely. Range of lies between $\color{orange} max(x, y)$ and $\color{orange}x*y$.

$\large\color{orange}LCM(3, 4)\;\;\;\;\;\;\;$ $\large\color{orange}3 \rarr3, 6, 9, 12, 15, ....\;\;\;\;\;\;\;$ $\large\color{orange} 4\rarr4, 8, 12, 16, ...\;\;\;\;\;\;\;$ $\large\color{orange}LCM(3, 4) = 12$

```python
def lcm_normal(a, b):
    big = max(a, b)
    for i in range(big, a*b+1):
        if i%a == 0 and i%b == 0:
            lcm = i
            break
    return lcm
```

***The above code is a normal way to find the LCM of two numbers. The time complexity of this code is $\color{orange}O(n)$, where $\color{orange}n$ is the product of the two numbers.***

```python
def lcm(a, b):
    return (a*b)//gcd_euclidean(a, b)
```

***The time complexity of the above code is $\color{orange}O(log(min(a, b)))$.***

### $\color{gold}\large\texttt{Logarithm}$

Logarithm is simply a function (Inverse function of exponentiation) that gives the number, let’s say $\color{orange}x$ to be raised on a base $\color{orange}b$ in order to obtain a fixed number $\color{orange}a$.

$\large \color{orange}b^{x} = a \;\;\;\;\;\;\; \text{Can be written as}\;\;\;\;\;\;\; x = log_{b}a$

***If the base is not specified, we assume it to be $\color{orange}e$ , the natural log, it is also written as $\color{orange}ln$. Power and log are inverse of each other.***

#### $\color{gold}\large\texttt{Some properties of Logarithm}$

$\large\color{orange}log_b(b^x) = x \;\;\;\;\;\;\;  b^x = c^{x / log_b(c)}$

$\large\color{orange}log_b(y^a) = a * log_b(y)  \;\;\;\;\;\; \log_a(x) = log_a(b)* log_b(x)$

$\large\color{orange}log_b(x) = \frac{log_a(x)}{log_a(b)} \;\;\;\;\;\; log_b(x) = \frac{1}{log_x(b)}$

### $\color{gold}\large\texttt{Prime Numbers}$

***Prime numbers are the numbers that are divisible by 1 and itself only. The first prime number is 2.***

$$\large\color{orange}2, \;3, \;5, \;7,\; 11,\; 13,\; 17, \;...$$

```python   
# Brute force way - Complexity O(n)
def isPrime(a: int) -> bool:
    if a <= 1: 
        return False
    for i in range(2, a):
        if a % i == 0:
            return False
    return True
```

```python
# checking till the square root of a number Complexity O(SQRT(n))
def isPrime(a: int) -> bool:
    if a <= 1: 
        return False
    i = 2
    while i*i <= a:
        if a % i == 0:
            return False
        i+=1
    return True
```

```python
# using the 6n+1 property
def isPrime1(a: int) -> bool:
    if a <= 1: 
        return False
    if a == 2 or a == 3:
        return True
    if a % 2 == 0 or a % 3 == 0:
        return False
    
    i = 5
    while i*i <= a:
        if a % i == 0 or a % i+2 == 0:
            return False
        i += 6
    
    return True
```

***The time complexity of the above code is $\color{orange}O(\sqrt{n})$.***

#### $\color{gold}\large\texttt{Sieve of Eratosthenes}$

***Sieve of Eratosthenes is an algorithm used to find all prime numbers up to a given limit.***

```python
def sieve(num):

    prime = [True for i in range(num+1)]
    prime[0], prime[1] = False, False
    
    for i in range(2, int(num**0.5) + 1):
        if prime[i]:
            for j in range(i*i, num+1, i):
                prime[j] = False
                
    return [x for x in range(num+1) if prime[x]]
    
print(sieve(100))
```

***The time complexity of the above code is $\color{orange}O(nlog(log(n)))$.***

#### $\color{gold}\large\texttt{Extended Sieve Method}$

```python
def sieve(num):

    prime = [True for i in range(num+1)]
    prime[0], prime[1] = False, False
    
    for i in range(2, int(num**0.5) + 1):
        if prime[i]:
            for j in range(i*i, num+1, i):
                prime[j] = False
                
    return prime

def extended_sieve(num):
    
        prime = sieve(num)
        prime[0], prime[1] = False, False
        for i in range(2, num+1):
            if prime[i]:
                for j in range(i*i, num+1, i):
                    prime[j] = i
                    
        return prime

print(extended_sieve(100))
```

***The time complexity of the above code is $\color{orange}O(nlog(log(n)))$.***

### $\color{gold}\large\texttt{Arithmetic Progression}$

$$\color{orange} 1,\;5,\;9,\;13,\;......\;,\;n$$

***Arithmetic progression is a sequence of numbers in which the difference of any two successive members is a constant (`Common difference`)***

- The $\color{orange}n^{th}$ term of the AP is given by the equation $\color{orange}a_n = a_1 + (n-1)d$, where $\color{orange}a_1$ is the first term and $\color{orange}d$ is the common difference.

- The sum of the first $\color{orange}n$ terms of an AP is given by the equation $\color{orange}S_n = \frac{n}{2}(2a_1 + (n-1)d)$

- Sum of first $\color{orange}n$ natural numbers is given by the equation $\color{orange}S_n = \frac{n(n+1)}{2}$

- Sum of first $\color{orange}n$ odd numbers is given by the equation $\color{orange}S_n = n^2$

- Sum of first $\color{orange}n$ even numbers is given by the equation $\color{orange}S_n = n(n+1)$

### $\color{gold}\large\texttt{Geometric Progression}$

$$\color{orange} 1,\;2,\;4,\;8,\;......\;,\;n$$

***Geometric progression is a sequence of numbers in which the ratio of any two successive members is a constant (`Common ratio`)***

- The $\color{orange}n^{th}$ term of the GP is given by the equation $\color{orange}a_n = a_1 * r^{(n-1)}$, where $\color{orange}a_1$ is the first term and $\color{orange}r$ is the common ratio.

- The sum of the first $\color{orange}n$ terms of a GP is given by the equation $\color{orange}S_n = \frac{a_1(1-r^n)}{1-r}$

- Sum of infinite terms of a GP is given by the equation $\color{orange}S = \frac{a_1}{1-r}$

### $\color{gold}\large\texttt{Harmonic Progression}$

$$\color{orange} \frac{1}{1},\;\frac{1}{2},\;\frac{1}{3},\;\frac{1}{4},\;......\;,\;\frac{1}{n}$$

***Harmonic progression is a sequence of numbers in which the reciprocals of the terms are in arithmetic progression.***

- The $\color{orange}n^{th}$ term of the HP is given by the equation $\color{orange}a_n = \frac{1}{a_1 + (n-1)d}$, where $\color{orange}a_1$ is the first term and $\color{orange}d$ is the common difference.

- The sum of the first $\color{orange}n$ terms of a HP is given by the equation $\color{orange}S_n = \frac{n}{\frac{1}{a_1} + \frac{1}{a_n}}$

### $\color{gold}\large\texttt{Permutations and Combinations}$

***In general if we have $\color{orange}n$ items; then we can arrange them in $\color{orange}n!$ ways. Permutations and Combinations are the ways to select items from a group.***

- ***Permutations are the ways to select items from a group in which the order of selection matters.*** 

- ***Combinations are the ways to select items from a group in which the order of selection does not matter.***

- If we have a list $\color{orange}lis=[\:start, \;end\:]$ of consecutive elements; then we have $\color{orange}end-start+1$ items in the list. If one of `start` or `end` is inclusive and other is exclusive, then we have $\color{orange}end-start$ items in the list. If both are inclusive, then we have $\color{orange}end-start-1$ items in the list.

#### $\color{gold}\large\texttt{Permutations}$

- ***Permutations of $\color{orange}n$ items taken $\color{orange}r$ at a time is given by the equation $\color{orange}P(n, r) = \frac{n!}{(n-r)!}$; where $\color{orange}n$ is the total number of items and $\color{orange}r$ is the number of items to be selected.***

- ***Permutations of $\color{orange}n$ items taken all at a time is given by the equation $\color{orange}P(n, n) = n!$; where $\color{orange}n$ is the total number of items.***

#### $\color{gold}\large\texttt{Combinations}$

- ***Combinations of $\color{orange}n$ items taken $\color{orange}r$ at a time is given by the equation $\color{orange}C(n, r) = \frac{n!}{r!(n-r)!}$; where $\color{orange}n$ is the total number of items and $\color{orange}r$ is the number of items to be selected.***

- ***Combinations of $\color{orange}n$ items taken all at a time is given by the equation $\color{orange}C(n, n) = 1$; where $\color{orange}n$ is the total number of items.***

### $\color{gold}\large\texttt{Square Root}$

***Square root of a number is a value that, when multiplied by itself, gives the original number.***

- ***Square root of a number $\color{orange}n$ is given by the equation $\color{orange}\sqrt{n} = n^{1/2}$.***

- ***We can geometrically interpret the square root of a number as the side of a square whose area is equal to the number.***

```python
def sqrt_newton(n: float) -> float:
    if n < 0:
        return -1
    x = n  # Initial guess
    e = 0.000001  # Tolerance level
    while True:
        next_x = 0.5 * (x + n / x)  # Newton-Raphson update
        if abs(x - next_x) < e:  # Check for convergence
            break
        x = next_x
    return x
```

***The time complexity of the above code is $\color{orange}O(log(log(n/e)))$.***

```python
def sqrt_binary_search(n: float) -> float:
    if n < 0:
        return -1
    if n == 0:
        return 0  # Base case

    low, high = (0, max(1, n)) 
    e = 0.000001  # Tolerance level

    while high - low > e:
        mid = (low + high) / 2  
        mid_squared = mid * mid

        if abs(mid_squared - n) < e:  # If mid^2 is close enough to n
            return mid
        elif mid_squared < n:
            low = mid  
        else:
            high = mid 

    return (low + high) / 2  
```

***The time complexity of the above code is $\color{orange}O(log(n/e))$.***

---
## Questions

1. Roman to Integer
2. Integer to Roman
3. Random Pick with weight
4. Palindrome Number
5. Reverse Integer
6. Power of a number -->