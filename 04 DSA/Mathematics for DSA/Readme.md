<h1 align="center"> Mathematics For Data Structures and Algorithms </h1>

Mathematics is a fundamental component in understanding data structures and designing efficient algorithms. It provides the tools to analyze time and space complexity, understand patterns, and establish the correctness of solutions.

## Variables

In mathematics, variables are symbols (like `a`, `b`, `x`, etc.) used to represent numerical values. They allow us to formulate general rules that hold for many values, and they are essential in constructing equations and inequalities.

- A **dependent variable** is one whose value depends on one or more other variables.

- An **independent variable** is one that does not depend on other variables.

> Example:
>
> If we are analyzing temperature variation over time in a specific region:
>
> Time is the independent variable.
> 
> Temperature is the dependent variable, since its value changes in response to time.

## Expressions

An expression is a combination of numbers, variables, and operators. It is simply a mathematics phrase that can be evaluated or simplified.

> Example:
>
> $2x + 5$
>
> $3 + 7$
>
> $y^2 - 4$

## Equations and Inequalities

Variables can be used to construct expressions such as equations and inequalities. An **equation** asserts equality between two expressions. It is used to find unknown values or to represent relationships.

> Example:
>
> $2x + 5 = 11$
>
> $y^2 - 4 = 0$
>
> $4 + 3 = 7$

- A **univariate** equation involves one variable.

- A **multivariate** equation involves two or more variables.

**Inequalities** represent relationships between non-equal numerical values. They help express how values compare to each other — whether one is greater than, less than, or not equal to another.

**[Solving Equations and Inequalities](./Addon_Solving_Equations_and_Inequalities.md)**

## Mathematical Proofs

A mathematical proof is a sequence of logical steps, starting from a set of accepted truths (axioms, definitions, previously proven theorems), and leading inexorably to the conclusion of the statement being proven. It must be:

- **Rigorous**: Every step must be justified by a logical rule or a previously established truth. 

- **Logical**: It must follow the rules of formal logic.

- **Convincing**: It should persuade any knowledgeable reader of the statement's truth.

> Examples and intuitive understanding are helpful for learning and application, especially in computer science, but do not substitute for formal proofs.

**[Common Types of Mathematical Proofs](./Addon_Mathematical_Proofs.md)**

Unlike empirical sciences, where theories are supported by experimental evidence, mathematics relies on deductive reasoning to demonstrate absolute certainty.

## Functions

A function is a rule or mapping that assigns each input exactly one output. They provide a formal way to describe relationships between quantities. More formally, a function $f$ from a set $A$ (called the domain) to a set $B$ (called the co-domain) is a relation that associates each element $x$ in $A$ to exactly one element $y$ in $B$.

- **Domain**: The set of all possible input values for which the function is defined.

- **Co-domain**: The set of all possible output values that the function could potentially produce.

- **Range**: The actual set of all output values that the function produces for its given domain. The range is a subset of the co-domain.

- **Notation**: $y = f(x)$, where $x$ is the independent variable (input) and $y$ is the dependent variable (output).

- For every input, there is only one unique output. If an input has more than one output, it's a relation, but not a function.

[More about functions](/Mathematics%20for%20DSA/Addon_Functions.md)

## Factors and Multiples

A **factor** of a number is a value that divides the number without leaving a remainder.

> Example: Factors of 6 are 1, 2, 3, and 6.

A **multiple** of a number is obtained by multiplying that number with an integer.

> Example: Multiples of 6 include 6, 12, 18, 24, …

**Important differences**:

- Factors of a number are always less than or equal to the number.

- Multiples are always greater than or equal to the number.

## Greatest Common Divisor (GCD or HCF)

The **Greatest Common Divisor** (GCD) — also called the **Highest Common Factor** (HCF) — of two numbers is the largest number that divides both numbers exactly. GCD of two numbers is less than or equal to the smaller of the two.

> Example: GCD(5, 15)
>
> Factors of 5 -> {1, 5}
>
> Factors of 15 -> {1, 3, 5, 15}
>
> Common factors -> {1, 5}
>
> GCD of 5 and 15 is the largest number of {1, 5}; i.e., 5.

**[Python implementation of GCD](./Addon_GCD.md)**

## Least Common Multiple (LCM)

The Least Common Multiple (LCM) of two numbers `x` and `y` is the smallest number `p` such that both `x` and `y` divide `p` without any remainder.

- Value of LCM lies in the range : $\text{max}(x, y)\leq LCM(x, y) \leq x \cdot y$

> Example: LCM(3, 4)
>
> Multiples of 3 -> {3, 6, 9, 12, ....}
>
> Multiples of 4 -> {4, 8, 12, 16, ....}
>
> Common multiples -> {12, 24, ...}
>
> LCM of 3 and 4 is the smallest common multiple of {12, 24, ...}; i.e., 12.

**[Python implementation of LCM](./Addon_LCM.md)**

## Logarithm

A logarithm is the inverse operation of exponentiation. It answers the question: $\text{"To what power must the base b be raised to get a?"}$

$\large b^{x} = a \;\;$ can be written as $\;\; x = log_{b}a$

- If the base is not specified, it is assumed to be the natural base `e`, and the logarithm is called the **natural logarithm**, written as $ \ln a$.

- Logarithm is the inverse operation of exponentiation.

$$log_b(b^x) = x \;\; \text{and} \;\; b^{log_b(x)} = x$$

### Important properties of Logarithm

- If $b^y = x$ then $log_b(x) = y$

- $log_b(b) = 1$ For any valid base $b$; i.e.; $b \ge 0$ and $b \neq 1$

- **Inverse Identity** : 

$$log_b(b^x) = x$$

- **Power Rule** : 

$$log_b(y^a) = a \cdot log_b(y)$$

- **Change of Base (Property-1)**

$$log_b(x) = \frac{log_a(x)}{log_a(b)}$$

- **Change of Base (Property-2)**

$$log_b(x) = \frac{1}{log_x(b)}$$

- **Logarithm as a Base Conversion Tool**

$$\large b^x = c^{\frac{x}{log_b(c)}}$$


## Prime Numbers

Prime numbers are natural numbers greater than 1 that are divisible only by 1 and themselves. In other words, a prime number has exactly two distinct positive divisors.

The first prime number is **2**, which is also the **only even prime number**. All other prime numbers are odd because any even number greater than 2 is divisible by 2 and thus not prime.

$$\large 2, \;3, \;5, \;7,\; 11,\; 13,\; 17, \;...$$

### Properties of Prime Numbers:

- 2 is the smallest and only even prime.

- Every natural number greater than 1 is either a prime or can be written as a product of prime numbers (this is called the Fundamental Theorem of Arithmetic).

- Prime numbers are the building blocks of integers. For example:

$$\text{60 = }2^3 \cdot 3 \cdot 5$$

- There are infinitely many prime numbers, a fact proven by Euclid over 2000 years ago.

- Prime numbers are critical in number theory and have applications in areas like hashing, cryptography (e.g., RSA encryption), and algorithm design.

**[Prime computation](./Addon_Prime.md)**

## Arithmetic Progression

An arithmetic progression (AP) is a sequence of numbers in which the difference between any two successive terms is constant. This fixed value is called the common difference.

$$1,\;5,\;9,\;13,\;......\;,\;n$$

- The $n$-th term of an AP is given by:

$$a_n = a_1 + (n-1)\cdot d$$

>Where; $a_1$ is the first term and $d$ is the common difference.

- The sum of first $n$ terms in an AP is:

$$S_n = \frac{n}{2} \cdot [2a_1 + (n-1) \cdot d]$$

### Common summation cases:

- Sum of first $n$ natural numbers:

$$S_n = \frac{n (n + 1)}{2}$$

- Sum of first $n$ odd numbers:

$$S_n = n^2$$

- Sum of first $n$ even numbers:

$$S_n = n(n+1)$$


## Geometric Progression

A geometric progression (GP) is a sequence of numbers where each term after the first is found by multiplying the previous one by a fixed, non-zero number called the common ratio.

$$1,\;2,\;4,\;8,\;......\;,\;n$$

- The $n^{\text{th}}$ term of a GP is:

$$a_n = a_1 \cdot r^{n-1}$$

> Where $r$ is the common ratio.

- Sum of first $n$ terms is:

$$S_n = \frac{a_1(1-r^n)}{1 - r}\;\;(\text{for r}\ne 1)$$

- The sum of an infinite GP (where $|r| \lt 1$) is:

$$S = \frac{a_1}{1 - r}$$

## Harmonic Progression

A harmonic progression (HP) is a sequence of numbers whose reciprocals form an arithmetic progression.

That is, if:

$$a_1\;, a_2\;, a_1\;, a_3\;, ...$$

is an HP, then:

$$\frac{1}{a_1},\;\frac{1}{a_2},\;\frac{1}{a_3},\;......$$

is an AP.

The $n^{\text{th}}$ term of a HP is:

$$a_n = \frac{1}{a + (n-1) \cdot d}$$

> Where $a$ and $d$ are the first term and common difference of the corresponding AP of reciprocals.

- There is no standard closed-form formula for the sum of $n$ terms of an HP like in AP or GP. However, for a finite HP derived from AP with known first and last terms:

$$S_n \approx \sum^{n}_{i = 1} \frac{1}{a_i}$$


## Permutations and Combinations

Permutations and combinations are methods of counting the number of ways to select or arrange elements from a given set.

- If we have $n$ distinct items, then the total number of ways to arrange all of them is $n!$ (n factorial).

> $n! = n \cdot (n-1) \cdot (n-2) \cdot \ldots \cdot 1$

- The key difference between permutation and combination is:

    - In permutation; The order matters.

    - In Combination; The order does not matters.

### Ranges

Consider a list $\text{lis = [start, end]}$ of consecutive integers:

- If both bounds are inclusive: $\text{count} = (end - start + 1)$

- If one bound is exclusive (as in Python's range): $\text{count} = (end - start)$

- If both bounds are exclusive: $\text{count} = (end - start - 1)$

### Permutations

A permutation is a way of arranging a set of elements in a specific order. If the order changes, it’s considered a different permutation.

- The number of permutations of $n$ items taken $r$ at a time is:

$$P(n, r) = \frac{n!}{(n-r)!}$$

> This refers to the number of ways to arrange $r$ objects selected from a set of $n$ distinct objects. 

- The number of permutations when all $n$ items are selected:

$$P(n, n) = n!$$

- In-place permutations are important in DSA for problems like:

    - Generating all permutations of a string or array (`backtracking`)

    - Solving problems on permutations of arrays (e.g., next permutation in lexicographical order)

    - Reordering data efficiently

### Combinations

A combination is a way of selecting items without considering order. Changing the order of selected items does not produce a new combination.

The number of combinations of $n$ items taken $r$ at a time is:

$$C(n, r) = \frac{n!}{r! \cdot (n-r)!}$$

- The number of combinations when all $n$ items are selected:

$$C(n, n) = 1$$

Understanding permutations and combinations is crucial for solving:

- **Combinatorial problems**: counting subsets, arrangements, k-combinations

- **Probability-based** problems in competitive programming and coding interviews

- **Backtracking algorithms**: generating all possible selections or arrangements

- **Dynamic Programming** problems involving combinatorics (e.g., Pascal's Triangle, binomial coefficients)

- **Hashing problems** involving counting unique combinations of elements

In real-world code (e.g., CP or interviews), you often avoid computing full factorials due to:

- Overflow

- Redundancy

- Modulo arithmetic

You might use:

- Precomputed factorials and modular inverses for fast $C(n, r) \bmod m$

- Memoization or DP to compute combinations dynamically (e.g., using Pascal’s Triangle)


## Square Root

The square root of a number is the value that, when multiplied by itself, gives the original number.

For any non-negative number $n$, its square root is denoted by:

$$\sqrt{n} = n^{1/2}$$

### Geometric Interpretation

You can visualize the square root of a number as the **length of the side of a square** whose **area is equal to that number**. 

For example: - The square root of 16 is 4, because $4^2 = 16$.  

**[Finding Square root](./Addon_Square_root.md)**
