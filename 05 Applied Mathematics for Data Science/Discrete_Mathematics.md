<!-- 
    Author : Kannan Jayachandran
    File : Discrete_mathematics.md
    Section : Applied Mathematics for Data Science
 -->

<h1 align="center" style="color: orange"> Discrete Mathematics</h1>

---

## Combinatorics

Let us take a detour from linear algebra and look at some basic combinatorics. `Combinatorics` is the branch of mathematics that deals with counting. It is used in probability theory, statistics, and machine learning. Python has extensive support for combinatorics also through libraries like `itertools`, `SciPy`, `NetworkX`, `SymPy`, and `combinations`.

### Permutations

Permutations are the number of ways in which we can arrange a set of objects. 

> For example, if we have three objects, say `a`, `b`, and `c`, then the number of ways in which we can arrange them is $3! = 3 \times 2 \times 1 = 6$. 

Formally we can define permutations as;

**Number of ordered configurations to arrange $k$ objects from a set of $n$ objects, without replacement and without repetition.**

$$P(n, r) = \frac{n!}{(n-r)!}$$

where $\color{#FF9900}n$ is the number of objects and $\color{#FF9900}r$ is the number of objects we want to arrange.

### Combinations

Combinations are the number of ways in which we can select a subset of objects from a set of objects.

> For example, if we have three objects, say `a`, `b`, and `c`, then the number of ways in which we can select two objects is $3C2 = 3$.

Formally we can define combinations as;

**Number of unordered configurations to select $k$ objects from a set of $n$ objects, without replacement and without repetition.**

$$C(n, r) = \frac{n!}{r!(n-r)!}$$

where $\color{#FF9900}n$ is the number of objects and $\color{#FF9900}r$ is the number of objects we want to select.
