# GCD - Python

## Naive Method

```py
def gcd_normal(a, b):
    small = min(a, b)
    for i in range(1, small + 1):
        if a % i == 0 and b % i == 0:
            gcd = i
    return gcd
```

Time Complexity: $O(n)$; (where n is the minimum of the two numbers)

## Euclidean Algorithm

The Euclidean Algorithm is a much faster approach based on the principle:

$$GCD(a, b) = GCD(b, a \% b)$$

### Recursive Implementation

```py
def gcd_euclidean(a, b):
    if b == 0:
        return a
    return gcd_euclidean(b, a % b)
```

### Iterative Implementation

```py
def gcd_euclidean(a, b):
    while b:
        a, b = b, a % b
    return a
```

Time Complexity: $O(\log(\min(a, b)))$

- This method is widely used in competitive programming and real-world systems because of its logarithmic performance.