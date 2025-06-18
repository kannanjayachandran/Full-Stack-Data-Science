# Least Common Multiple - Python

## Naive Implementation

```py
def lcm_normal(a, b):
    large = max(a, b)
    for i in range(large, a * b + 1):
        if i % a == 0 and i % b == 0:
            return i
```

- Time Complexity: $O(n)$; (where `n` can go up to `a * b` in the worst case)

## Efficient LCM using GCD

Using the formula: 

$$\text{LCM}(a, b) = \frac{a \cdot b}{\text{GCD}(a, b)}$$

```py
def gcd_euclidean(a, b):
    while b:
        a, b = b, a % b
    return a


def lcm(a, b):
    return (a * b) // gcd_euclidean(a, b)
```

- Time Complexity: $O(\log(\min(a, b)))$.

