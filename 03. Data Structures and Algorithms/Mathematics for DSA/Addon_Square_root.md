# Square root

## Square Root using Newton-Raphson Method

The **Newton-Raphson method** is a fast iterative method for approximating roots of real-valued functions. For $\sqrt{n}$, we solve the equation:

$$f(x) = x^2 - n = 0$$

Its root (i.e., where $f(x) = 0$) is the square root of $n$.

```python
def sqrt_newton(n: float) -> float:
    if n < 0:
        return -1  # Square root of negative number is not real
    x = n  # Initial guess
    e = 0.000001  # Tolerance level
    while True:
        next_x = 0.5 * (x + n / x)  # Newton-Raphson formula
        if abs(x - next_x) < e:  # Converged to desired precision
            break
        x = next_x
    return x
```

- Time complexity : $\text{O(log(log(} \frac{n}{e}\text{)))}$

## Square root using binary search

Binary search is a powerful tool for numerical approximation when the function is monotonic. In this case, we search for $x$ such that $x^2 \approx n$.

```python
def sqrt_binary_search(n: float) -> float:
    if n < 0:
        return -1  # Invalid input
    if n == 0:
        return 0  # Square root of 0 is 0

    low, high = 0, max(1, n)  # Set bounds depending on input
    e = 0.000001  # Tolerance level

    while high - low > e:
        mid = (low + high) / 2
        mid_squared = mid * mid

        if abs(mid_squared - n) < e:
            return mid
        elif mid_squared < n:
            low = mid
        else:
            high = mid

    return (low + high) / 2  # Final approximation
```

- Time complexity : $\text{O(log(} \frac{n}{e} \text{))}$