# Prime Numbers

In algorithmic problem-solving, we usually interact with prime numbers in two key ways:

1. **Primality Testing** – Checking if a given number is prime.

2. **Prime Enumeration** – Finding all prime numbers within a given range.

## Primality Testing

1. **Brute force approach** - Time Complexity $O(n)$

    - This method checks for divisibility from $2$ to $n-1$.

```python   
def is_prime(n: int) -> bool:
    if n <= 1:
        return False
    for i in range(2, n):
        if n % i == 0:
            return False
    return True
```

> This is inefficient for large values of `n`.


2. **Optimized Square Root Approach** — Time Complexity: $O(\sqrt{n})$

    - Instead of checking all numbers up to `n`, we only check up to $\sqrt{n}$ because if *n = a × b*, then at least one of a or b must be less than or equal to $\sqrt{n}$.

```python
def is_prime(n: int) -> bool:
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
```

3. **Further Optimization Using $6k \pm 1$** — Time Complexity: $O(\sqrt{n})$

    - All primes greater than 3 can be written in the form $6k \pm 1$. This property helps reduce the number of iterations.

```python
def is_prime(n: int) -> bool:
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False

    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True
```

4. **Miller–Rabin Primality Test** (Probabilistic) — Time Complexity: $O(k \log^3 n)$

The **Miller–Rabin** test is a probabilistic algorithm that efficiently determines if a number is probably prime or composite. It is widely used in cryptographic systems and competitive programming due to its speed and reliability for large numbers.

**Idea:**

- Express $n-1$ as $2^r \cdot d$.

- For a randomly chosen base $a$, check:

    - If $a^d \mod n = 1$, or

    - If $a^{2^i \cdot d} \mod n = n - 1$ for some $0 \le i < r$.

- If neither condition holds, then n is composite.

```python
import random

def is_probably_prime(n: int, k: int = 5) -> bool:
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0:
        return False

    # Write n-1 as 2^r * d
    r, d = 0, n - 1
    while d % 2 == 0:
        d //= 2
        r += 1

    for _ in range(k):
        a = random.randint(2, n - 2)
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True
```

- $k$ is the number of iterations; higher $k$ improves accuracy.

- Deterministic variants exist for $n$ up to $2^{64}$ with specific bases

## Finding All Primes in a Range

1. **Sieve of Eratosthenes** — Time Complexity: $O(n \log ( \log n))$

This classic algorithm finds all prime numbers up to n efficiently. This is how it works:

- Start with all numbers marked as prime.

- Begin with the first prime 2, and mark all multiples of 2 as non-prime.

- Move to the next unmarked number and repeat.

```python
def sieve(n: int) -> list:
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False

    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False

    return [i for i, prime in enumerate(is_prime) if prime]
```


