<h1 align="center"> Dynamic Programming </h1>

**Dynamic Programming (DP)** is an optimization technique that solves complex problems by:
1. Breaking them into overlapping subproblems
2. Solving each subproblem once
3. Storing results to avoid recomputation (memoization)

**Key Characteristics**:
- **Optimal substructure**: Optimal solution contains optimal solutions to subproblems
- **Overlapping subproblems**: Same subproblems solved multiple times

DP transforms exponential time complexity ($O(2^n)$) into polynomial ($O(n)$, $O(n^2)$)

$$\text{DP = Recursion + Memoization}$$

> Think recursively, optimize with storage.

## 1 Dimensional Dynamic Programming (1D DP)

State can be represented by a single variable (typically index or value).

### Python Implementation

**Classic Example: Fibonacci**:

```python
def fibonacci_naive(n):
    """
    Naive recursion - exponential time!
    
    T(n) = T(n-1) + T(n-2) + O(1) → O(2^n)
    """
    if n <= 1:
        return n
    return fibonacci_naive(n - 1) + fibonacci_naive(n - 2)

# Time: O(2^n), Space: O(n) call stack


def fibonacci_memoization(n, memo=None):
    """
    Top-down DP (memoization).
    
    Store computed results to avoid recomputation.
    """
    if memo is None:
        memo = {}
    
    if n in memo:
        return memo[n]
    
    if n <= 1:
        return n
    
    memo[n] = fibonacci_memoization(n - 1, memo) + fibonacci_memoization(n - 2, memo)
    return memo[n]

# Time: O(n), Space: O(n)


def fibonacci_tabulation(n):
    """
    Bottom-up DP (tabulation).
    
    Build solution iteratively from base cases.
    """
    if n <= 1:
        return n
    
    dp = [0] * (n + 1)
    dp[1] = 1
    
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    
    return dp[n]

# Time: O(n), Space: O(n)


def fibonacci_optimized(n):
    """
    Space-optimized DP.
    
    Only need last 2 values, not entire array.
    """
    if n <= 1:
        return n
    
    prev2, prev1 = 0, 1
    
    for _ in range(2, n + 1):
        current = prev1 + prev2
        prev2, prev1 = prev1, current
    
    return prev1

# Time: O(n), Space: O(1)
```

**Climbing Stairs**:

```python
def climb_stairs(n):
    """
    Count ways to climb n stairs (can take 1 or 2 steps).
    
    Pattern: dp[i] = dp[i-1] + dp[i-2]
    Same recurrence as Fibonacci!
    
    Intuition: To reach step i, came from i-1 (1 step) or i-2 (2 steps)
    """
    if n <= 2:
        return n
    
    dp = [0] * (n + 1)
    dp[1], dp[2] = 1, 2
    
    for i in range(3, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    
    return dp[n]

# Time: O(n), Space: O(n)


def climb_stairs_optimized(n):
    """Space-optimized version."""
    if n <= 2:
        return n
    
    prev2, prev1 = 1, 2
    
    for _ in range(3, n + 1):
        current = prev1 + prev2
        prev2, prev1 = prev1, current
    
    return prev1

# Time: O(n), Space: O(1)
```

**House Robber**:

```python
def rob(nums):
    """
    Maximum money you can rob from houses (can't rob adjacent houses).
    
    Example: [2,7,9,3,1] → rob 2,9,1 = 12
    
    Recurrence: dp[i] = max(dp[i-1], dp[i-2] + nums[i])
    - Don't rob i: take dp[i-1]
    - Rob i: can't rob i-1, so take dp[i-2] + nums[i]
    """
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]
    
    n = len(nums)
    dp = [0] * n
    dp[0] = nums[0]
    dp[1] = max(nums[0], nums[1])
    
    for i in range(2, n):
        dp[i] = max(dp[i - 1], dp[i - 2] + nums[i])
    
    return dp[n - 1]

# Time: O(n), Space: O(n)


def rob_optimized(nums):
    """Space-optimized version."""
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]
    
    prev2 = nums[0]
    prev1 = max(nums[0], nums[1])
    
    for i in range(2, len(nums)):
        current = max(prev1, prev2 + nums[i])
        prev2, prev1 = prev1, current
    
    return prev1

# Time: O(n), Space: O(1)
```

**Coin Change**:

```python
def coin_change(coins, amount):
    """
    Find minimum number of coins to make amount.
    
    Example: coins=[1,2,5], amount=11 → 3 (5+5+1)
    
    Recurrence: dp[i] = min(dp[i - coin] + 1) for all coins
    """
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0  # Base case: 0 coins for amount 0
    
    for i in range(1, amount + 1):
        for coin in coins:
            if i >= coin:
                dp[i] = min(dp[i], dp[i - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1

# Time: O(amount × coins), Space: O(amount)


def coin_change_ways(coins, amount):
    """
    Count number of ways to make amount (order doesn't matter).
    
    Example: coins=[1,2,5], amount=5 → 4 ways
    [5], [2,2,1], [2,1,1,1], [1,1,1,1,1]
    """
    dp = [0] * (amount + 1)
    dp[0] = 1  # One way to make 0: use no coins
    
    # Important: iterate coins in outer loop to avoid counting permutations
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] += dp[i - coin]
    
    return dp[amount]

# Time: O(amount × coins), Space: O(amount)
```

**Longest Increasing Subsequence (LIS)**:

```python
def length_of_lis(nums):
    """
    Find length of longest increasing subsequence.
    
    Example: [10,9,2,5,3,7,101,18] → 4 ([2,3,7,101])
    
    dp[i] = length of LIS ending at index i
    Recurrence: dp[i] = max(dp[j] + 1) for all j < i where nums[j] < nums[i]
    """
    if not nums:
        return 0
    
    n = len(nums)
    dp = [1] * n  # Each element is a subsequence of length 1
    
    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    
    return max(dp)

# Time: O(n²), Space: O(n)


def length_of_lis_optimized(nums):
    """
    O(n log n) solution using binary search.
    
    Maintain array of smallest tail elements for each length.
    """
    import bisect
    
    if not nums:
        return 0
    
    tails = []  # tails[i] = smallest tail of all LIS of length i+1
    
    for num in nums:
        # Find position where num should be inserted
        pos = bisect.bisect_left(tails, num)
        
        if pos == len(tails):
            tails.append(num)  # Extend LIS
        else:
            tails[pos] = num  # Replace with smaller tail
    
    return len(tails)

# Time: O(n log n), Space: O(n)
```

**Maximum Subarray (Kadane's Algorithm)**:

```python
def max_subarray(nums):
    """
    Find maximum sum of contiguous subarray.
    
    Example: [-2,1,-3,4,-1,2,1,-5,4] → 6 ([4,-1,2,1])
    
    Recurrence: dp[i] = max(nums[i], dp[i-1] + nums[i])
    Either start new subarray or extend previous
    """
    if not nums:
        return 0
    
    max_sum = nums[0]
    current_sum = nums[0]
    
    for i in range(1, len(nums)):
        current_sum = max(nums[i], current_sum + nums[i])
        max_sum = max(max_sum, current_sum)
    
    return max_sum

# Time: O(n), Space: O(1)
# This is actually greedy, but often taught with DP


def max_subarray_with_indices(nums):
    """Return max sum and the subarray indices."""
    if not nums:
        return 0, 0, 0
    
    max_sum = nums[0]
    current_sum = nums[0]
    start = end = 0
    temp_start = 0
    
    for i in range(1, len(nums)):
        if nums[i] > current_sum + nums[i]:
            current_sum = nums[i]
            temp_start = i
        else:
            current_sum += nums[i]
        
        if current_sum > max_sum:
            max_sum = current_sum
            start = temp_start
            end = i
    
    return max_sum, start, end

# Time: O(n), Space: O(1)
```

**Decode Ways**:

```python
def num_decodings(s):
    """
    Count ways to decode string where 'A'=1, 'B'=2, ..., 'Z'=26.
    
    Example: "226" → 3 ("BZ", "VF", "BBF")
    
    Recurrence: dp[i] = dp[i-1] + dp[i-2]
    - Single digit (if valid): dp[i-1]
    - Two digits (if valid): dp[i-2]
    """
    if not s or s[0] == '0':
        return 0
    
    n = len(s)
    dp = [0] * (n + 1)
    dp[0] = 1  # Empty string
    dp[1] = 1  # First character
    
    for i in range(2, n + 1):
        # One digit
        one_digit = int(s[i - 1])
        if 1 <= one_digit <= 9:
            dp[i] += dp[i - 1]
        
        # Two digits
        two_digits = int(s[i - 2:i])
        if 10 <= two_digits <= 26:
            dp[i] += dp[i - 2]
    
    return dp[n]

# Time: O(n), Space: O(n)


def num_decodings_optimized(s):
    """Space-optimized version."""
    if not s or s[0] == '0':
        return 0
    
    prev2 = 1  # dp[i-2]
    prev1 = 1  # dp[i-1]
    
    for i in range(1, len(s)):
        current = 0
        
        # One digit
        if s[i] != '0':
            current += prev1
        
        # Two digits
        two_digits = int(s[i - 1:i + 1])
        if 10 <= two_digits <= 26:
            current += prev2
        
        prev2, prev1 = prev1, current
    
    return prev1

# Time: O(n), Space: O(1)
```

**Word Break**:

```python
def word_break(s, word_dict):
    """
    Check if string can be segmented into dictionary words.
    
    Example: s="leetcode", dict=["leet","code"] → True
    
    dp[i] = True if s[0:i] can be segmented
    """
    word_set = set(word_dict)
    n = len(s)
    dp = [False] * (n + 1)
    dp[0] = True  # Empty string
    
    for i in range(1, n + 1):
        for j in range(i):
            # Check if s[0:j] can be segmented and s[j:i] is a word
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                break
    
    return dp[n]

# Time: O(n² × m) where m = avg word length
# Space: O(n)


def word_break_optimized(s, word_dict):
    """Optimized with max word length."""
    word_set = set(word_dict)
    max_len = max(len(word) for word in word_dict) if word_dict else 0
    
    n = len(s)
    dp = [False] * (n + 1)
    dp[0] = True
    
    for i in range(1, n + 1):
        # Only check words up to max_len characters back
        for j in range(max(0, i - max_len), i):
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                break
    
    return dp[n]

# Time: O(n × max_len), Space: O(n)
```

**Jump Game**:

```python
def can_jump(nums):
    """
    Check if can jump from first to last index.
    Each element is max jump length from that position.
    
    Example: [2,3,1,1,4] → True (jump 1→3→4)
    Example: [3,2,1,0,4] → False (stuck at index 3)
    
    Greedy approach (also counts as DP).
    """
    max_reach = 0
    
    for i in range(len(nums)):
        if i > max_reach:
            return False  # Can't reach this position
        
        max_reach = max(max_reach, i + nums[i])
        
        if max_reach >= len(nums) - 1:
            return True
    
    return True

# Time: O(n), Space: O(1)


def min_jumps(nums):
    """
    Find minimum number of jumps to reach last index.
    
    Example: [2,3,1,1,4] → 2 (jump 1→3→4)
    """
    if len(nums) <= 1:
        return 0
    
    n = len(nums)
    dp = [float('inf')] * n
    dp[0] = 0
    
    for i in range(n):
        for j in range(1, nums[i] + 1):
            if i + j < n:
                dp[i + j] = min(dp[i + j], dp[i] + 1)
    
    return dp[n - 1] if dp[n - 1] != float('inf') else -1

# Time: O(n²), Space: O(n)


def min_jumps_optimized(nums):
    """
    O(n) greedy solution using BFS-like approach.
    """
    if len(nums) <= 1:
        return 0
    
    jumps = 0
    current_end = 0
    farthest = 0
    
    for i in range(len(nums) - 1):
        farthest = max(farthest, i + nums[i])
        
        if i == current_end:
            jumps += 1
            current_end = farthest
            
            if current_end >= len(nums) - 1:
                break
    
    return jumps

# Time: O(n), Space: O(1)
```

### Complexity Analysis

| Problem | State | Transition | Time | Space |
|---------|-------|------------|------|-------|
| Fibonacci | dp[i] | dp[i-1] + dp[i-2] | O(n) | O(1) optimized |
| Climbing Stairs | dp[i] | dp[i-1] + dp[i-2] | O(n) | O(1) optimized |
| House Robber | dp[i] | max(dp[i-1], dp[i-2]+nums[i]) | O(n) | O(1) optimized |
| Coin Change | dp[amount] | min(dp[amount-coin]+1) | O(n×m) | O(n) |
| LIS | dp[i] | max(dp[j]+1) for j<i | O(n²) | O(n) |
| Max Subarray | dp[i] | max(nums[i], dp[i-1]+nums[i]) | O(n) | O(1) |
| Word Break | dp[i] | dp[j] && s[j:i] in dict | O(n²) | O(n) |

**Space Optimization Pattern**:
If `dp[i]` only depends on fixed number of previous states (like `dp[i-1]`, `dp[i-2]`), can optimize from $O(n)$ to $O(1)$ by keeping only those states.

### Common Questions

1. "How to identify DP problems?"
   - **Answer**:
     - Ask: "Can I break into smaller subproblems?"
     - Check for optimal substructure (optimal solution uses optimal subsolutions)
     - Check for overlapping subproblems (same subproblems computed multiple times)
     - Keywords: "maximum/minimum", "count ways", "longest/shortest"
     - If brute force is exponential due to repeated work → likely DP

2. "Top-down (memoization) vs bottom-up (tabulation)?"
   - **Answer**:
     - **Top-down**: Recursive, natural to write, only computes needed subproblems
     - **Bottom-up**: Iterative, avoids recursion overhead, computes all subproblems
     - **When to use**:
       - Top-down: Complex state dependencies, not all states needed
       - Bottom-up: Simple dependencies, need all states, avoid stack overflow
     - **Conversion**: Can usually convert either direction

3. "How to optimize space in DP?"
   - **Answer**:
     - Identify which previous states are actually needed
     - If dp[i] only depends on dp[i-1], dp[i-2]: use 2 variables
     - If dp[i] depends on previous row in 2D: use 2 arrays
     - Rolling array technique: reuse same array with modulo indexing

4. "Common DP patterns?"
   - **Answer**:
     - **Linear sequence**: Process elements left to right
     - **Decision at each step**: Include/exclude, take/skip
     - **Min/max optimization**: Find best choice among options
     - **Counting**: Sum all ways to reach state
     - **State compression**: Use bitmask for subset states

### Problem-Solving Template

```python
def dp_template(input_data):
    """
    Step 1: Define state
    - What does dp[i] represent?
    - What are the dimensions?
    
    Step 2: Base cases
    - Smallest subproblems with known answers
    
    Step 3: Recurrence relation
    - How to compute dp[i] from smaller subproblems?
    
    Step 4: Iteration order
    - Ensure dependencies computed before use
    
    Step 5: Final answer
    - Where is the answer stored?
    """
    
    # Initialize DP array
    dp = [initial_value] * size
    
    # Base cases
    dp[0] = base_case
    
    # Fill DP table
    for i in range(1, size):
        # Recurrence relation
        dp[i] = compute_from_previous(dp)
    
    return dp[-1]  # or max(dp), etc.
```

### Common Mistakes
- Not initializing base cases correctly
- Wrong iteration order (computing dp[i] before its dependencies)
- Off-by-one errors in array indices
- Forgetting to handle empty input
- Not considering all transition cases

### Edge Cases
- Empty input
- Single element
- All elements same
- Negative numbers (when applicable)
- Very large inputs (integer overflow, though rare in Python)

---

## 2 Dimensional Dynamic Programming (2D DP)

State requires two variables (typically two indices, or index + value).

### Common Patterns

- **Grid-based**: Moving through 2D grid (path counting, obstacles)
- **Two sequences**: Comparing/matching two strings or arrays (edit distance, LCS)
- **Range queries**: Processing subarrays or substrings
- **Optimization over 2D space**: Making decisions based on two parameters

**Key Insight**: Build solution by processing one dimension at a time, using results from previous dimension.

### Python Implementation

**Unique Paths**:

```python
def unique_paths(m, n):
    """
    Count paths from top-left to bottom-right in m×n grid.
    Can only move right or down.
    
    Example: 3×7 grid → 28 paths
    
    dp[i][j] = dp[i-1][j] + dp[i][j-1]
    """
    dp = [[0] * n for _ in range(m)]
    
    # Base cases: first row and column have only 1 path
    for i in range(m):
        dp[i][0] = 1
    for j in range(n):
        dp[0][j] = 1
    
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
    
    return dp[m - 1][n - 1]

# Time: O(m×n), Space: O(m×n)


def unique_paths_optimized(m, n):
    """Space-optimized version using 1D array."""
    dp = [1] * n  # Previous row
    
    for i in range(1, m):
        for j in range(1, n):
            dp[j] += dp[j - 1]  # dp[j] = old dp[j] + dp[j-1]
    
    return dp[n - 1]

# Time: O(m×n), Space: O(n)


def unique_paths_with_obstacles(grid):
    """
    Count paths in grid with obstacles (marked as 1).
    
    Example: [[0,0,0],[0,1,0],[0,0,0]] → 2 paths
    """
    if not grid or grid[0][0] == 1:
        return 0
    
    m, n = len(grid), len(grid[0])
    dp = [[0] * n for _ in range(m)]
    dp[0][0] = 1
    
    # First column
    for i in range(1, m):
        dp[i][0] = dp[i - 1][0] if grid[i][0] == 0 else 0
    
    # First row
    for j in range(1, n):
        dp[0][j] = dp[0][j - 1] if grid[0][j] == 0 else 0
    
    # Fill rest of grid
    for i in range(1, m):
        for j in range(1, n):
            if grid[i][j] == 1:
                dp[i][j] = 0
            else:
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
    
    return dp[m - 1][n - 1]

# Time: O(m×n), Space: O(m×n)
```

**Minimum Path Sum**:

```python
def min_path_sum(grid):
    """
    Find path from top-left to bottom-right with minimum sum.
    Can only move right or down.
    
    Example: [[1,3,1],[1,5,1],[4,2,1]] → 7 (1→3→1→1→1)
    
    dp[i][j] = grid[i][j] + min(dp[i-1][j], dp[i][j-1])
    """
    if not grid:
        return 0
    
    m, n = len(grid), len(grid[0])
    dp = [[0] * n for _ in range(m)]
    dp[0][0] = grid[0][0]
    
    # First row
    for j in range(1, n):
        dp[0][j] = dp[0][j - 1] + grid[0][j]
    
    # First column
    for i in range(1, m):
        dp[i][0] = dp[i -1][0] + grid[i][0]
    
    # Fill rest
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = grid[i][j] + min(dp[i - 1][j], dp[i][j - 1])
    
    return dp[m - 1][n - 1]

# Time: O(m×n), Space: O(m×n)


def min_path_sum_optimized(grid):
    """Space-optimized to O(n)."""
    if not grid:
        return 0
    
    n = len(grid[0])
    dp = [0] * n
    dp[0] = grid[0][0]
    
    # Initialize first row
    for j in range(1, n):
        dp[j] = dp[j - 1] + grid[0][j]
    
    # Process remaining rows
    for i in range(1, len(grid)):
        dp[0] += grid[i][0]  # Update first column
        
        for j in range(1, n):
            dp[j] = grid[i][j] + min(dp[j], dp[j - 1])
    
    return dp[n - 1]

# Time: O(m×n), Space: O(n)
```

**Longest Common Subsequence (LCS)**:

```python
def longest_common_subsequence(text1, text2):
    """
    Find length of longest subsequence common to both strings.
    
    Example: "abcde", "ace" → 3 ("ace")
    
    dp[i][j] = LCS length of text1[0:i] and text2[0:j]
    
    Recurrence:
    - If text1[i-1] == text2[j-1]: dp[i][j] = dp[i-1][j-1] + 1
    - Else: dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    """
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    return dp[m][n]

# Time: O(m×n), Space: O(m×n)


def lcs_with_string(text1, text2):
    """Return actual LCS string, not just length."""
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    # Backtrack to find actual sequence
    lcs = []
    i, j = m, n
    
    while i > 0 and j > 0:
        if text1[i - 1] == text2[j - 1]:
            lcs.append(text1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    
    return ''.join(reversed(lcs))

# Time: O(m×n), Space: O(m×n)
```

**Edit Distance (Levenshtein Distance)**:

```python
def min_distance(word1, word2):
    """
    Minimum operations (insert/delete/replace) to convert word1 to word2.
    
    Example: "horse", "ros" → 3 (replace h→r, remove o, remove e)
    
    dp[i][j] = min operations to convert word1[0:i] to word2[0:j]
    """
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Base cases
    for i in range(m + 1):
        dp[i][0] = i  # Delete all
    for j in range(n + 1):
        dp[0][j] = j  # Insert all
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # No operation
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # Delete
                    dp[i][j - 1],      # Insert
                    dp[i - 1][j - 1]   # Replace
                )
    
    return dp[m][n]

# Time: O(m×n), Space: O(m×n)


def min_distance_optimized(word1, word2):
    """Space-optimized to O(min(m,n))."""
    # Make word1 the shorter one
    if len(word1) > len(word2):
        word1, word2 = word2, word1
    
    m, n = len(word1), len(word2)
    prev = list(range(n + 1))
    
    for i in range(1, m + 1):
        curr = [i]  # First element is i
        
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                curr.append(prev[j - 1])
            else:
                curr.append(1 + min(prev[j], curr[j - 1], prev[j - 1]))
        
        prev = curr
    
    return prev[n]

# Time: O(m×n), Space: O(min(m,n))
```

**Interleaving String**:

```python
def is_interleave(s1, s2, s3):
    """
    Check if s3 is formed by interleaving s1 and s2.
    
    Example: s1="aabcc", s2="dbbca", s3="aadbbcbcac" → True
    
    dp[i][j] = True if s3[0:i+j] is interleaving of s1[0:i] and s2[0:j]
    """
    m, n = len(s1), len(s2)
    
    if m + n != len(s3):
        return False
    
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True
    
    # First row (only s2)
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j - 1] and s2[j - 1] == s3[j - 1]
    
    # First column (only s1)
    for i in range(1, m + 1):
        dp[i][0] = dp[i - 1][0] and s1[i - 1] == s3[i - 1]
    
    # Fill rest
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = (
                (dp[i - 1][j] and s1[i - 1] == s3[i + j - 1]) or
                (dp[i][j - 1] and s2[j - 1] == s3[i + j - 1])
            )
    
    return dp[m][n]

# Time: O(m×n), Space: O(m×n)
```

**Maximal Square**:

```python
def maximal_square(matrix):
    """
    Find largest square containing only 1's.
    
    Example: [["1","0","1","0","0"],
              ["1","0","1","1","1"],
              ["1","1","1","1","1"]]
    → 4 (2×2 square)
    
    dp[i][j] = side length of largest square with bottom-right at (i,j)
    """
    if not matrix:
        return 0
    
    m, n = len(matrix), len(matrix[0])
    dp = [[0] * n for _ in range(m)]
    max_side = 0
    
    for i in range(m):
        for j in range(n):
            if matrix[i][j] == '1':
                if i == 0 or j == 0:
                    dp[i][j] = 1
                else:
                    dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
                
                max_side = max(max_side, dp[i][j])
    
    return max_side * max_side

# Time: O(m×n), Space: O(m×n)


def maximal_square_optimized(matrix):
    """Space-optimized to O(n)."""
    if not matrix:
        return 0
    
    n = len(matrix[0])
    prev = [0] * n
    max_side = 0
    
    for i in range(len(matrix)):
        curr = [0] * n
        for j in range(n):
            if matrix[i][j] == '1':
                if i == 0 or j == 0:
                    curr[j] = 1
                else:
                    curr[j] = min(prev[j], curr[j - 1], prev[j - 1]) + 1
                
                max_side = max(max_side, curr[j])
        
        prev = curr
    
    return max_side * max_side

# Time: O(m×n), Space: O(n)
```

**Distinct Subsequences**:

```python
def num_distinct(s, t):
    """
    Count distinct subsequences of s that equal t.
    
    Example: s="rabbbit", t="rabbit" → 3
    
    dp[i][j] = number of distinct subsequences of s[0:i] that equal t[0:j]
    """
    m, n = len(s), len(t)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Base case: empty t can be matched in one way (delete all)
    for i in range(m + 1):
        dp[i][0] = 1
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # Don't use s[i-1]
            dp[i][j] = dp[i - 1][j]
            
            # Use s[i-1] if it matches t[j-1]
            if s[i - 1] == t[j - 1]:
                dp[i][j] += dp[i - 1][j - 1]
    
    return dp[m][n]

# Time: O(m×n), Space: O(m×n)
```

**Regular Expression Matching**:

```python
def is_match(s, p):
    """
    Check if string matches pattern with '.' and '*'.
    '.' matches any single character
    '*' matches zero or more of preceding element
    
    Example: s="aa", p="a*" → True
    
    dp[i][j] = True if s[0:i] matches p[0:j]
    """
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True
    
    # Handle patterns like a*, a*b*, etc. (can match empty string)
    for j in range(2, n + 1):
        if p[j - 1] == '*':
            dp[0][j] = dp[0][j - 2]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j - 1] == '*':
                # Match zero occurrences
                dp[i][j] = dp[i][j - 2]
                
                # Match one or more occurrences
                if p[j - 2] == '.' or p[j - 2] == s[i - 1]:
                    dp[i][j] = dp[i][j] or dp[i - 1][j]
            elif p[j - 1] == '.' or p[j - 1] == s[i - 1]:
                dp[i][j] = dp[i - 1][j - 1]
    
    return dp[m][n]

# Time: O(m×n), Space: O(m×n)
```

**Wildcard Matching**:

```python
def is_match_wildcard(s, p):
    """
    Check if string matches pattern with '?' and '*'.
    '?' matches any single character
    '*' matches any sequence (including empty)
    
    Example: s="aa", p="*" → True
    
    dp[i][j] = True if s[0:i] matches p[0:j]
    """
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True
    
    # Handle leading '*' in pattern
    for j in range(1, n + 1):
        if p[j - 1] == '*':
            dp[0][j] = dp[0][j - 1]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j - 1] == '*':
                # Match empty or match one+ characters
                dp[i][j] = dp[i][j - 1] or dp[i - 1][j]
            elif p[j - 1] == '?' or p[j - 1] == s[i - 1]:
                dp[i][j] = dp[i - 1][j - 1]
    
    return dp[m][n]

# Time: O(m×n), Space: O(m×n)
```

**Palindromic Substrings**:

```python
def count_substrings(s):
    """
    Count palindromic substrings.
    
    Example: "abc" → 3 ("a", "b", "c")
    Example: "aaa" → 6 ("a", "a", "a", "aa", "aa", "aaa")
    
    dp[i][j] = True if s[i:j+1] is palindrome
    """
    n = len(s)
    dp = [[False] * n for _ in range(n)]
    count = 0
    
    # Single characters are palindromes
    for i in range(n):
        dp[i][i] = True
        count += 1
    
    # Length 2
    for i in range(n - 1):
        if s[i] == s[i + 1]:
            dp[i][i + 1] = True
            count += 1
    
    # Length 3+
    for length in range(3, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            
            if s[i] == s[j] and dp[i + 1][j - 1]:
                dp[i][j] = True
                count += 1
    
    return count

# Time: O(n²), Space: O(n²)


def longest_palindromic_substring(s):
    """
    Find longest palindromic substring.
    
    Example: "babad" → "bab" or "aba"
    """
    if not s:
        return ""
    
    n = len(s)
    dp = [[False] * n for _ in range(n)]
    start, max_len = 0, 1
    
    # Single characters
    for i in range(n):
        dp[i][i] = True
    
    # Length 2
    for i in range(n - 1):
        if s[i] == s[i + 1]:
            dp[i][i + 1] = True
            start = i
            max_len = 2
    
    # Length 3+
    for length in range(3, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            
            if s[i] == s[j] and dp[i + 1][j - 1]:
                dp[i][j] = True
                start = i
                max_len = length
    
    return s[start:start + max_len]

# Time: O(n²), Space: O(n²)
```

**Best Time to Buy and Sell Stock (Multiple Transactions)**:

```python
def max_profit_unlimited(prices):
    """
    Maximum profit with unlimited transactions.
    Can buy and sell same day.
    
    Example: [7,1,5,3,6,4] → 7 (buy 1 sell 5, buy 3 sell 6)
    """
    profit = 0
    
    for i in range(1, len(prices)):
        if prices[i] > prices[i - 1]:
            profit += prices[i] - prices[i - 1]
    
    return profit

# Time: O(n), Space: O(1)


def max_profit_k_transactions(k, prices):
    """
    Maximum profit with at most k transactions.
    
    dp[i][j] = max profit using at most i transactions up to day j
    
    Complex 2D DP problem.
    """
    if not prices or k == 0:
        return 0
    
    n = len(prices)
    
    # If k >= n/2, can do unlimited transactions
    if k >= n // 2:
        return max_profit_unlimited(prices)
    
    # dp[t][d] = max profit with t transactions up to day d
    dp = [[0] * n for _ in range(k + 1)]
    
    for t in range(1, k + 1):
        max_diff = -prices[0]  # max(dp[t-1][j] - prices[j])
        
        for d in range(1, n):
            # Don't transact on day d or transact
            dp[t][d] = max(dp[t][d - 1], prices[d] + max_diff)
            
            # Update max_diff for next iteration
            max_diff = max(max_diff, dp[t - 1][d] - prices[d])
    
    return dp[k][n - 1]

# Time: O(k×n), Space: O(k×n)
```

### Complexity Analysis

| Problem | State Dimension | Time | Space | Space Optimized |
|---------|----------------|------|-------|-----------------|
| Unique Paths | (row, col) | O(m×n) | O(m×n) | O(n) |
| Min Path Sum | (row, col) | O(m×n) | O(m×n) | O(n) |
| LCS | (i, j) in 2 strings | O(m×n) | O(m×n) | O(min(m,n)) |
| Edit Distance | (i, j) in 2 strings | O(m×n) | O(m×n) | O(min(m,n)) |
| Maximal Square | (row, col) | O(m×n) | O(m×n) | O(n) |
| Palindrome Substring | (start, end) | O(n²) | O(n²) | Hard to optimize |
| Stock with k Trans | (transactions, day) | O(k×n) | O(k×n) | O(n) |

**Space Optimization Patterns**:
1. **Rolling array**: If `dp[i][j]` only depends on `dp[i-1][...]`, keep only current and previous rows
2. **Single array**: If dependencies are local, can update in-place with careful ordering
3. **Compressed states**: Combine multiple dimensions if possible

### Common Questions

1. "How to approach 2D DP problems?"
   - **Answer**:
     1. Define state: What do both indices represent?
     2. Base cases: Initialize edges (first row/column)
     3. Recurrence: How to compute dp[i][j] from neighbors?
     4. Fill order: Typically left-to-right, top-to-bottom
     5. Answer location: Usually bottom-right corner

2. "Grid DP vs String DP - differences?"
   - **Answer**:
     - **Grid DP**: Spatial movement (right, down), physical paths
     - **String DP**: Sequence matching, character-by-character decisions
     - **Similarity**: Both 2D, but interpretation differs
     - **Base cases**: Grid uses edges, String uses empty prefixes

3. "How to backtrack to find actual solution?"
   - **Answer**: Store additional information during DP:
     - Parent pointers: Track which previous state led to current
     - Decision array: Record which choice was made at each state
     - Reconstruct: Start from final state, follow pointers backward

4. "Common 2D DP mistakes?"
   - **Answer**:
     - Wrong base case initialization
     - Off-by-one errors in indexing (especially with 0-indexed strings)
     - Not considering all transition cases
     - Wrong iteration order (computing state before dependencies ready)
     - Forgetting edge cases (empty strings, single element)

### Problem-Solving Patterns

1. **Grid traversal**: Paths, obstacles, costs
   - Template: `dp[i][j] = f(dp[i-1][j], dp[i][j-1])`

2. **String matching**: LCS, edit distance, pattern matching
   - Template: Compare characters, decide match/mismatch transitions

3. **Range DP**: Palindromes, burst balloons
   - Template: `dp[i][j]` represents range [i, j]
   - Fill diagonally (small ranges first)

4. **Interval DP**: Optimal splits, matrix chain multiplication
   - Template: Try all split points in range

## Common Mistakes

- Not initializing base cases correctly
- Accessing dp array out of bounds
- Confusing 0-indexed vs 1-indexed
- Not handling empty input
- Wrong space optimization (losing needed previous states)

## Edge Cases

- Empty strings/arrays
- Single element
- All elements same
- Very large dimensions (memory constraints)

---

## Advanced Dynamic Programming

**Advanced DP** includes:
- **State compression**: Using bitmasks to represent subsets
- **DP on trees**: Solving problems on tree structures
- **DP with data structures**: Combining DP with segment trees, tries, etc.
- **Multi-dimensional DP**: More than 2 dimensions
- **Optimization techniques**: Divide and conquer optimization, convex hull trick

**Key Insight**: Sometimes simple state representation won't work - need creative encoding (bitmasks, trees) or optimization tricks.

### Python Implementation

**Bitmask DP - Traveling Salesman Problem**:

```python
def tsp(dist):
    """
    Traveling Salesman Problem using bitmask DP.
    
    Find shortest path visiting all cities exactly once.
    
    State: dp[mask][i] = min cost to visit cities in 'mask' ending at city i
    mask: bitmask where bit j=1 means city j visited
    
    Example: 4 cities → 2^4 = 16 possible subsets
    """
    n = len(dist)
    ALL_VISITED = (1 << n) - 1  # All bits set
    
    # dp[mask][i] = min cost to reach state (mask, i)
    dp = [[float('inf')] * n for _ in range(1 << n)]
    dp[1][0] = 0  # Start at city 0
    
    for mask in range(1 << n):
        for u in range(n):
            # Check if city u is in current mask
            if not (mask & (1 << u)):
                continue
            
            if dp[mask][u] == float('inf'):
                continue
            
            # Try going to unvisited cities
            for v in range(n):
                if mask & (1 << v):  # Already visited
                    continue
                
                new_mask = mask | (1 << v)
                dp[new_mask][v] = min(dp[new_mask][v], 
                                     dp[mask][u] + dist[u][v])
    
    # Return to start city
    result = float('inf')
    for i in range(n):
        result = min(result, dp[ALL_VISITED][i] + dist[i][0])
    
    return result

# Time: O(n² × 2^n), Space: O(n × 2^n)
# Practical for n ≤ 20


def tsp_path(dist):
    """Return actual path, not just cost."""
    n = len(dist)
    ALL_VISITED = (1 << n) - 1
    
    dp = [[float('inf')] * n for _ in range(1 << n)]
    parent = [[None] * n for _ in range(1 << n)]
    dp[1][0] = 0
    
    for mask in range(1 << n):
        for u in range(n):
            if not (mask & (1 << u)) or dp[mask][u] == float('inf'):
                continue
            
            for v in range(n):
                if mask & (1 << v):
                    continue
                
                new_mask = mask | (1 << v)
                new_cost = dp[mask][u] + dist[u][v]
                
                if new_cost < dp[new_mask][v]:
                    dp[new_mask][v] = new_cost
                    parent[new_mask][v] = u
    
    # Backtrack to find path
    min_cost = float('inf')
    last_city = -1
    
    for i in range(n):
        cost = dp[ALL_VISITED][i] + dist[i][0]
        if cost < min_cost:
            min_cost = cost
            last_city = i
    
    # Reconstruct path
    path = []
    mask = ALL_VISITED
    current = last_city
    
    while current is not None:
        path.append(current)
        prev = parent[mask][current]
        if prev is not None:
            mask ^= (1 << current)
        current = prev
    
    path.append(0)  # Return to start
    return list(reversed(path)), min_cost

# Time: O(n² × 2^n), Space: O(n × 2^n)
```

**Bitmask DP - Assignment Problem**:

```python
def min_cost_assignment(cost):
    """
    Assign n tasks to n workers minimizing total cost.
    
    cost[i][j] = cost of assigning task j to worker i
    
    State: dp[mask] = min cost to assign tasks in 'mask'
    """
    n = len(cost)
    dp = [float('inf')] * (1 << n)
    dp[0] = 0
    
    for mask in range(1 << n):
        if dp[mask] == float('inf'):
            continue
        
        # Count assigned tasks = next worker to assign
        worker = bin(mask).count('1')
        
        if worker >= n:
            continue
        
        # Try assigning each unassigned task to this worker
        for task in range(n):
            if mask & (1 << task):  # Task already assigned
                continue
            
            new_mask = mask | (1 << task)
            dp[new_mask] = min(dp[new_mask], 
                              dp[mask] + cost[worker][task])
    
    return dp[(1 << n) - 1]

# Time: O(n × 2^n), Space: O(2^n)
# More efficient than O(n!) brute force
```

**Subset Sum with Bitmask**:

```python
def subset_sum_count(nums, target):
    """
    Count subsets that sum to target using bitmask.
    
    Alternative to standard DP when need to track which elements used.
    """
    n = len(nums)
    count = 0
    
    # Try all 2^n subsets
    for mask in range(1 << n):
        subset_sum = 0
        
        for i in range(n):
            if mask & (1 << i):
                subset_sum += nums[i]
        
        if subset_sum == target:
            count += 1
    
    return count

# Time: O(n × 2^n), Space: O(1)
# Only use bitmask approach for very small n (≤ 20)


def subset_sum_dp(nums, target):
    """
    Standard DP approach - more efficient for larger n.
    """
    dp = [0] * (target + 1)
    dp[0] = 1
    
    for num in nums:
        for s in range(target, num - 1, -1):
            dp[s] += dp[s - num]
    
    return dp[target]

# Time: O(n × target), Space: O(target)
# Better when target is reasonable
```

**DP on Trees - House Robber III**:

```python
def rob_tree(root):
    """
    House robber on binary tree (can't rob adjacent nodes).
    
    For each node, decide: rob it or not
    
    Returns: (rob_root, not_rob_root)
    """
    def dfs(node):
        if not node:
            return (0, 0)  # (rob, not_rob)
        
        left_rob, left_not_rob = dfs(node.left)
        right_rob, right_not_rob = dfs(node.right)
        
        # Rob current node: can't rob children
        rob = node.val + left_not_rob + right_not_rob
        
        # Don't rob current: take max from children
        not_rob = max(left_rob, left_not_rob) + max(right_rob, right_not_rob)
        
        return (rob, not_rob)
    
    return max(dfs(root))

# Time: O(n), Space: O(h) where h = height


def rob_tree_memo(root):
    """Alternative with memoization."""
    memo = {}
    
    def dfs(node, can_rob):
        if not node:
            return 0
        
        if (node, can_rob) in memo:
            return memo[(node, can_rob)]
        
        if can_rob:
            # Either rob this node or skip
            rob = node.val + dfs(node.left, False) + dfs(node.right, False)
            skip = dfs(node.left, True) + dfs(node.right, True)
            result = max(rob, skip)
        else:
            # Can't rob, must skip
            result = dfs(node.left, True) + dfs(node.right, True)
        
        memo[(node, can_rob)] = result
        return result
    
    return dfs(root, True)

# Time: O(n), Space: O(n)
```

**DP on Trees - Tree Distance**:

```python
def tree_diameter(edges, n):
    """
    Find diameter of tree (longest path between any two nodes).
    
    Use DP on tree: for each node, track max depth in each subtree.
    """
    from collections import defaultdict
    
    # Build adjacency list
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
    
    diameter = 0
    
    def dfs(node, parent):
        nonlocal diameter
        
        # Track two deepest paths from this node
        max1 = max2 = 0
        
        for neighbor in graph[node]:
            if neighbor == parent:
                continue
            
            depth = dfs(neighbor, node)
            
            if depth > max1:
                max2 = max1
                max1 = depth
            elif depth > max2:
                max2 = depth
        
        # Update diameter (path through this node)
        diameter = max(diameter, max1 + max2)
        
        # Return depth from this node
        return max1 + 1
    
    dfs(0, -1)
    return diameter

# Time: O(n), Space: O(n)
```

**Knapsack Variants**:

```python
def knapsack_01(weights, values, capacity):
    """
    0/1 Knapsack: each item used at most once.
    
    dp[i][w] = max value using first i items with capacity w
    """
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            # Don't take item i-1
            dp[i][w] = dp[i - 1][w]
            
            # Take item i-1 if fits
            if weights[i - 1] <= w:
                dp[i][w] = max(dp[i][w], 
                              dp[i - 1][w - weights[i - 1]] + values[i - 1])
    
    return dp[n][capacity]

# Time: O(n × capacity), Space: O(n × capacity)


def knapsack_01_optimized(weights, values, capacity):
    """Space-optimized to O(capacity)."""
    dp = [0] * (capacity + 1)
    
    for i in range(len(weights)):
        # Traverse backwards to avoid using same item twice
        for w in range(capacity, weights[i] - 1, -1):
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    
    return dp[capacity]

# Time: O(n × capacity), Space: O(capacity)


def knapsack_unbounded(weights, values, capacity):
    """
    Unbounded Knapsack: each item can be used unlimited times.
    
    Similar to coin change problem.
    """
    dp = [0] * (capacity + 1)
    
    for w in range(1, capacity + 1):
        for i in range(len(weights)):
            if weights[i] <= w:
                dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    
    return dp[capacity]

# Time: O(n × capacity), Space: O(capacity)


def knapsack_with_items(weights, values, capacity):
    """Return selected items, not just max value."""
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    # Fill DP table
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            dp[i][w] = dp[i - 1][w]
            
            if weights[i - 1] <= w:
                dp[i][w] = max(dp[i][w], 
                              dp[i - 1][w - weights[i - 1]] + values[i - 1])
    
    # Backtrack to find items
    selected = []
    w = capacity
    
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            selected.append(i - 1)
            w -= weights[i - 1]
    
    return dp[n][capacity], list(reversed(selected))

# Time: O(n × capacity), Space: O(n × capacity)
```

**Partition Equal Subset Sum**:

```python
def can_partition(nums):
    """
    Check if array can be partitioned into two equal sum subsets.
    
    Reduce to: can we make sum = total_sum / 2?
    This is subset sum problem.
    """
    total = sum(nums)
    
    if total % 2 != 0:
        return False
    
    target = total // 2
    dp = [False] * (target + 1)
    dp[0] = True
    
    for num in nums:
        # Traverse backwards
        for s in range(target, num - 1, -1):
            dp[s] = dp[s] or dp[s - num]
    
    return dp[target]

# Time: O(n × sum), Space: O(sum)


def partition_k_subsets(nums, k):
    """
    Check if can partition into k equal sum subsets.
    
    Use backtracking with DP (bitmask) for visited elements.
    """
    total = sum(nums)
    
    if total % k != 0:
        return False
    
    target = total // k
    nums.sort(reverse=True)
    
    if nums[0] > target:
        return False
    
    n = len(nums)
    memo = {}
    
    def backtrack(mask, k_remaining, current_sum):
        if k_remaining == 0:
            return True
        
        if current_sum == target:
            # Start new subset
            return backtrack(mask, k_remaining - 1, 0)
        
        if mask in memo:
            return memo[mask]
        
        for i in range(n):
            if mask & (1 << i):  # Already used
                continue
            
            if current_sum + nums[i] > target:
                continue
            
            if backtrack(mask | (1 << i), k_remaining, current_sum + nums[i]):
                memo[mask] = True
                return True
        
        memo[mask] = False
        return False
    
    return backtrack(0, k, 0)

# Time: O(k × 2^n), Space: O(2^n)
```

**Matrix Chain Multiplication**:

```python
def matrix_chain_order(dimensions):
    """
    Find minimum operations to multiply chain of matrices.
    
    dimensions[i-1] × dimensions[i] is size of matrix i
    
    dp[i][j] = min operations to multiply matrices i to j
    """
    n = len(dimensions) - 1  # Number of matrices
    
    # dp[i][j] = min cost for matrices i to j
    dp = [[0] * n for _ in range(n)]
    
    # Length of chain
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = float('inf')
            
            # Try all split points
            for k in range(i, j):
                cost = (dp[i][k] + dp[k + 1][j] + 
                       dimensions[i] * dimensions[k + 1] * dimensions[j + 1])
                dp[i][j] = min(dp[i][j], cost)
    
    return dp[0][n - 1]

# Time: O(n³), Space: O(n²)
# Example: dimensions = [10, 20, 30, 40, 30]
# Means matrices: 10×20, 20×30, 30×40, 40×30
```

**Burst Balloons**:

```python
def max_coins(nums):
    """
    Burst balloons to maximize coins.
    Bursting balloon i gives nums[i-1] × nums[i] × nums[i+1] coins.
    
    Key insight: Think about which balloon to burst LAST in range [i, j]
    
    dp[i][j] = max coins from bursting balloons in range (i, j)
    """
    # Add boundary balloons with value 1
    nums = [1] + nums + [1]
    n = len(nums)
    
    dp = [[0] * n for _ in range(n)]
    
    # Length of range
    for length in range(3, n + 1):
        for left in range(n - length + 1):
            right = left + length - 1
            
            # Try bursting each balloon last in this range
            for k in range(left + 1, right):
                coins = (dp[left][k] + dp[k][right] + 
                        nums[left] * nums[k] * nums[right])
                dp[left][right] = max(dp[left][right], coins)
    
    return dp[0][n - 1]

# Time: O(n³), Space: O(n²)
```

**Optimal Binary Search Tree**:

```python
def optimal_bst(keys, freq):
    """
    Construct BST minimizing expected search cost.
    
    freq[i] = frequency of searching for keys[i]
    
    dp[i][j] = min cost for BST with keys[i:j+1]
    """
    n = len(keys)
    dp = [[0] * n for _ in range(n)]
    
    # Precompute cumulative frequencies
    cum_freq = [0] * (n + 1)
    for i in range(n):
        cum_freq[i + 1] = cum_freq[i] + freq[i]
    
    def freq_sum(i, j):
        return cum_freq[j + 1] - cum_freq[i]
    
    # Single keys
    for i in range(n):
        dp[i][i] = freq[i]
    
    # Multiple keys
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = float('inf')
            
            # Try each key as root
            for k in range(i, j + 1):
                left_cost = dp[i][k - 1] if k > i else 0
                right_cost = dp[k + 1][j] if k < j else 0
                
                cost = left_cost + right_cost + freq_sum(i, j)
                dp[i][j] = min(dp[i][j], cost)
    
    return dp[0][n - 1]

# Time: O(n³), Space: O(n²)
```

**Egg Drop Problem**:

```python
def egg_drop(n_eggs, n_floors):
    """
    Find minimum trials to find critical floor with n eggs.
    
    dp[e][f] = min trials with e eggs and f floors
    """
    # Initialize DP table
    dp = [[float('inf')] * (n_floors + 1) for _ in range(n_eggs + 1)]
    
    # Base cases
    for e in range(n_eggs + 1):
        dp[e][0] = 0  # No floors
        dp[e][1] = 1  # One floor
    
    for f in range(n_floors + 1):
        dp[1][f] = f  # One egg: must try each floor
    
    # Fill table
    for e in range(2, n_eggs + 1):
        for f in range(2, n_floors + 1):
            # Try dropping from each floor
            for k in range(1, f + 1):
                # Egg breaks: check floors below
                # Egg doesn't break: check floors above
                worst_case = 1 + max(dp[e - 1][k - 1], dp[e][f - k])
                dp[e][f] = min(dp[e][f], worst_case)
    
    return dp[n_eggs][n_floors]

# Time: O(n_eggs × n_floors²), Space: O(n_eggs × n_floors)


def egg_drop_optimized(n_eggs, n_floors):
    """
    Optimized using binary search.
    
    Observation: dp[e][k] is monotonic in k
    """
    dp = [[0] * (n_floors + 1) for _ in range(n_eggs + 1)]
    
    for f in range(n_floors + 1):
        dp[1][f] = f
    
    for e in range(2, n_eggs + 1):
        for f in range(1, n_floors + 1):
            # Binary search for optimal floor
            left, right = 1, f
            
            while left < right:
                mid = (left + right) // 2
                
                breaks = dp[e - 1][mid - 1]
                not_breaks = dp[e][f - mid]
                
                if breaks > not_breaks:
                    right = mid
                else:
                    left = mid + 1
            
            dp[e][f] = 1 + max(dp[e - 1][left - 1], dp[e][f - left])
    
    return dp[n_eggs][n_floors]

# Time: O(n_eggs × n_floors × log(n_floors)), Space: O(n_eggs × n_floors)
```

### Complexity Analysis

| Problem Type | State Space | Time | Space | Notes |
|--------------|-------------|------|-------|-------|
| Bitmask DP | O(2^n) | O(n × 2^n) to O(n² × 2^n) | O(2^n) to O(n × 2^n) | TSP, assignment |
| Tree DP | O(n) | O(n) to O(n²) | O(n) | Recursion or iteration |
| Range DP | O(n²) | O(n³) | O(n²) | Matrix chain, burst balloons |
| Knapsack | O(n × W) | O(n × W) | O(W) optimized | W = capacity |
| Multi-dim DP | O(n^d) | O(n^d × transitions) | O(n^d) | d dimensions |

### Optimization Techniques

1. **Space optimization**: Rolling arrays, only keep needed previous states
2. **State compression**: Bitmasks for subsets
3. **Monotonic optimization**: Binary search, convex hull trick
4. **Divide and conquer optimization**: Reduce from O(n³) to O(n² log n)


### Common Questions

1. "When to use bitmask DP?"
   - **Answer**:
     - Small n (typically n ≤ 20)
     - Need to track which elements are included/excluded
     - Subset enumeration problems
     - State can be represented as binary choices
     - Examples: TSP, assignment problems, subset selection with constraints

2. "How to optimize DP from O(n³) to O(n²)?"
   - **Answer**:
     - **Monotonic queue**: For range min/max queries
     - **Convex hull trick**: For linear function optimization
     - **Divide and conquer**: For special structure
     - **Data structures**: Segment trees, sparse tables

3. "DP on trees vs DP on DAGs?"
   - **Answer**:
     - **Trees**: DFS with memoization, process children before parent
     - **DAGs**: Topological sort + DP in topo order
     - **Key difference**: Trees have unique parent, DAGs can have multiple paths
     - **Both**: Use post-order processing

4. "How to handle large state spaces?"
   - **Answer**:
     - State compression (bitmasks)
     - Only compute reachable states
     - Use memoization (top-down) instead of tabulation
     - Optimize away dimensions (rolling arrays)
     - Consider approximation algorithms if exact DP infeasible

### Problem-Solving Patterns

1. **Bitmask**:
   - Enumerate subsets
   - Track which elements used
   - Small n constraint (≤ 20)

2. **Range DP**:
   - Process by increasing length
   - Try all split points
   - `dp[i][j]` = answer for range `[i, j]`

3. **Tree DP**:
   - DFS from root
   - Combine results from children
   - Often returns tuple (option1, option2)

4. **Multi-stage**:
   - Break into stages/rounds
   - State includes stage number
   - Often has resource constraints

### Common Mistakes
- Using bitmask for large n (exponential blow-up)
- Wrong iteration order in range DP (must process smaller ranges first)
- Not considering all possible last operations/splits
- Forgetting to handle tree root specially
- Integer overflow in large state spaces

### Edge Cases
- Empty input
- Single element
- All elements same
- Maximum constraints (overflow, memory)
- Invalid states (pruning needed)

---

**[Dynamic Programming Questions Notebook](./Notebooks/DP.ipynb)**
