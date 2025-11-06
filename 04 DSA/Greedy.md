<h1 align="center"> Greedy Algorithms </h1>

**Greedy algorithms** make locally optimal choices at each step, hoping to find a global optimum. Unlike DP, greedy doesn't reconsider past decisions.

**Key Characteristics**:
- **Greedy choice property**: Local optimum leads to global optimum
- **Optimal substructure**: Optimal solution contains optimal subsolutions
- **No backtracking**: Once choice made, never reconsidered

When applicable, greedy is simpler and faster than DP (often $O(n)$ or $O(n \og n)$ vs $O(n^2)$).

**Key Insight**: Not all problems have greedy solutions. Need to prove correctness (exchange argument, stay-ahead argument).

**When to use**: 
- Problem exhibits greedy choice property
- Making locally optimal choice doesn't prevent reaching global optimum
- Often involves sorting first

## Python Implementation

**Activity Selection / Interval Scheduling**:

```python
def max_activities(start, end):
    """
    Select maximum number of non-overlapping activities.
    
    Example: start=[1,3,0,5,8,5], end=[2,4,6,7,9,9]
    Result: 4 activities
    
    Greedy: Always pick activity that finishes earliest
    """
    n = len(start)
    
    # Create list of (start, end) and sort by end time
    activities = list(zip(start, end))
    activities.sort(key=lambda x: x[1])
    
    count = 1
    last_end = activities[0][1]
    
    for i in range(1, n):
        if activities[i][0] >= last_end:
            count += 1
            last_end = activities[i][1]
    
    return count

# Time: O(n log n), Space: O(n)
# Greedy works because earliest finish gives most room for future activities
```

**Jump Game II (Minimum Jumps)**:

```python
def min_jumps_greedy(nums):
    """
    Find minimum jumps to reach last index.
    
    Example: [2,3,1,1,4] → 2 jumps
    
    Greedy: At each position, track farthest reachable
    """
    if len(nums) <= 1:
        return 0
    
    jumps = 0
    current_end = 0  # End of current jump range
    farthest = 0     # Farthest position reachable
    
    for i in range(len(nums) - 1):
        # Update farthest reachable
        farthest = max(farthest, i + nums[i])
        
        # If reached end of current jump range
        if i == current_end:
            jumps += 1
            current_end = farthest
            
            # Early termination
            if current_end >= len(nums) - 1:
                break
    
    return jumps

# Time: O(n), Space: O(1)
# Much faster than DP O(n²) approach
```

**Gas Station**:

```python
def can_complete_circuit(gas, cost):
    """
    Find starting gas station to complete circular route.
    
    gas[i] = gas available at station i
    cost[i] = gas needed to travel from station i to i+1
    
    Return starting station index, or -1 if impossible
    
    Greedy insight: If total gas >= total cost, solution exists
    Start from station where we first run out of gas
    """
    total_gas = total_cost = 0
    current_gas = 0
    start = 0
    
    for i in range(len(gas)):
        total_gas += gas[i]
        total_cost += cost[i]
        current_gas += gas[i] - cost[i]
        
        # If we can't reach next station
        if current_gas < 0:
            # Start from next station
            start = i + 1
            current_gas = 0
    
    # Check if solution exists
    return start if total_gas >= total_cost else -1

# Time: O(n), Space: O(1)
# Greedy: If we fail at position i, starting anywhere before i also fails
```

**Partition Labels**:

```python
def partition_labels(s):
    """
    Partition string so each letter appears in at most one part.
    Maximize number of parts.
    
    Example: "ababcbacadefegdehijhklij"
    Result: [9,7,8] → "ababcbaca", "defegde", "hijhklij"
    
    Greedy: Extend partition until last occurrence of all letters seen
    """
    # Find last occurrence of each character
    last_occurrence = {char: i for i, char in enumerate(s)}
    
    partitions = []
    start = 0
    end = 0
    
    for i, char in enumerate(s):
        # Extend partition to include last occurrence of this char
        end = max(end, last_occurrence[char])
        
        # If we've reached the end of current partition
        if i == end:
            partitions.append(end - start + 1)
            start = i + 1
    
    return partitions

# Time: O(n), Space: O(1) - alphabet size constant
```

**Queue Reconstruction by Height**:

```python
def reconstruct_queue(people):
    """
    Reconstruct queue based on height and count of taller people.
    
    people[i] = [height_i, k_i] where k_i = number of people in front
    with height >= height_i
    
    Example: [[7,0],[4,4],[7,1],[5,0],[6,1],[5,2]]
    Result: [[5,0],[7,0],[5,2],[6,1],[4,4],[7,1]]
    
    Greedy: Sort by height desc, insert each person at their k index
    """
    # Sort by height descending, then by k ascending
    people.sort(key=lambda x: (-x[0], x[1]))
    
    result = []
    
    for person in people:
        height, k = person
        # Insert at position k
        result.insert(k, person)
    
    return result

# Time: O(n²) due to insertions, Space: O(n)
# Greedy works because taller people don't affect shorter people's positions
```

**Candy Distribution**:

```python
def candy(ratings):
    """
    Distribute candies to children based on ratings.
    
    Rules:
    - Each child must get at least 1 candy
    - Children with higher rating get more candy than neighbors
    
    Minimize total candies.
    
    Example: [1,0,2] → 5 candies [2,1,2]
    
    Greedy: Two passes - left-to-right and right-to-left
    """
    n = len(ratings)
    candies = [1] * n
    
    # Left-to-right: ensure right neighbor has more if rating higher
    for i in range(1, n):
        if ratings[i] > ratings[i - 1]:
            candies[i] = candies[i - 1] + 1
    
    # Right-to-left: ensure left neighbor has more if rating higher
    for i in range(n - 2, -1, -1):
        if ratings[i] > ratings[i + 1]:
            candies[i] = max(candies[i], candies[i + 1] + 1)
    
    return sum(candies)

# Time: O(n), Space: O(n)


def candy_optimized(ratings):
    """Space-optimized to O(1)."""
    if not ratings:
        return 0
    
    n = len(ratings)
    total = 1  # First child gets 1 candy
    up = down = peak = 1
    
    for i in range(1, n):
        if ratings[i] > ratings[i - 1]:
            # Ascending
            up += 1
            peak = up
            down = 1
            total += up
        elif ratings[i] == ratings[i - 1]:
            # Equal
            up = down = peak = 1
            total += 1
        else:
            # Descending
            down += 1
            up = 1
            
            # If descent longer than peak, need to add candy to peak
            total += down + (1 if down > peak else 0)
    
    return total

# Time: O(n), Space: O(1)
```

**Task Scheduler**:

```python
def least_interval(tasks, n):
    """
    Schedule tasks with cooling period n between same tasks.
    
    Example: tasks=['A','A','A','B','B','B'], n=2
    Result: 8 → "A B _ A B _ A B" (or any valid arrangement)
    
    Greedy: Schedule most frequent tasks first with gaps
    """
    from collections import Counter
    
    # Count task frequencies
    freq = Counter(tasks)
    max_freq = max(freq.values())
    
    # Count how many tasks have max frequency
    max_count = sum(1 for f in freq.values() if f == max_freq)
    
    # Calculate minimum intervals needed
    # (max_freq - 1) chunks of (n + 1) length, plus max_count tasks at end
    min_intervals = (max_freq - 1) * (n + 1) + max_count
    
    # Can't be less than total number of tasks
    return max(min_intervals, len(tasks))

# Time: O(n), Space: O(1) - at most 26 unique tasks
# Greedy works by filling gaps optimally


def task_scheduler_simulation(tasks, n):
    """Alternative: simulate scheduling with max heap."""
    from collections import Counter
    import heapq
    
    freq = Counter(tasks)
    # Max heap (negate frequencies)
    max_heap = [-f for f in freq.values()]
    heapq.heapify(max_heap)
    
    time = 0
    
    while max_heap:
        cycle = []
        
        # Try to schedule n+1 different tasks
        for _ in range(n + 1):
            if max_heap:
                freq = heapq.heappop(max_heap)
                if freq + 1 < 0:  # Still has remaining
                    cycle.append(freq + 1)
        
        # Add back tasks with remaining frequency
        for freq in cycle:
            heapq.heappush(max_heap, freq)
        
        # Add time for this cycle
        time += (n + 1) if max_heap else len(cycle)
    
    return time

# Time: O(n log k), Space: O(k) where k = unique tasks
```

**Remove K Digits**:

```python
def remove_k_digits(num, k):
    """
    Remove k digits to make smallest possible number.
    
    Example: num="1432219", k=3 → "1219"
    
    Greedy: Use stack, remove digits that are greater than next digit
    """
    stack = []
    
    for digit in num:
        # Remove larger digits from stack while we can
        while stack and k > 0 and stack[-1] > digit:
            stack.pop()
            k -= 1
        
        stack.append(digit)
    
    # If still need to remove digits, remove from end
    if k > 0:
        stack = stack[:-k]
    
    # Remove leading zeros and convert to string
    result = ''.join(stack).lstrip('0')
    
    return result if result else '0'

# Time: O(n), Space: O(n)
# Greedy: Always remove first larger digit we encounter
```

**Minimum Number of Arrows**:

```python
def find_min_arrows(points):
    """
    Find minimum arrows to burst all balloons.
    
    points[i] = [x_start, x_end] represents balloon covering [x_start, x_end]
    Arrow at x bursts all balloons where x_start <= x <= x_end
    
    Example: [[10,16],[2,8],[1,6],[7,12]] → 2 arrows
    
    Greedy: Sort by end position, shoot arrow at earliest end
    """
    if not points:
        return 0
    
    # Sort by end position
    points.sort(key=lambda x: x[1])
    
    arrows = 1
    arrow_pos = points[0][1]
    
    for start, end in points[1:]:
        # If balloon starts after current arrow position
        if start > arrow_pos:
            arrows += 1
            arrow_pos = end  # Shoot new arrow at this balloon's end
    
    return arrows

# Time: O(n log n), Space: O(1)
# Similar to interval scheduling
```

**Lemonade Change**:

```python
def lemonade_change(bills):
    """
    Check if can provide correct change for all customers.
    
    Lemonade costs $5, customers pay with $5, $10, or $20 bills.
    
    Example: [5,5,5,10,20] → True
    Example: [5,5,10,10,20] → False
    
    Greedy: Always use larger bills for change when possible
    """
    five = ten = 0
    
    for bill in bills:
        if bill == 5:
            five += 1
        elif bill == 10:
            if five == 0:
                return False
            five -= 1
            ten += 1
        else:  # bill == 20
            # Prefer giving 1 ten + 1 five (saves fives for later)
            if ten > 0 and five > 0:
                ten -= 1
                five -= 1
            elif five >= 3:
                five -= 3
            else:
                return False
    
    return True

# Time: O(n), Space: O(1)
```

**Boats to Save People**:

```python
def num_rescue_boats(people, limit):
    """
    Minimum boats to rescue people (max 2 per boat, weight limit).
    
    Example: people=[3,2,2,1], limit=3 → 3 boats
    
    Greedy: Pair heaviest with lightest if possible
    """
    people.sort()
    
    boats = 0
    left = 0
    right = len(people) - 1
    
    while left <= right:
        # Try to pair heaviest with lightest
        if people[left] + people[right] <= limit:
            left += 1  # Both get on boat
        
        right -= 1  # Heaviest gets on boat
        boats += 1
    
    return boats

# Time: O(n log n), Space: O(1)
```

**Largest Number**:

```python
def largest_number(nums):
    """
    Arrange numbers to form largest number.
    
    Example: [3,30,34,5,9] → "9534330"
    
    Greedy: Custom comparator - compare concatenations
    """
    from functools import cmp_to_key
    
    # Convert to strings
    nums = list(map(str, nums))
    
    # Custom comparator: x + y vs y + x
    def compare(x, y):
        if x + y > y + x:
            return -1  # x should come before y
        elif x + y < y + x:
            return 1
        else:
            return 0
    
    nums.sort(key=cmp_to_key(compare))
    
    # Handle all zeros case
    result = ''.join(nums)
    return '0' if result[0] == '0' else result

# Time: O(n log n), Space: O(n)
# Greedy works because comparison is transitive
```

**Best Time to Buy and Sell Stock II**:

```python
def max_profit_unlimited_transactions(prices):
    """
    Maximum profit with unlimited buy/sell transactions.
    
    Can buy and sell on same day.
    
    Example: [7,1,5,3,6,4] → 7 (buy 1 sell 5, buy 3 sell 6)
    
    Greedy: Take every positive difference
    """
    profit = 0
    
    for i in range(1, len(prices)):
        if prices[i] > prices[i - 1]:
            profit += prices[i] - prices[i - 1]
    
    return profit

# Time: O(n), Space: O(1)
# Greedy: Capture every upward price movement
```

**Wiggle Subsequence**:

```python
def wiggle_max_length(nums):
    """
    Find longest wiggle subsequence length.
    
    Wiggle: differences alternate between positive and negative
    Example: [1,7,4,9,2,5] → 6 (entire sequence)
    Example: [1,17,5,10,13,15,10,5,16,8] → 7
    
    Greedy: Track peaks and valleys
    """
    if len(nums) < 2:
        return len(nums)
    
    up = down = 1  # Length ending in up/down
    
    for i in range(1, len(nums)):
        if nums[i] > nums[i - 1]:
            up = down + 1
        elif nums[i] < nums[i - 1]:
            down = up + 1
    
    return max(up, down)

# Time: O(n), Space: O(1)


def wiggle_max_length_explicit(nums):
    """Alternative: track last difference."""
    if len(nums) < 2:
        return len(nums)
    
    length = 1
    prev_diff = 0
    
    for i in range(1, len(nums)):
        diff = nums[i] - nums[i - 1]
        
        # If direction changed (or first difference)
        if (diff > 0 and prev_diff <= 0) or (diff < 0 and prev_diff >= 0):
            length += 1
            prev_diff = diff
    
    return length

# Time: O(n), Space: O(1)
```

## Complexity Analysis


| Problem | Sorting | Time | Space | Key Greedy Choice |
|---------|---------|------|-------|-------------------|
| Activity Selection | Yes | O(n log n) | O(1) | Earliest finish time |
| Jump Game | No | O(n) | O(1) | Maximize reach |
| Gas Station | No | O(n) | O(1) | Start after failure |
| Candy | No | O(n) | O(n) | Two-pass comparison |
| Task Scheduler | No | O(n) | O(1) | Most frequent first |
| Intervals | Yes | O(n log n) | O(1) | Earliest end |

**Greedy vs DP**:
- **Greedy**: O(n) or O(n log n), no backtracking
- **DP**: O(n²) or worse, considers all options
- **Trade-off**: Greedy faster but not always correct

## Common Questions

1. "How to prove a greedy algorithm is correct?"
   - **Answer**:
     - **Exchange argument**: Show that swapping greedy choice with any other doesn't improve solution
     - **Stay-ahead argument**: Show greedy stays at least as good as any other solution at each step
     - **Structural argument**: Prove optimal solution has greedy choice property
     - **Example**: Activity selection - choosing earliest finish leaves most room for future

2. "When does greedy fail?"
   - **Answer**:
     - **Shortest path with negative weights**: Dijkstra fails, need Bellman-Ford
     - **0/1 Knapsack**: Greedy by value/weight ratio fails, need DP
     - **Longest path in DAG**: Greedy doesn't work, need DP or topological sort
     - **Counterexample is key**: One counterexample proves greedy wrong

3. "Greedy vs DP - how to decide?"
   - **Answer**:
     - **Try greedy first**: Simpler if it works
     - **Check for greedy choice property**: Does local optimum lead to global?
     - **Look for counterexample**: If found, need DP
     - **Pattern recognition**: Intervals, scheduling often greedy; counting ways often DP

4. "Common greedy patterns?"
   - **Answer**:
     - **Sorting**: Many greedy algorithms start with sorting
     - **Two pointers**: After sorting, often use two pointers
     - **Priority queue**: Greedily select best option
     - **Stack**: Monotonic stack for maintaining order

## Problem-Solving Template

```python
def greedy_template(input_data):
    """
    Step 1: Sort or preprocess data
    - Often need specific ordering
    
    Step 2: Initialize result
    
    Step 3: Iterate and make greedy choice
    - Select locally optimal option
    - Update state
    
    Step 4: Return result
    """
    # Sort if needed
    sorted_data = sorted(input_data, key=lambda x: greedy_criterion(x))
    
    result = initialize_result()
    
    for item in sorted_data:
        if satisfies_constraint(item, result):
            result = update_with_greedy_choice(result, item)
    
    return result
```

## Common Mistakes
- Assuming greedy works without proof
- Wrong sorting criterion
- Not considering all constraints
- Off-by-one errors in comparisons
- Not handling edge cases (empty input, single element)

## Edge Cases
- Empty input
- Single element
- All elements same
- Already optimal input
- Worst-case ordering

---

**[Greedy Questions Notebook](./Notebooks/Greedy.ipynb)**
