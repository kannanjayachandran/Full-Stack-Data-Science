<h1 align="center"> Divide & Conquer </h1>

**Divide and Conquer** breaks problem into smaller independent subproblems, solve recursively, then combine results.

**Three Steps**:
1. **Divide**: Split problem into smaller subproblems
2. **Conquer**: Solve subproblems recursively
3. **Combine**: Merge solutions to get final answer

Divide and Conquer algorithms reduce time complexity from $O(n^2)\;\rightarrow\;O(n \log n)$

**Key Insight**: Unlike backtracking (tries all possibilities), divide and conquer solves independent subproblems that don't interfere with each other.

**Difference from DP**: Subproblems are **independent** (no overlap), while DP has **overlapping** subproblems.

## Python Implementation

**Merge Sort**:

```python
def merge_sort(arr):
    """
    Sort array using divide and conquer.
    
    Divide: Split array in half
    Conquer: Sort each half
    Combine: Merge sorted halves
    """
    # Base case
    if len(arr) <= 1:
        return arr
    
    # Divide
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    # Combine
    return merge(left, right)


def merge(left, right):
    """Merge two sorted arrays."""
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    # Add remaining elements
    result.extend(left[i:])
    result.extend(right[j:])
    
    return result

# Time: O(n log n), Space: O(n)
# Recurrence: T(n) = 2T(n/2) + O(n)


def merge_sort_inplace(arr, left, right):
    """
    In-place merge sort (more space efficient).
    
    Note: Still needs O(n) auxiliary space for merging,
    but doesn't create new arrays during recursion.
    """
    if left >= right:
        return
    
    mid = (left + right) // 2
    
    # Divide and conquer
    merge_sort_inplace(arr, left, mid)
    merge_sort_inplace(arr, mid + 1, right)
    
    # Combine
    merge_inplace(arr, left, mid, right)


def merge_inplace(arr, left, mid, right):
    """Merge two sorted subarrays in place."""
    # Create temporary arrays
    left_arr = arr[left:mid + 1]
    right_arr = arr[mid + 1:right + 1]
    
    i = j = 0
    k = left
    
    while i < len(left_arr) and j < len(right_arr):
        if left_arr[i] <= right_arr[j]:
            arr[k] = left_arr[i]
            i += 1
        else:
            arr[k] = right_arr[j]
            j += 1
        k += 1
    
    # Copy remaining
    while i < len(left_arr):
        arr[k] = left_arr[i]
        i += 1
        k += 1
    
    while j < len(right_arr):
        arr[k] = right_arr[j]
        j += 1
        k += 1

# Usage: merge_sort_inplace(arr, 0, len(arr) - 1)
```

**Quick Sort**:

```python
def quick_sort(arr):
    """
    Sort using quicksort (divide and conquer).
    
    Divide: Partition around pivot
    Conquer: Sort partitions
    Combine: No work needed (in-place)
    """
    if len(arr) <= 1:
        return arr
    
    # Choose pivot (here: last element)
    pivot = arr[-1]
    
    # Partition
    left = [x for x in arr[:-1] if x <= pivot]
    right = [x for x in arr[:-1] if x > pivot]
    
    # Recursively sort and combine
    return quick_sort(left) + [pivot] + quick_sort(right)

# Time: O(n log n) average, O(n²) worst
# Space: O(log n) average (recursion)


def quick_sort_inplace(arr, low, high):
    """In-place quicksort (more efficient)."""
    if low < high:
        # Partition and get pivot index
        pivot_idx = partition(arr, low, high)
        
        # Sort left and right of pivot
        quick_sort_inplace(arr, low, pivot_idx - 1)
        quick_sort_inplace(arr, pivot_idx + 1, high)


def partition(arr, low, high):
    """
    Partition array around pivot (last element).
    
    Returns index of pivot after partitioning.
    """
    pivot = arr[high]
    i = low - 1  # Index of smaller element
    
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    
    # Place pivot in correct position
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

# Usage: quick_sort_inplace(arr, 0, len(arr) - 1)
# Average: O(n log n), Worst: O(n²), Space: O(log n)
```

**Quick Select (Find k-th Smallest)**:

```python
def quick_select(arr, k):
    """
    Find k-th smallest element (0-indexed).
    
    Similar to quicksort but only recurse on one partition.
    Average O(n), worst O(n²).
    """
    if len(arr) == 1:
        return arr[0]
    
    pivot = arr[len(arr) // 2]
    
    # Partition
    left = [x for x in arr if x < pivot]
    mid = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    # Determine which partition k is in
    if k < len(left):
        return quick_select(left, k)
    elif k < len(left) + len(mid):
        return mid[0]
    else:
        return quick_select(right, k - len(left) - len(mid))

# Time: O(n) average, O(n²) worst
# Space: O(log n) average


def quick_select_inplace(arr, left, right, k):
    """In-place version of quick select."""
    if left == right:
        return arr[left]
    
    # Partition
    pivot_idx = partition(arr, left, right)
    
    # Check which partition k is in
    if k == pivot_idx:
        return arr[k]
    elif k < pivot_idx:
        return quick_select_inplace(arr, left, pivot_idx - 1, k)
    else:
        return quick_select_inplace(arr, pivot_idx + 1, right, k)

# Usage: quick_select_inplace(arr, 0, len(arr) - 1, k)
```

**Binary Tree Divide and Conquer**:

```python
def max_path_sum(root):
    """
    Find maximum path sum in binary tree.
    
    Divide: Compute max path through left and right subtrees
    Conquer: Recursively solve for subtrees
    Combine: Max of (left + node + right), (left + node), (right + node)
    """
    max_sum = float('-inf')
    
    def max_gain(node):
        """Return max path sum starting from node going down."""
        nonlocal max_sum
        
        if not node:
            return 0
        
        # Divide: get max gain from children (ignore negative)
        left_gain = max(max_gain(node.left), 0)
        right_gain = max(max_gain(node.right), 0)
        
        # Combine: path through current node
        path_sum = node.val + left_gain + right_gain
        
        # Update global maximum
        max_sum = max(max_sum, path_sum)
        
        # Return max gain continuing upward (can't use both children)
        return node.val + max(left_gain, right_gain)
    
    max_gain(root)
    return max_sum

# Time: O(n), Space: O(h)


def diameter_of_binary_tree(root):
    """
    Find longest path between any two nodes.
    
    Divide and conquer on tree structure.
    """
    diameter = 0
    
    def depth(node):
        """Return depth and update diameter."""
        nonlocal diameter
        
        if not node:
            return 0
        
        # Conquer
        left_depth = depth(node.left)
        right_depth = depth(node.right)
        
        # Combine: path through this node
        diameter = max(diameter, left_depth + right_depth)
        
        return 1 + max(left_depth, right_depth)
    
    depth(root)
    return diameter

# Time: O(n), Space: O(h)
```

**Closest Pair of Points**:

```python
def closest_pair_of_points(points):
    """
    Find closest pair of points in 2D plane.
    
    Brute force: O(n²)
    Divide and conquer: O(n log n)
    """
    import math
    
    def distance(p1, p2):
        """Euclidean distance."""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def brute_force(points):
        """For small arrays, use brute force."""
        min_dist = float('inf')
        n = len(points)
        
        for i in range(n):
            for j in range(i + 1, n):
                min_dist = min(min_dist, distance(points[i], points[j]))
        
        return min_dist
    
    def strip_closest(strip, d):
        """Find closest in strip of width 2d."""
        min_dist = d
        strip.sort(key=lambda p: p[1])  # Sort by y
        
        for i in range(len(strip)):
            j = i + 1
            # Only check points within distance d
            while j < len(strip) and (strip[j][1] - strip[i][1]) < min_dist:
                min_dist = min(min_dist, distance(strip[i], strip[j]))
                j += 1
        
        return min_dist
    
    def closest_pair_recursive(Px, Py):
        """
        Px: points sorted by x
        Py: points sorted by y
        """
        n = len(Px)
        
        # Base case: use brute force for small n
        if n <= 3:
            return brute_force(Px)
        
        # Divide: split at median x
        mid = n // 2
        midpoint = Px[mid]
        
        Pyl = [p for p in Py if p[0] <= midpoint[0]]
        Pyr = [p for p in Py if p[0] > midpoint[0]]
        
        # Conquer: find minimum in each half
        dl = closest_pair_recursive(Px[:mid], Pyl)
        dr = closest_pair_recursive(Px[mid:], Pyr)
        
        # Find minimum of two
        d = min(dl, dr)
        
        # Combine: check points in strip
        strip = [p for p in Py if abs(p[0] - midpoint[0]) < d]
        
        return min(d, strip_closest(strip, d))
    
    # Sort points
    Px = sorted(points, key=lambda p: p[0])
    Py = sorted(points, key=lambda p: p[1])
    
    return closest_pair_recursive(Px, Py)

# Time: O(n log n), Space: O(n)
```

**Counting Inversions**:

```python
def count_inversions(arr):
    """
    Count inversions: pairs (i, j) where i < j but arr[i] > arr[j].
    
    Example: [2, 4, 1, 3, 5] has 3 inversions: (2,1), (4,1), (4,3)
    
    Divide and conquer: Count while merging (like merge sort).
    """
    def merge_count(arr, temp, left, mid, right):
        """Merge and count inversions."""
        i = left      # Left subarray
        j = mid + 1   # Right subarray
        k = left      # Merged array
        inv_count = 0
        
        while i <= mid and j <= right:
            if arr[i] <= arr[j]:
                temp[k] = arr[i]
                i += 1
            else:
                # arr[i] > arr[j], so all remaining elements in left
                # form inversions with arr[j]
                temp[k] = arr[j]
                inv_count += (mid - i + 1)
                j += 1
            k += 1
        
        # Copy remaining
        while i <= mid:
            temp[k] = arr[i]
            i += 1
            k += 1
        
        while j <= right:
            temp[k] = arr[j]
            j += 1
            k += 1
        
        # Copy back to original array
        for i in range(left, right + 1):
            arr[i] = temp[i]
        
        return inv_count
    
    def merge_sort_count(arr, temp, left, right):
        """Merge sort with inversion counting."""
        inv_count = 0
        
        if left < right:
            mid = (left + right) // 2
            
            # Count inversions in left and right
            inv_count += merge_sort_count(arr, temp, left, mid)
            inv_count += merge_sort_count(arr, temp, mid + 1, right)
            
            # Count split inversions
            inv_count += merge_count(arr, temp, left, mid, right)
        
        return inv_count
    
    n = len(arr)
    temp = [0] * n
    return merge_sort_count(arr, temp, 0, n - 1)

# Time: O(n log n), Space: O(n)
```

**Maximum Subarray (Kadane's vs Divide and Conquer)**:

```python
def max_subarray_divide_conquer(arr, left, right):
    """
    Find maximum subarray sum using divide and conquer.
    
    Note: Kadane's algorithm is O(n) and simpler.
    This is educational - shows D&C approach.
    """
    # Base case
    if left == right:
        return arr[left]
    
    # Divide
    mid = (left + right) // 2
    
    # Conquer: max in left and right halves
    left_max = max_subarray_divide_conquer(arr, left, mid)
    right_max = max_subarray_divide_conquer(arr, mid + 1, right)
    
    # Combine: max crossing midpoint
    # Find max sum ending at mid
    left_sum = float('-inf')
    curr_sum = 0
    for i in range(mid, left - 1, -1):
        curr_sum += arr[i]
        left_sum = max(left_sum, curr_sum)
    
    # Find max sum starting at mid+1
    right_sum = float('-inf')
    curr_sum = 0
    for i in range(mid + 1, right + 1):
        curr_sum += arr[i]
        right_sum = max(right_sum, curr_sum)
    
    cross_sum = left_sum + right_sum
    
    # Return maximum of three
    return max(left_max, right_max, cross_sum)

# Time: O(n log n), Space: O(log n)
# Kadane's is better: O(n), O(1)


def max_subarray_kadane(arr):
    """
    Maximum subarray sum using Kadane's algorithm.
    
    Simpler and more efficient than D&C for this problem.
    """
    max_sum = float('-inf')
    curr_sum = 0
    
    for num in arr:
        curr_sum = max(num, curr_sum + num)
        max_sum = max(max_sum, curr_sum)
    
    return max_sum

# Time: O(n), Space: O(1)
```

## Complexity Analysis

**Master Theorem**: For recurrence T(n) = aT(n/b) + f(n):

| Case | Condition | Complexity |
|------|-----------|------------|
| 1 | f(n) = O(n^c) where c < log_b(a) | Θ(n^log_b(a)) |
| 2 | f(n) = Θ(n^c) where c = log_b(a) | Θ(n^c log n) |
| 3 | f(n) = Ω(n^c) where c > log_b(a) | Θ(f(n)) |

**Common Algorithms**:

| Algorithm | Recurrence | Time | Space | Notes |
|-----------|------------|------|-------|-------|
| Merge Sort | T(n) = 2T(n/2) + O(n) | O(n log n) | O(n) | Stable, predictable |
| Quick Sort | T(n) = 2T(n/2) + O(n) | O(n log n) avg | O(log n) | In-place, unstable |
| Quick Select | T(n) = T(n/2) + O(n) | O(n) avg | O(log n) | Find k-th element |
| Binary Search | T(n) = T(n/2) + O(1) | O(log n) | O(1) | On sorted array |
| Strassen Matrix Mult | T(n) = 7T(n/2) + O(n²) | O(n^2.81) | O(n²) | Better than O(n³) |

## Common Questions

1. "Explain divide and conquer vs dynamic programming"
   - **Answer**:
     - **D&C**: Independent subproblems, no overlap
     - **DP**: Overlapping subproblems, cache results
     - **D&C example**: Merge sort (left and right halves independent)
     - **DP example**: Fibonacci (f(n-1) and f(n-2) share subproblems)
     - **Conversion**: D&C → DP when subproblems overlap

2. "Why is merge sort O(n log n)?"
   - **Answer**:
     - **Height of recursion tree**: log n (divide by 2 each level)
     - **Work per level**: O(n) (merging all elements)
     - **Total**: O(n) × log n levels = O(n log n)
     - **Master theorem**: T(n) = 2T(n/2) + O(n) → Case 2 → Θ(n log n)

3. "When to use merge sort vs quick sort?"
   - **Answer**:
     - **Merge sort**: Guaranteed O(n log n), stable, good for linked lists, external sorting
     - **Quick sort**: O(n log n) average, in-place, cache-friendly, faster in practice
     - **Use merge**: Need stability, worst-case guarantee, sorting linked lists
     - **Use quick**: Need in-place, average case sufficient, random data

4. "Explain quick select and when to use it"
   - **Answer**:
     - **Quick select**: Find k-th smallest in O(n) average
     - **Better than sorting**: O(n) vs O(n log n)
     - **Use cases**: Median finding, percentiles, top-k selection
     - **Optimization**: Randomize pivot to avoid O(n²) worst case

## Problem-Solving Patterns

1. **Sorting**: Merge sort, quick sort
2. **Selection**: Quick select for k-th element
3. **Search**: Binary search and variants
4. **Tree problems**: Divide on left/right subtrees
5. **Array problems**: Divide at midpoint, solve halves, combine

## Common Mistakes
- Not handling base case correctly
- Incorrect combination logic
- Creating too many temporary arrays (space inefficiency)
- Not considering worst-case (e.g., quicksort on sorted data)
- Forgetting to handle odd/even array lengths

## Edge Cases
- Empty array
- Single element
- All elements equal
- Already sorted (best/worst case depending on algorithm)
- Reverse sorted

---

**[Divide & Conquer Questions Notebook](./Notebooks/Divide&Conquer.ipynb)**
