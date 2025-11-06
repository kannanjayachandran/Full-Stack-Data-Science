<h1 align="center"> Searching & Sorting </h1>

**Searching** is finding an element in a collection.

**Sorting** is arranging elements in a specific order (ascending/descending).

**Key Insight**: The "best" algorithm depends on:

- Data characteristics (size, distribution, partially sorted?)
- Memory constraints (in-place vs extra space)
- Stability requirement (preserve relative order of equal elements)
- Online vs batch (streaming data vs all at once)

**Sorting Complexity Lower Bound**: Comparison-based sorting cannot be faster than $O(n \log n)$ in worst case

## Python Implementation

#### Searching

**Linear Search**:

```python
def linear_search(arr, target):
    """
    Basic search - check each element sequentially.
    
    Use when: Unsorted data, small arrays, searching once
    """
    for i, val in enumerate(arr):
        if val == target:
            return i
    return -1

# Time: O(n), Space: O(1)
# Best for: Small or unsorted arrays


def linear_search_all(arr, target):
    """Find all occurrences of target."""
    return [i for i, val in enumerate(arr) if val == target]

# Time: O(n), Space: O(k) where k = number of matches
```

**Binary Search** (covered earlier):

```python
def binary_search(arr, target):
    """
    Search in sorted array by repeatedly halving search space.
    
    Requires: Sorted array
    """
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1

# Time: O(log n), Space: O(1)
```

**Interpolation Search**:

```python
def interpolation_search(arr, target):
    """
    Search using interpolation for uniformly distributed data.
    
    Better than binary search for uniformly distributed data.
    Estimates position based on value distribution.
    """
    left, right = 0, len(arr) - 1
    
    while left <= right and target >= arr[left] and target <= arr[right]:
        # Avoid division by zero
        if left == right:
            if arr[left] == target:
                return left
            return -1
        
        # Interpolation formula
        pos = left + ((target - arr[left]) * (right - left) // 
                     (arr[right] - arr[left]))
        
        if arr[pos] == target:
            return pos
        elif arr[pos] < target:
            left = pos + 1
        else:
            right = pos - 1
    
    return -1

# Time: O(log log n) for uniform distribution, O(n) worst case
# Space: O(1)
```

**Jump Search**:

```python
def jump_search(arr, target):
    """
    Jump ahead by fixed steps, then linear search in block.
    
    Optimal jump size: √n
    Better than linear for large sorted arrays when binary search overhead high.
    """
    import math
    
    n = len(arr)
    step = int(math.sqrt(n))
    prev = 0
    
    # Jump to find block containing target
    while arr[min(step, n) - 1] < target:
        prev = step
        step += int(math.sqrt(n))
        if prev >= n:
            return -1
    
    # Linear search in block
    while arr[prev] < target:
        prev += 1
        if prev == min(step, n):
            return -1
    
    if arr[prev] == target:
        return prev
    
    return -1

# Time: O(√n), Space: O(1)
# Use case: When backward jump is costly (tape drives, etc.)
```

**Exponential Search**:

```python
def exponential_search(arr, target):
    """
    Double search range until target range found, then binary search.
    
    Useful for unbounded/infinite arrays or when target is near beginning.
    """
    if arr[0] == target:
        return 0
    
    # Find range for binary search by doubling
    i = 1
    while i < len(arr) and arr[i] <= target:
        i *= 2
    
    # Binary search in range [i//2, min(i, n-1)]
    return binary_search_range(arr, target, i // 2, min(i, len(arr) - 1))


def binary_search_range(arr, target, left, right):
    """Binary search in given range."""
    while left <= right:
        mid = left + (right - left) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1

# Time: O(log n), Space: O(1)
# Better than binary search when target is near start
```

#### Sorting

**Bubble Sort**:

```python
def bubble_sort(arr):
    """
    Repeatedly swap adjacent elements if in wrong order.
    
    Simple but inefficient. Good for: teaching, nearly sorted data.
    """
    n = len(arr)
    
    for i in range(n):
        swapped = False
        
        # Last i elements already in place
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        
        # If no swaps, array is sorted
        if not swapped:
            break
    
    return arr

# Time: O(n²), Space: O(1)
# Best case: O(n) when already sorted
# Stable: Yes
# Use case: Small arrays, nearly sorted, educational
```

**Selection Sort**:

```python
def selection_sort(arr):
    """
    Repeatedly select minimum element and place at beginning.
    
    Simple, fewer swaps than bubble sort, but still O(n²).
    """
    n = len(arr)
    
    for i in range(n):
        # Find minimum in remaining array
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        
        # Swap minimum with current position
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    
    return arr

# Time: O(n²), Space: O(1)
# Always O(n²), even if sorted
# Stable: No (can be made stable with modifications)
# Use case: Small arrays, memory writes are expensive
```

**Insertion Sort**:

```python
def insertion_sort(arr):
    """
    Build sorted array one element at a time by inserting into correct position.
    
    Efficient for small arrays and nearly sorted data.
    """
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        
        # Move elements greater than key one position ahead
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        
        arr[j + 1] = key
    
    return arr

# Time: O(n²) worst/avg, O(n) best (sorted)
# Space: O(1)
# Stable: Yes
# Use case: Small arrays, nearly sorted, online sorting
```

**Merge Sort** (covered in D&C):

```python
def merge_sort(arr):
    """
    Divide array, sort halves, merge sorted halves.
    
    Guaranteed O(n log n), stable, good for linked lists.
    """
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
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
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result

# Time: O(n log n) always, Space: O(n)
# Stable: Yes
# Use case: Need stability, predictable performance, external sorting
```

**Quick Sort** (covered in D&C):

```python
def quick_sort(arr, low=0, high=None):
    """
    Partition around pivot, sort partitions.
    
    Fast in practice, in-place, but O(n²) worst case.
    """
    if high is None:
        high = len(arr) - 1
    
    if low < high:
        pivot_idx = partition(arr, low, high)
        quick_sort(arr, low, pivot_idx - 1)
        quick_sort(arr, pivot_idx + 1, high)
    
    return arr


def partition(arr, low, high):
    """Partition array around pivot (last element)."""
    pivot = arr[high]
    i = low - 1
    
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

# Time: O(n log n) avg, O(n²) worst, Space: O(log n)
# Stable: No
# Use case: General purpose, large arrays, when average case sufficient
```

**Heap Sort**:

```python
def heap_sort(arr):
    """
    Build max heap, repeatedly extract maximum.
    
    In-place, O(n log n) guaranteed, but not stable.
    """
    n = len(arr)
    
    # Build max heap
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)
    
    # Extract elements from heap
    for i in range(n - 1, 0, -1):
        # Move current root to end
        arr[0], arr[i] = arr[i], arr[0]
        
        # Heapify reduced heap
        heapify(arr, i, 0)
    
    return arr


def heapify(arr, n, i):
    """Maintain max heap property."""
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2
    
    if left < n and arr[left] > arr[largest]:
        largest = left
    
    if right < n and arr[right] > arr[largest]:
        largest = right
    
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)

# Time: O(n log n) always, Space: O(1)
# Stable: No
# Use case: When O(n log n) guaranteed and O(1) space needed
```

**Counting Sort**:

```python
def counting_sort(arr):
    """
    Count occurrences, place in sorted order.
    
    Linear time for small integer ranges.
    Not comparison-based!
    """
    if not arr:
        return arr
    
    # Find range
    min_val, max_val = min(arr), max(arr)
    range_size = max_val - min_val + 1
    
    # Count occurrences
    count = [0] * range_size
    for num in arr:
        count[num - min_val] += 1
    
    # Cumulative count (for stable sort)
    for i in range(1, range_size):
        count[i] += count[i - 1]
    
    # Build output array (stable)
    output = [0] * len(arr)
    for num in reversed(arr):  # Reverse for stability
        output[count[num - min_val] - 1] = num
        count[num - min_val] -= 1
    
    return output

# Time: O(n + k) where k = range, Space: O(n + k)
# Stable: Yes
# Use case: Small integer ranges, need linear time
```

**Radix Sort**:

```python
def radix_sort(arr):
    """
    Sort by processing digits from least to most significant.
    
    Uses counting sort as subroutine.
    Linear time for bounded integers.
    """
    if not arr:
        return arr
    
    # Find maximum to determine number of digits
    max_val = max(arr)
    
    # Sort by each digit
    exp = 1
    while max_val // exp > 0:
        counting_sort_by_digit(arr, exp)
        exp *= 10
    
    return arr


def counting_sort_by_digit(arr, exp):
    """Counting sort by specific digit."""
    n = len(arr)
    output = [0] * n
    count = [0] * 10  # Digits 0-9
    
    # Count occurrences of digit
    for num in arr:
        digit = (num // exp) % 10
        count[digit] += 1
    
    # Cumulative count
    for i in range(1, 10):
        count[i] += count[i - 1]
    
    # Build output (stable)
    for i in range(n - 1, -1, -1):
        digit = (arr[i] // exp) % 10
        output[count[digit] - 1] = arr[i]
        count[digit] -= 1
    
    # Copy to original
    for i in range(n):
        arr[i] = output[i]

# Time: O(d·(n+k)) where d=digits, k=base, Space: O(n+k)
# Typically O(n) for fixed-size integers
# Stable: Yes
# Use case: Large numbers of integers with bounded digits
```

**Bucket Sort**:

```python
def bucket_sort(arr):
    """
    Distribute elements into buckets, sort buckets, concatenate.
    
    Efficient for uniformly distributed data.
    """
    if not arr:
        return arr
    
    # Determine bucket count and range
    n = len(arr)
    min_val, max_val = min(arr), max(arr)
    bucket_count = n
    bucket_range = (max_val - min_val) / bucket_count
    
    # Create buckets
    buckets = [[] for _ in range(bucket_count)]
    
    # Distribute elements into buckets
    for num in arr:
        if num == max_val:
            idx = bucket_count - 1
        else:
            idx = int((num - min_val) / bucket_range)
        buckets[idx].append(num)
    
    # Sort each bucket and concatenate
    result = []
    for bucket in buckets:
        result.extend(sorted(bucket))  # Use any sort algorithm
    
    return result

# Time: O(n + k) average, O(n²) worst, Space: O(n + k)
# Stable: Depends on bucket sorting algorithm
# Use case: Uniformly distributed floating-point numbers
```

**Tim Sort (Python's default)**:

```python
def tim_sort_concept():
    """
    Python's built-in sort uses Tim Sort (hybrid merge + insertion).
    
    Optimized for real-world data:
    - Identifies and exploits existing runs (sorted subsequences)
    - Uses insertion sort for small runs
    - Merges runs efficiently
    
    Time: O(n log n) worst, O(n) best
    Space: O(n)
    Stable: Yes
    
    Implementation is complex - use built-in sorted() or list.sort()
    """
    pass

# Example usage
arr = [3, 1, 4, 1, 5, 9, 2, 6]
sorted_arr = sorted(arr)  # Creates new sorted list
arr.sort()  # In-place sorting
```

## Complexity Analysis

**Comparison-Based Sorting Lower Bound**: Ω(n log n)

| Algorithm | Best | Average | Worst | Space | Stable  | Use case |
|-----------|------|---------|-------|-------|--------| ----- |
| Bubble | O(n) | O(n²) | O(n²) | O(1) | Yes | Small/nearly sorted, educational |
| Selection | O(n²) | O(n²) | O(n²) | O(1) | No | Small arrays, minimizing swaps |
| Insertion | O(n) | O(n²) | O(n²) | O(1) | Yes | Good for small/sorted, online |
| Merge | O(n log n) | O(n log n) | O(n log n) | O(n) | Yes  | Need stability, predictable performance |
| Quick | O(n log n) | O(n log n) | O(n²) | O(log n) | No | Fast average, General purpose, large arrays |
| Heap | O(n log n) | O(n log n) | O(n log n) | O(1) | No | In-place, Guaranteed $O(n \log n)$ with $O(1)$ space |
| Counting | O(n+k) | O(n+k) | O(n+k) | O(k) | Yes | Small integer range (where $k$ is range) |
| Radix | O(d·n) | O(d·n) | O(d·n) | O(n+k) | Yes |Fixed-size integers (where $d$ is digits and base $k$) |
| Bucket | O(n+k) | O(n+k) | O(n²) | O(n+k) | Yes | Uniformly distributed floats |
| Tim | O(n) | O(n log n) | O(n log n) | O(n) | Yes | Python default |

## Common Questions

1. "Which sorting algorithm would you choose and why?"
   - **Answer depends on context**:
     - **General purpose**: Quick sort (fast average) or Tim sort (Python default)
     - **Need stability**: Merge sort or Tim sort
     - **Memory constrained**: Heap sort (O(1) space, guaranteed O(n log n))
     - **Nearly sorted**: Insertion sort (O(n) best case) or Tim sort
     - **Small integers**: Counting sort or Radix sort (linear time)
     - **Small array (< 50)**: Insertion sort (low overhead)
     - **External sorting**: Merge sort (sequential access)

2. "Explain stability and when it matters"
   - **Answer**: Stable sort preserves relative order of equal elements
   - **Why it matters**:
     - Sorting by multiple keys (sort by age, then by name - want same-age people in name order)
     - Database queries with ORDER BY multiple columns
     - Maintaining original order as tie-breaker
   - **Stable**: Merge, Insertion, Bubble, Counting, Radix, Tim
   - **Unstable**: Quick, Heap, Selection

3. "How would you sort a linked list?"
   - **Answer**: 
     - **Merge sort**: Natural fit (O(1) space for linked lists, no random access needed)
     - **Not suitable**: Quick sort (needs random access), Heap sort (array-based)
     - Implementation:
   ```python
   def merge_sort_linked_list(head):
       """Merge sort for linked list."""
       if not head or not head.next:
           return head
       
       # Find middle using slow/fast pointers
       slow, fast = head, head.next
       while fast and fast.next:
           slow = slow.next
           fast = fast.next.next
       
       # Split list
       mid = slow.next
       slow.next = None
       
       # Sort halves
       left = merge_sort_linked_list(head)
       right = merge_sort_linked_list(mid)
       
       # Merge
       return merge_lists(left, right)
   ```

4. "Find k-th largest element efficiently"
   - **Answer**: 
     - **Quick Select**: O(n) average (covered earlier)
     - **Min Heap**: O(n log k) - maintain heap of size k
     - **Full sort**: O(n log n) - overkill
     - **Choice**: Quick select for one-time query, heap for multiple queries

## Problem-Solving Patterns

1. **When to sort first**:
   - Two-sum, three-sum problems
   - Finding pairs with given difference
   - Merge overlapping intervals
   - Many optimization problems

2. **Custom comparators**:
   ```python
   # Sort by multiple criteria
   people = [(name, age) for ...]
   people.sort(key=lambda x: (x[1], x[0]))  # By age, then name
   
   # Sort complex objects
   from dataclasses import dataclass
   @dataclass
   class Person:
       name: str
       age: int
   
   people.sort(key=lambda p: p.age)
   ```

3. **Partial sorting**:
   - Use `heapq.nlargest/nsmallest` for top-k
   - Use `numpy.argpartition` for partial sorting
   - Don't full-sort when partial suffices

## Common Mistakes

- Using $O(n^2)$ algorithms for large arrays
- Not considering stability when needed
- Forgetting to handle edge cases (empty, single element)
- Comparing objects without proper comparison logic
- Not using built-in sort when appropriate

## Edge Cases

- Empty array
- Single element
- All elements equal
- Already sorted (best case for some algorithms)
- Reverse sorted (worst case for some algorithms)
- Duplicates

---

**[Searching & Sorting Questions Notebook](./Notebooks/Searching-Sorting.ipynb)**
