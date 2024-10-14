<h1 align="center"> Backtracking </h1>

Backtracking is a problem-solving algorithm that explores all possible solutions by incrementally building candidates and abandoning (`backtracking`) those that fail to satisfy the problem constraints.

We generally employ backtracking when we have the following premises:

1. We are searching for a solution in a search space.

2. The problem is formulated in terms of choices and we may need to make a sequence of choices to arrive at a solution.

4. We need to backtrack when we hit a dead end and continue the search from the last valid position.

5. We need to explore all possible choices to find the solution.

## Use cases of Backtracking

- `Constraint satisfaction problems` (e.g., n-queen, map coloring, crypto-arithmetic puzzles)

- `Combinatorial optimization` (e.g., traveling salesman problem, knapsack problem, subset sum problem)

- `Parsing in compilers` (e.g., recursive descent parsing)

- `AI and game playing` (e.g., solving chess endgames)

- `Circuit design and verification` (e.g., placing components on a circuit board)

- `Permutation and combinations` (e.g., generating all possible permutations of a set of numbers, combination sum)

- `Path finding in Graphs` (e.g., finding all possible paths in a maze)

## Efficiency

Backtracking is a brute-force algorithm that explores all possible solutions to find the correct solution. The efficiency of backtracking depends on the number of choices we have at each step and the depth of the search tree. For large inputs; the time complexity of backtracking can be exponential. It works best when combined with pruning techniques to eliminate many potential solutions early.

## Types of Backtracking

1. `Simple Backtracking`: In simple backtracking, we explore all possible solutions to find the correct solution. We backtrack when we hit a dead end and continue the search from the last valid position.

2. `Optimized Backtracking`: In optimized backtracking, we use pruning techniques to eliminate many potential solutions early. This reduces the number of choices we need to explore and makes the algorithm more efficient.

## Backtracking Template

```python
def backtrack(candidate, data):
    if is_valid(candidate):
        if is_goal(candidate):
            output(candidate)
            return
        for next_candidate in get_candidates(candidate, data):
            backtrack(next_candidate, data)
```
