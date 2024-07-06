<h1 align="center"> Strings </h1>

A string is a sequence of characters. In Python, strings are immutable, which means that once a string is created, it cannot be changed. There is no `char` data type in Python, and a character is represented as a string of length 1. 

The operations that can be performed on strings include:

1. **Concatenation**: Combining two or more strings to create a new string.
2. **Indexing**: Accessing individual characters in a string using their position.
3. **Slicing**: Extracting a substring from a string.
4. **Length**: Finding the length of a string.
5. **Membership**: Checking if a character or substring is present in a string.
6. **Repetition**: Repeating a string multiple times.
7. **Comparison**: Comparing two strings lexicographically.
8. **Traversal**: Iterating over the characters of a string.

### Time Complexity

The time complexity of the operations on strings is as follows:

1. **Indexing**: `O(1)`
2. **Slicing**: `O(k)` where `k` is the length of the slice.
3. **Concatenation**: `O(n)` where `n` is the length of the resulting string.
4. **Length**: `O(1)`
5. **Membership**: `O(n)` where `n` is the length of the string.
6. **Repetition**: `O(n)` where `n` is the number of repetitions.
7. **Comparison**: `O(n)` where `n` is the length of the shorter string.
8. **Traversal**: `O(n)` where `n` is the length of the string.

## Questions

1. Minimum Deletions to Make Character Frequencies Unique
