<h1 align="center"> Stack </h1>

A stack is a linear data structure that follows the Last In First Out (LIFO) principle. The last element that is added to the stack is the first one to be removed. The operations of a stack are performed at one end of the stack, which is known as the top of the stack. The operations of a stack are:

1. **Push**: Adds an element to the top of the stack.
2. **Pop**: Removes an element from the top of the stack.
3. **Peek**: Returns the top element of the stack without removing it.
4. **isEmpty**: Returns true if the stack is empty, else false.

### Implementations

1. **Using Arrays**: In this implementation, we use an array to store the elements of the stack. The top of the stack is represented by the index of the last element in the array. The operations of the stack are performed by manipulating the index of the top element.

2. **Using Linked List**: In this implementation, we use a linked list to store the elements of the stack. The top of the stack is represented by the head of the linked list. The operations of the stack are performed by manipulating the head of the linked list.

### Time Complexity

The time complexity of the operations of a stack implemented using an array or a linked list is as follows:

1. **Push**: `O(1)`
2. **Pop**: `O(1)`
3. **Peek**: `O(1)`
4. **isEmpty**: `O(1)`
5. **Search**: `O(n)`
6. **Size**: `O(1)`
7. **Clear**: `O(1)`

## Questions

1. Baseball game
2. Valid Parentheses
3. Implement Stack using Queues
4. Implement a min stack
5. Number of students unable to eat lunch
