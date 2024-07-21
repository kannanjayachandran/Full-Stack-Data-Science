<h1 align="center"> LinkedLists </h1>

Linked lists are a linear data structure consisting of nodes where each node contains a data field and a reference to the next node in the sequence. The first node is called the `head`, and the last node is called the `tail`. The `tail` node points to `null` or `None` to indicate the end of the list. Linked lists are dynamic data structures that can grow or shrink in size during execution. They are efficient for insertion and deletion operations.

## Types of Linked Lists

1. **Singly Linked List**: Each node has a data field and a reference to the next node.

2. **Doubly Linked List**: Each node has a data field and references to the next and previous nodes.

3. **Circular Linked List**: The last node points to the first node, forming a circle.

### Singly Linked List

![Singly Linked List](./Singly_linked_list.png)

A simple Node would look like this:

```python
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None
```

We can create a simple naive implementation of linked list using the `Node` class:

```python
head = Node(10)
head.next = Node(20)
head.next.next = Node(30)
```

We can traverse the above linked list and print the data of each node using the following code:

```python   
def print_linked_list(head):
    current = head
    while current:
        print(current.data)
        current = current.next
```

We search for a particular element in the linked list and return the position if found, else return -1:

```python
def search(head, key):
    current = head
    position = 1
    while current:
        if current.data == key:
            return position
        current = current.next
        position += 1
    return -1
```

We can insert a new node at the beginning of the linked list using the following code:

```python
def insert_at_beginning(head, data):
    new_node = Node(data)
    new_node.next = head
    return new_node
```

We can insert a new node at the end of the linked list using the following code:

```python
def insert_at_end(head, data):
    new_node = Node(data)
    if not head:
        return new_node
    current = head
    while current.next:
        current = current.next
    current.next = new_node
    return head
```

We can insert a new node at a given position in the linked list using the following code:

```python
def insert_at_pos(head, data, position):
    new_node = Node(data)
    if position == 1:
        new_node.next = head
        return new_node
    current = head
    for _ in range(position-2):
        if current is None:
            raise IndexError("Position out of bound")
        current = current.next

    if current is None:
        raise IndexError("Position out of bound")

    new_node.next = current.next
    current.next = new_node

    return head
```

We can delete the first node of the linked list using the following code:

```python
def delete_first_node(head):
    if not head:
        return None
    return head.next
```

We can delete the last node of the linked list using the following code:
    
```python
def delete_last_node(head):
    if head is None or head.next is None:
        return None
    current = head
    while current.next.next:
        current = current.next
    current.next = None
    return head
```

We can reverse a linked list using mainly 2 methods:

- Using auxiliary space (Use a stack or list to store the elements of the linked list and then pop them to create a new linked list)

- Reversing the links and updating the head

Following is the code to reverse a linked list using both the methods:

```python
def reverse_linked_list(head):
    prev = None
    current = head
    while current is not None:
        next_node = current.next
        current.next = prev
        prev = current
        current = next_node
    return prev
```

```python
def reverse_list_stack(head):
    stack = []
    current = head
    while current is not None:
        stack.append(current.data)
        current = current.next
    new_head = Node(stack.pop())
    current = new_head
    while stack:
        current.next = Node(stack.pop())
        current = current.next
    return new_head
```

The second method is more efficient as it has a time complexity of $O(n)$ and space complexity of $O(1)$.

Recursive solution to reverse a linked list:

```python
def reverse_linked_list_recursive(current, prev):
    if current is None:
        return prev
    next_node = current.next
    current.next = prev
    return reverse_linked_list_recursive(next_node, current)
```

**[Implementation](./Implementation.ipynb)**

## Questions

1. Reverse a linked list
2. Merge two sorted linked lists
3. Design browser history using linked list
4. Find the duplicate element {Floyd's Cycle Detection Algorithm}
