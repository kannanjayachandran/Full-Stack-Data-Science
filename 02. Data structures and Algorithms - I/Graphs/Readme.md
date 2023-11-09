# Graphs



```py
from graph import Graph

g = Graph()
g.add_vertex('A')
g.add_vertex('B')
g.add_vertex('C')
g.add_vertex('D')

g.add_edge('A', 'B')
g.add_edge('A', 'C')
g.add_edge('B', 'D')
g.add_edge('C', 'D')

g.print_graph()

    A: B C 
    B: A D 
    C: A D 
    D: B C


# Adjacency list representation of graph
class GraphNode():

    def __init__(self, val):
        self.val = val
        self.neighbors = []

# OR use dictionary
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D'],
    'C': ['A', 'D'],
    'D': ['B', 'C']
}
```