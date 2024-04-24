import random
import matplotlib.pyplot as plt
import networkx as nx
import time
import heapq

def generate_graph(nodes, sparse=True):
    if sparse:
        edges = nodes - 5
    else:
        edges = nodes * (nodes - 1) // 4  # A more dense graph
    G = nx.gnm_random_graph(nodes, edges)

    for (u, v) in G.edges():
        G.edges[u, v]['weight'] = random.randint(1, 10)  # Assign a random weight

    dict_G = nx.to_dict_of_dicts(G)
    for i, j in dict_G.items():
        for k in j:
            weight = dict_G[i][k]['weight']
            dict_G[i][k] = None
            dict_G[i][k] = weight

    return dict_G


def generate_sparse_graph(num_vertices, edge_probability=0.1):
    G = nx.erdos_renyi_graph(num_vertices, edge_probability, directed=True)
    graph = nx.to_dict_of_dicts(G)
    for u in graph:
        for v in graph[u]:
            graph[u][v] = random.randint(1, 10)
    return graph

def generate_dense_graph(num_vertices, edge_probability=0.9):
    G = nx.erdos_renyi_graph(num_vertices, edge_probability, directed=True)
    graph = nx.to_dict_of_dicts(G)
    for u in graph:
        for v in graph[u]:
            graph[u][v] = random.randint(1, 10)
    return graph

# Test parameters
num_nodes = [25, 30, 35, 40, 45, 50, 75, 100, 200, 300, 400, 500]
num_nodes += [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
sparse_density = 0.2
sparse_graphs = []
dense_graphs = []

# Generate graphs
for n in num_nodes:
    start_time = time.time()
    # sparse_graph = generate_graph(n, sparse=True)
    sparse_graph = generate_sparse_graph(n)
    sparse_time = time.time() - start_time
    start_time = time.time()
    # dense_graph = generate_graph(n, sparse=False)
    dense_graph = generate_dense_graph(n)
    dense_time = time.time() - start_time
    print("Number of nodes:", n)
    sparse_graphs.append(sparse_graph)
    dense_graphs.append(dense_graph)

def visualize_graph(graph):
    G = nx.Graph()
    for node, neighbors in graph.items():
        for neighbor, weight in neighbors.items():
            G.add_edge(node, neighbor, weight=weight)

    pos = nx.spring_layout(G)  # positions for all nodes

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=400)

    # Draw edges
    nx.draw_networkx_edges(G, pos)

    # Draw edge labels
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')

    plt.axis('off')
    plt.show()
def kruskal(graph):
    # This will store the resultant MST
    result = []

    # An index variable, used for result[]
    e = 0

    # Convert the graph dictionary to a list of edges
    graph2 = []
    for u in graph:
        for v, w in graph[u].items():
            graph2.append((u, v, w))

    # Sort all the edges in
    # non-decreasing order of their
    # weight
    graph2 = sorted(graph2,
                        key=lambda item: item[2])

    parent = {}
    rank = {}

    # Function to find set of an element i
    def find(i):
        if parent[i] != i:
            parent[i] = find(parent[i])
        return parent[i]

    # Function to perform union of two sets
    def union(x, y):
        xroot = find(x)
        yroot = find(y)

        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot
        else:
            parent[yroot] = xroot
            rank[xroot] += 1

    # Create subsets with single elements
    for node in graph:
        parent[node] = node
        rank[node] = 0

    # Number of edges to be taken is less than to V-1
    while e < len(graph) - 1:
        # Pick the smallest edge
        u, v, w = graph2[e]
        e = e + 1
        x = find(u)
        y = find(v)

        # If including this edge doesn't
        # cause cycle, then include it in result
        if x != y:
            e = e + 1
            result.append([u, v, w])
            union(x, y)

    return result


def prim(graph):
    # Prim's algorithm for finding Minimum Spanning Tree (MST)
    # Initialize an empty list to store the MST
    mst = []

    # Initialize a set to keep track of visited vertices
    visited = set()

    # Choose an arbitrary starting vertex
    start = next(iter(graph))

    # Use a priority queue to keep track of the edges crossing the cut
    edges = [(weight, start, neighbor) for neighbor, weight in graph[start].items()]
    heapq.heapify(edges)

    # Mark the starting vertex as visited
    visited.add(start)

    # Loop until all vertices are visited
    while edges:
        # Get the edge with the smallest weight crossing the cut
        weight, u, v = heapq.heappop(edges)

        # If the destination vertex is not visited, add the edge to the MST
        if v not in visited:
            mst.append((u, v, weight))
            visited.add(v)

            # Add edges from the newly visited vertex to the priority queue
            for neighbor, weight in graph[v].items():
                if neighbor not in visited:
                    heapq.heappush(edges, (weight, v, neighbor))

    return mst

# Visualize sparse graph
visualize_graph(sparse_graphs[1])
print(sparse_graphs[1])
# Visualize dense graph
visualize_graph(dense_graphs[1])
print(dense_graphs[1])

# Measure execution times
kruskal_times_sparse = []
prim_times_sparse = []

for sparse_graph in sparse_graphs:
    start_time = time.time()
    kruskal(sparse_graph)
    kruskal_time = time.time() - start_time
    kruskal_times_sparse.append(kruskal_time)

    start_time = time.time()
    prim(sparse_graph)
    prim_time = time.time() - start_time
    prim_times_sparse.append(prim_time)

# Plot the comparison of Kruskal's and Prim's algorithms on sparse graphs
plt.figure(figsize=(10, 6))
plt.plot(num_nodes, kruskal_times_sparse, marker='o', label="Kruskal's Algorithm")
plt.plot(num_nodes, prim_times_sparse, marker='s', label="Prim's Algorithm")
plt.xlabel('Number of Vertices')
plt.ylabel('Time (seconds)')
plt.title('Comparison of Kruskal\'s and Prim\'s Algorithms on Sparse Graphs')
plt.legend()
plt.grid(True)
plt.show()

# Measure execution times
kruskal_times_dense = []
prim_times_dense = []

for dense_graph in dense_graphs:
    start_time = time.time()
    kruskal(dense_graph)
    kruskal_time = time.time() - start_time
    kruskal_times_dense.append(kruskal_time)

    start_time = time.time()
    prim(dense_graph)
    prim_time = time.time() - start_time
    prim_times_dense.append(prim_time)

# Plot the comparison of Kruskal's and Prim's algorithms on sparse graphs
# plt.figure(figsize=(10, 6))
# plt.plot(num_nodes, prim_times_dense, marker='s', label="Kruskal's Algorithm")
# plt.plot(num_nodes, kruskal_times_dense, marker='o', label="Prim's Algorithm")
#
# plt.xlabel('Number of Vertices')
# plt.ylabel('Time (seconds)')
# plt.title('Comparison of Kruskal\'s and Prim\'s Algorithms on Dense Graphs')
# plt.legend()
# plt.grid(True)
# plt.show()

plt.figure(figsize=(10, 6))
plt.plot(num_nodes, prim_times_sparse, marker='s', label="Dense")
plt.plot(num_nodes, prim_times_dense, marker='o', label="Sparse")

plt.xlabel('Number of Vertices')
plt.ylabel('Time (seconds)')
plt.title('Comparison of Prim\'s Algorithms on Sparse and Dense Graphs')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(num_nodes, kruskal_times_sparse, marker='s', label="Sparse")
plt.plot(num_nodes, prim_times_dense, marker='o', label="Dense")

plt.xlabel('Number of Vertices')
plt.ylabel('Time (seconds)')
plt.title('Comparison of Kruskal\'s Algorithms on Sparse and Dense Graphs')
plt.legend()
plt.grid(True)
plt.show()