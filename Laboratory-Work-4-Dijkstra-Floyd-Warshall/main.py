import random
import matplotlib.pyplot as plt
import networkx as nx
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


def generate_sparse_graph(num_vertices):
    # Parameters
    num_nodes = num_vertices
    sparsity = 0.05  # Probability of edge creation
    min_weight = 20
    max_weight = 20

    # Generate a random graph (Erdős-Rényi) with given sparsity
    G = nx.erdos_renyi_graph(num_nodes, sparsity)

    # Assign random weights to edges
    for u, v in G.edges():
        weight = random.randint(min_weight, max_weight)
        G[u][v]['weight'] = weight

    dict_G = nx.to_dict_of_dicts(G)
    for i, j in dict_G.items():
        for k in j:
            weight = dict_G[i][k]['weight']
            dict_G[i][k] = None
            dict_G[i][k] = weight

    return dict_G


def generate_dense_graph(num_nodes):
    graph = {node: {} for node in range(num_nodes)}
    max_weight = 10

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                weight = random.randint(1, max_weight)
                graph[i][j] = weight

    return graph


def dijkstra(graph, start):
    # Initialize distances from start node to all other nodes as infinity
    distances = {node: float('inf') for node in graph}
    distances[start] = 0

    # Set to keep track of visited nodes
    visited = set()

    while True:
        # Find the node with the minimum distance from start among unvisited nodes
        min_distance = float('inf')
        min_node = None
        for node in graph:
            if node not in visited and distances[node] < min_distance:
                min_distance = distances[node]
                min_node = node

        # If all nodes have been visited or there are no more reachable nodes, break
        if min_node is None:
            break

        # Mark the minimum distance node as visited
        visited.add(min_node)

        # Update distances to neighbors of the current node
        for neighbor, weight in graph[min_node].items():
            distance = distances[min_node] + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance

    return distances


def floyd_warshall(graph, start):
    # Number of vertices in the graph
    n = len(graph)

    # Initialize distance matrix with the same values as the graph
    distance = [[float('inf') if i != j else 0 for j in range(n)] for i in range(n)]

    # Create a mapping from node labels to indices
    node_indices = {node: i for i, node in enumerate(graph)}

    # Update distance matrix with known edges
    for u in graph:
        for v in graph[u]:
            distance[node_indices[u]][node_indices[v]] = graph[u][v]

    # Floyd-Warshall dynamic programming algorithm
    for k in range(n):
        for i in range(n):
            for j in range(n):
                distance[i][j] = min(distance[i][j], distance[i][k] + distance[k][j])

    # Convert the distance matrix back to a dictionary format
    distances = {node: distance[node_indices[start]][node_indices[node]] for node in graph}

    # Return the shortest distances from the start node to all other vertices
    return distances


# Example graph
graph = {
    'A': {'B': 1, 'C': 4},
    'B': {'A': 1, 'C': 2, 'D': 5},
    'C': {'A': 4, 'B': 2, 'D': 1},
    'D': {'B': 5, 'C': 1}
}

start_node = 'A'  # Specify the start node

print("Dijkstra - Shortest distances from node", start_node, ":", dijkstra(graph, start_node))
print("Floyd-Warshall - Shortest distances from node", start_node, ":", floyd_warshall(graph, start_node))

import time

# Test parameters
num_nodes = [25, 30, 35, 40, 45, 50, 75, 100, 200, 300, 400, 500]
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
    nx.draw_networkx_nodes(G, pos, node_size=700)

    # Draw edges
    nx.draw_networkx_edges(G, pos)

    # Draw edge labels
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=20, font_family='sans-serif')

    plt.axis('off')
    plt.show()


def measure_dijkstra_time(graph):
    start_time = time.time()
    dijkstra_distances = dijkstra(graph, 0)
    end_time = time.time()
    return end_time - start_time, dijkstra_distances

def measure_floyd_warshall_time(graph):
    start_time = time.time()
    floyd_distances = floyd_warshall(graph, 0)
    end_time = time.time()
    return end_time - start_time, floyd_distances


# Visualize sparse graph
visualize_graph(sparse_graphs[2])
print(sparse_graphs[2])
# Visualize dense graph
visualize_graph(dense_graphs[2])
print(dense_graphs[2])

dijkstra_times_sparse = []
floyd_times_sparse = []
dijkstra_times_dense = []
floyd_times_dense = []


def plot_graphs(graph_sizes, dijkstra_times, floyd_times, type_graphs):
    plt.figure(figsize=(10, 6))
    plt.plot(graph_sizes, dijkstra_times, marker='o', label='Dijkstra')
    plt.plot(graph_sizes, floyd_times, marker='s', label='Floyd-Warshall')
    plt.xlabel('Nr. of Vertices')
    plt.ylabel('Time (seconds)')
    plt.title(f'Implemented Dijkstra and Floyd-Warshall Time Comparison on {type_graphs}')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_method(plt, graph_sizes, times_sparse, times_dense, type_algorithm, type_graphs):
    plt.plot(graph_sizes, times_sparse, marker='o', label=f'{type_algorithm} on Sparse Graph')
    plt.plot(graph_sizes, times_dense, marker='s', label=f'{type_algorithm} on Dense Graph')
    plt.xlabel('Nr. of Vertices')
    plt.ylabel('Time (seconds)')
    plt.title(f'Implemented {type_algorithm} Time Comparison on {type_graphs}')


def plot_graphs_Floyd(graph_sizes, floyd_times_sparse, floyd_times_dense, type_graphs):
    plt.figure(figsize=(10, 6))
    plot_method(plt, graph_sizes, floyd_times_sparse, floyd_times_dense, 'Floyd-Warshall', type_graphs)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_graphs_Dijkstra(graph_sizes, dijkstra_times_sparse, dijkstra_times_dense, type_graphs):
    plt.figure(figsize=(10, 6))
    plot_method(plt, graph_sizes, dijkstra_times_sparse, dijkstra_times_dense, 'Dijkstra', type_graphs)
    plt.legend()
    plt.grid(True)
    plt.show()


for sparse_graph in sparse_graphs:
    dijkstra_time, _ = measure_dijkstra_time(graph=sparse_graph)
    floyd_time, _ = measure_floyd_warshall_time(graph=sparse_graph)
    dijkstra_times_sparse.append(dijkstra_time)
    floyd_times_sparse.append(floyd_time)

for dense_graph in dense_graphs:
    dijkstra_time, _ = measure_dijkstra_time(graph=dense_graph)
    floyd_time, _ = measure_floyd_warshall_time(graph=dense_graph)
    dijkstra_times_dense.append(dijkstra_time)
    floyd_times_dense.append(floyd_time)

plot_graphs(graph_sizes=num_nodes, dijkstra_times=dijkstra_times_sparse, floyd_times=floyd_times_sparse,
            type_graphs="Sparse Graphs")
plot_graphs(graph_sizes=num_nodes, dijkstra_times=dijkstra_times_dense, floyd_times=floyd_times_dense,
            type_graphs="Dense Graphs")

plot_graphs_Floyd(graph_sizes=num_nodes, floyd_times_sparse=floyd_times_sparse, floyd_times_dense=floyd_times_dense,
                  type_graphs="Sparse and Dense Graphs")
plot_graphs_Dijkstra(graph_sizes=num_nodes, dijkstra_times_sparse=dijkstra_times_sparse,
                     dijkstra_times_dense=dijkstra_times_dense, type_graphs="Sparse and Dense Graphs")



# Test parameters
num_nodes = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
sparse_graphs = []
dense_graphs = []

# Generate graphs
for n in num_nodes:
    start_time = time.time()
    sparse_graph = generate_sparse_graph(n)
    sparse_time = time.time() - start_time
    start_time = time.time()
    dense_graph = generate_dense_graph(n)
    dense_time = time.time() - start_time
    print("Number of nodes:", n)
    sparse_graphs.append(sparse_graph)
    dense_graphs.append(dense_graph)

dijkstra_times_sparse = []
dijkstra_times_dense = []

for sparse_graph in sparse_graphs:
    dijkstra_time, _ = measure_dijkstra_time(graph=sparse_graph)
    dijkstra_times_sparse.append(dijkstra_time)

for dense_graph in dense_graphs:
    dijkstra_time, _ = measure_dijkstra_time(graph=dense_graph)
    dijkstra_times_dense.append(dijkstra_time)

plot_graphs_Dijkstra(graph_sizes=num_nodes, dijkstra_times_sparse=dijkstra_times_sparse,
                     dijkstra_times_dense=dijkstra_times_dense, type_graphs="Sparse and Dense Graphs")