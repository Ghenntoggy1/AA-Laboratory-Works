import random
import time
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


# Path from Start Node to End Node Complete/Incomplete Undirected Graphs
def dfs(graph, start, end):
    stack = [(start, [start])]
    while stack:
        # print("DFS Stack:", stack)
        (node, path) = stack.pop()
        if node == end:
            return path
        if node in graph:
            for neighbor in graph[node]:
                if neighbor not in path:
                    stack.append((neighbor, path + [neighbor]))
    return None


# Breadth-First Search (BFS) algorithm without recursion
def bfs(graph, start, end):
    queue = [(start, [start])]
    while queue:
        # print("BFS Queue:", queue)
        (node, path) = queue.pop(0)
        if node == end:
            return path
        if node in graph:
            for neighbor in graph[node]:
                if neighbor not in path:
                    queue.append((neighbor, path + [neighbor]))
    return None


# Function to generate a random undirected graph
def generate_random_graph(num_nodes):
    # Generate an empty directed graph
    G = nx.DiGraph()
    # Add edges to the graph randomly
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and random.random() < 0.2:
                G.add_edge(i, j)
    # print(G.is_directed())
    # print(G.adjacency())
    return G


# Function to plot the graph
# def plot_graph(graph, type_graphs, num_nodes):
#     fig, ax = plt.subplots()
#     G = nx.Graph()
#     for node, neighbors in graph.items():
#         for neighbor in neighbors:
#             G.add_edge(node, neighbor)
#     pos = nx.spring_layout(G)  # Compute node positions using spring layout
#     nx.draw(G, pos, with_labels=True, node_size=300, node_color='skyblue', edge_color='gray', linewidths=0.5)
#     limits = plt.axis('on')  # turns on axis
#     ax.tick_params(left=True, bottom=True)
#     plt.title(f'{type_graphs} with Number of Vertices = {num_nodes}')
#     plt.show()
def plot_graph(G, num_nodes):
    # Visualize the graph
    fig, ax = plt.subplots()
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=500, node_color="skyblue", font_size=10, arrowsize=10)
    plt.title(f"Directed Graph with {num_nodes} nodes")
    limits = plt.axis('on')  # turns on axis
    ax.tick_params(left=True, bottom=True)
    plt.show()

# Function to print the adjacency matrix
def print_adjacency_matrix(G):
    adjacency_matrix = nx.adjacency_matrix(G).todense()
    print("Adjacency Matrix:")
    print(adjacency_matrix)


# Main function to measure execution times, plot the constructed graph, and plot the results
if __name__ == "__main__":
    num_nodes_list = [10, 15, 25, 50, 75, 100] + [i for i in range(100, 2000, 50)]

    dfs_times = []
    bfs_times = []

    for num_nodes in num_nodes_list:
        print(f"\nNumber of Nodes: {num_nodes}")
        # Generate random directed graph
        graph = generate_random_graph(num_nodes)

        start_node = 0
        # Print the adjacency matrix
        # print_adjacency_matrix(graph)
        # Plot the constructed directed graph
        if num_nodes <= 50:
            print("Plotting the constructed directed graph...")
            plot_graph(graph, num_nodes)

        end_node = num_nodes - 1

        # Measure execution time for DFS
        start_time = time.time()
        dfs_path = dfs(graph, start_node, end_node)
        end_time = time.time()
        dfs_time = end_time - start_time
        print("DFS Path: ", dfs_path)
        print("DFS Time: ", dfs_time)
        dfs_times.append(dfs_time)

        # Measure execution time for BFS
        start_time = time.time()
        bfs_path = bfs(graph, start_node, end_node)
        end_time = time.time()
        bfs_time = end_time - start_time
        print("BFS Path: ", bfs_path)
        print("BFS Time: ", bfs_time)
        bfs_times.append(bfs_time)

    # Plotting
    plt.plot(num_nodes_list, dfs_times, label='DFS')
    plt.plot(num_nodes_list, bfs_times, label='BFS')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Time (seconds)')
    plt.title('Implemented DFS and BFS on Directed Graph with Start and End Node Far From Each Other')
    plt.legend()
    plt.grid(True)
    plt.show()
