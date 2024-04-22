import random
import time
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


# Path from Start Node to End Node Complete/Incomplete Directed Graphs
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


def plot_traversed_dfs(G, dfs_nodes):
    fig, ax = plt.subplots()
    # Visualize the graph and highlight visited nodes
    pos = nx.spring_layout(G)  # Layout for better visualization
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_edges(G, pos, edgelist=[(dfs_nodes[i], dfs_nodes[i + 1]) for i in range(len(dfs_nodes) - 1)],
                           edge_color='r', width=2)

    # Highlight starting node
    nx.draw_networkx_nodes(G, pos, nodelist=[dfs_nodes[0]], node_color='g', node_size=700)
    plt.title("Graph with DFS Traversal")
    limits = plt.axis('on')  # turns on axis
    ax.tick_params(left=True, bottom=True)
    plt.show()


def plot_traversed_bfs(G, bfs_nodes):
    fig, ax = plt.subplots()
    # Visualize the graph and highlight visited nodes
    pos = nx.spring_layout(G)  # Layout for better visualization
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_edges(G, pos, edgelist=[(bfs_nodes[i], bfs_nodes[i + 1]) for i in range(len(bfs_nodes) - 1)],
                           edge_color='b', width=2)

    # Highlight starting node
    nx.draw_networkx_nodes(G, pos, nodelist=[bfs_nodes[0]], node_color='g', node_size=700)
    limits = plt.axis('on')  # turns on axis
    ax.tick_params(left=True, bottom=True)
    plt.title("Graph with BFS Traversal")
    plt.show()


# Main function to measure execution times, plot the constructed graph, and plot the results
if __name__ == "__main__":
    num_nodes_list = [5, 10, 15, 25, 50, 75, 100] + [i for i in range(100, 1500, 50)]

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
        if num_nodes <= 25:
            print("Plotting the constructed directed graph...")
            plot_graph(graph, num_nodes)

        end_node = num_nodes // 2

        # Measure execution time for DFS
        start_time = time.time()
        dfs_path = dfs(graph, start_node, end_node)
        end_time = time.time()
        dfs_time = end_time - start_time
        print("DFS Path: ", dfs_path)
        print("DFS Time: ", dfs_time)
        dfs_times.append(dfs_time)
        if dfs_path is not None and num_nodes <= 25:
            plot_traversed_dfs(graph, dfs_path)

        # Measure execution time for BFS
        start_time = time.time()
        bfs_path = bfs(graph, start_node, end_node)
        end_time = time.time()
        bfs_time = end_time - start_time
        print("BFS Path: ", bfs_path)
        print("BFS Time: ", bfs_time)
        bfs_times.append(bfs_time)
        if bfs_path is not None and num_nodes <= 25:
            plot_traversed_bfs(graph, bfs_path)

    # Plotting
    plt.plot(num_nodes_list, dfs_times, label='DFS')
    plt.plot(num_nodes_list, bfs_times, label='BFS')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Time (seconds)')
    plt.title('Implemented DFS and BFS on Directed Graph with Start and End Node Far From Each Other')
    plt.legend()
    plt.grid(True)
    plt.show()
