import random
import time
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from collections import defaultdict, deque

from networkx import dfs_tree, bfs_tree, dfs_edges, bfs_edges


# Full Traversal of Complete Undirected Graphs
class Graph:
    def __init__(self):
        self.graph = defaultdict(list)

    def add_edge(self, u, v):
        self.graph[u].append(v)

    def dfs(self, v):
        visited = [False] * len(self.graph)
        stack = []
        stack.append(v)
        while stack:
            v = stack.pop()
            if not visited[v]:
                visited[v] = True
                for i in self.graph[v]:
                    if not visited[i]:
                        stack.append(i)

    def bfs(self, v):
        visited = [False] * len(self.graph)
        queue = []
        visited[v] = True
        queue.append(v)
        while queue:
            v = queue.pop(0)
            for i in self.graph[v]:
                if not visited[i]:
                    visited[i] = True
                    queue.append(i)

def measure_time(func, *args):
    start_time = time.time()
    func(*args)
    end_time = time.time()
    return end_time - start_time


def analyze_scalability(lst_graphs):
    dfs_times = []
    bfs_times = []
    graph_sizes = []

    for graph in lst_graphs:
        dfs_time = measure_time(graph.dfs, 0)
        bfs_time = measure_time(graph.bfs, 0)
        # nx_graph = nx.Graph(graph.graph)
        # start_time = time.time()
        # dfs_nodes = dfs_edges(nx_graph, 0)
        # end_time = time.time()
        # dfs_time = end_time - start_time
        # start_time = time.time()
        # bfs_nodes = bfs_edges(nx_graph, 0)
        # end_time = time.time()
        # bfs_time = end_time - start_time

        print("Num Nodes:", len(graph.graph.keys()))
        print("DFS:", dfs_time)
        print("BFS:", bfs_time)

        # plot_traversed(nx_graph, list(dfs_nodes), list(bfs_nodes))

        dfs_times.append(dfs_time)
        bfs_times.append(bfs_time)
        graph_sizes.append(len(graph.graph.keys()))

    return graph_sizes, dfs_times, bfs_times


def generate_wider_shallower_graph(num_nodes):
    graph_class = Graph()
    graph = {}
    for i in range(num_nodes):
        graph[i] = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            # Adjust the probability condition for adding edges
            if random.random() < (2 / (j - i)):  # Adjusted probability for wider and shallower graph
                graph[i].append(j)
                graph[j].append(i)
    graph_class.graph = graph
    print(len(graph_class.graph.keys()))
    if num_nodes <= 100:
        plot_graph(graph, "Wide Shallow Graphs", num_nodes)
    return graph_class


def generate_deep_narrow_graph(num_nodes):
    graph_class = Graph()
    graph = {}
    for i in range(num_nodes):
        graph[i] = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            # Adjust the probability condition for adding edges
            if random.random() < (0.5 / (j - i)):  # Adjusted probability for deep and narrow graph
                graph[i].append(j)
                graph[j].append(i)
    graph_class.graph = graph
    print(len(graph_class.graph.keys()))
    if num_nodes <= 100:
        plot_graph(graph, "Deep Narrow Graphs", num_nodes)
    return graph_class


def plot_graphs(graph_sizes, dfs_times, bfs_times, type_graphs):
    plt.figure(figsize=(10, 6))
    plt.plot(graph_sizes, dfs_times, marker='o', label='DFS')
    plt.plot(graph_sizes, bfs_times, marker='s', label='BFS')
    plt.xlabel('Nr. of Vertices')
    plt.ylabel('Time (seconds)')
    plt.title(f'Implemented BFS and DFS Time Comparison on {type_graphs}')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_graph(graph, type_graphs, num_nodes):
    fig, ax = plt.subplots()
    G = nx.Graph()
    for node, neighbors in graph.items():
        for neighbor in neighbors:
            G.add_edge(node, neighbor)
    pos = nx.spring_layout(G)  # Compute node positions using spring layout
    nx.draw(G, pos, with_labels=True, node_size=300, node_color='skyblue', edge_color='gray', linewidths=0.5)
    limits = plt.axis('on')  # turns on axis
    ax.tick_params(left=True, bottom=True)
    plt.title(f'{type_graphs} with Number of Vertices = {num_nodes}')
    plt.show()

def plot_traversed(G, bfs_nodes, dfs_nodes):
    # Visualize the graph and highlight visited nodes
    pos = nx.spring_layout(G)  # Layout for better visualization
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_edges(G, pos, edgelist=bfs_nodes, edge_color='b', width=2)
    nx.draw_networkx_edges(G, pos, edgelist=dfs_nodes, edge_color='r', width=2)

    # Highlight starting node
    nx.draw_networkx_nodes(G, pos, nodelist=[0], node_color='g', node_size=700)

    plt.title("Graph with built-in BFS and DFS Traversal")
    plt.show()

# lst_num_nodes = [15, 20, 25, 100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 15000, 20000, 25000, 30000]
# lst_complete_graphs_wider_shallower = []
# for num_nodes in lst_num_nodes:
#     lst_complete_graphs_wider_shallower.append(generate_wider_shallower_graph(num_nodes))
# graph_sizes, dfs_times, bfs_times = analyze_scalability(lst_complete_graphs_wider_shallower)
#
# plot_graphs(graph_sizes, dfs_times, bfs_times, "Wide Shallow Graphs")

lst_num_nodes = [15, 20, 25, 100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 15000, 20000, 25000, 30000]
# lst_num_nodes = [15, 20, 25, 100, 500, 1000, 2000, 3000, 4000, 5000]

lst_complete_graphs_deep_narrow = []
for num_nodes in lst_num_nodes:
    lst_complete_graphs_deep_narrow.append(generate_deep_narrow_graph(num_nodes))
graph_sizes, dfs_times, bfs_times = analyze_scalability(lst_complete_graphs_deep_narrow)

plot_graphs(graph_sizes, dfs_times, bfs_times, "Deep Narrow Graphs")
