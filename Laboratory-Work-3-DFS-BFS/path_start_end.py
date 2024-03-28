import random
import time
import matplotlib.pyplot as plt
import networkx as nx


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
    graph = {}
    for i in range(num_nodes):
        graph[i] = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if random.random() < 1.1:  # Complete Undirected Graph
            # if random.random() < 0.5:  # Incomplete Undirected Graph
                graph[i].append(j)
                graph[j].append(i)
    return graph


# Function to plot an undirected graph
def plot_graph(graph, num_nodes):
    G = nx.Graph()
    fig, ax = plt.subplots()
    pos = nx.spring_layout(G)
    for node, neighbors in graph.items():
        for neighbor in neighbors:
            G.add_edge(node, neighbor)
    nx.draw(G, with_labels=True)
    plt.title(f"Undirected Complete Graph with {num_nodes} nodes")
    limits = plt.axis('on')  # turns on axis
    ax.tick_params(left=True, bottom=True)
    plt.show()


# Main function to measure execution times, plot the constructed graph, and plot the results
if __name__ == "__main__":
    num_nodes_list = [i for i in range(25, 101, 25)] + [i for i in range(100, 1000, 50)]

    dfs_times = []
    bfs_times = []

    for num_nodes in num_nodes_list:
        print(f"\nNumber of Nodes: {num_nodes}")
        # Generate random undirected graph
        graph = generate_random_graph(num_nodes)

        start_node = 0

        # Plot the constructed undirected graph
        if num_nodes <= 100:
            print("Plotting the constructed undirected graph...")
            plot_graph(graph, num_nodes)
        # if num_nodes <= 25:
        #     end_node = int(input("Input End Node: "))
        # else:
        #     end_node = 1
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
    plt.title('Implemented DFS and BFS on Undirected Complete Graph with Start and End Node Far from Each Other')
    plt.legend()
    plt.grid(True)
    plt.show()
