import networkx as nx
import matplotlib.pyplot as plt
import random

def generate_graph(nodes, sparse=True):
    if sparse:
        edges = nodes - 5
    else:
        edges = nodes * (nodes - 1) // 4  # A more dense graph
    G = nx.gnm_random_graph(nodes, edges)

    for (u, v) in G.edges():
        G.edges[u, v]['weight'] = random.randint(1, 10)  # Assign a random weight
    return G

def visualize_graph(graph):
    pos = nx.spring_layout(graph)  # Positions for all nodes
    weights = nx.get_edge_attributes(graph, 'weight')  # Get edge weights

    # Draw nodes and edges
    nx.draw(graph, pos, with_labels=True, node_color='skyblue', node_size=700)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=weights)

    # Show the plot
    plt.show()

# Example usage
graph = generate_graph(75, sparse=True)  # Generate a graph with 10 nodes
visualize_graph(graph)  # Visualize the generated graph
