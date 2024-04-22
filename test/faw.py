import networkx as nx
import random
import matplotlib.pyplot as plt



# Visualize the graph
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10)
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.title('Random Sparse Weighted Graph')
plt.show()

dict_G = nx.to_dict_of_dicts(G)
for i, j in dict_G.items():
    for k in j:
        weight = dict_G[i][k]['weight']
        dict_G[i][k] = None
        dict_G[i][k] = weight

print(dict_G)