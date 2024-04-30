import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch_geometric.utils import to_networkx
import networkx as nx

def tsne(h, color):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())
    color = color.detach().cpu().numpy()

    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.show()

def plot_cora(data, labels):
    G = to_networkx(data, to_undirected=True)
    color_map = []

    for i, label in enumerate(labels):
        color_map.append(label)

    plt.figure(figsize=(12, 8))
    nx.draw(G, node_color=color_map, cmap=plt.get_cmap('Set1'), node_size=20, linewidths=0.5, width=0.5)
    plt.show()