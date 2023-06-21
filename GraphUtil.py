####################################################################################################################
# Projet - An End-to-End Deep Learning Architecture for Graph Classification
####################################################################################################################

import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE
import torch
from torch_geometric.utils import to_networkx


def dataset_overview(dataset, shuffle: bool = False, visualization: bool = False):
    """
    Vue d'ensemble du datasets
    :param dataset:
    :param shuffle: mélange aléatoire du datasets (permet de visualiser les caractéristiques d'un graph différent)
    :param visualization: Affichage du premier graph décrit
    :return:
    """
    print('==================================================================')
    print(f'Statistiques du datasets: {dataset}:')
    print('==================================================================')
    print(f'Nombre de graphes : {len(dataset)}')
    print(f'Nombre d\'attributs : {dataset.num_features}')
    print(f'Nombre de classes : {dataset.num_classes}')

    if shuffle:
        dataset = dataset.shuffle()
    data = dataset[0]  # On prend le premier graph

    print('==================================================================')
    print(f"Caractéristiques d'un graphe du datasets : {data}")
    print('==================================================================')
    print(f'Nombre de noeuds : {data.num_nodes}')
    print(f'Nombre d\'arêtes : {data.num_edges}')
    print(f'Degré moyen du graph : {data.num_edges / data.num_nodes:.2f}')
    print(f'Le graph contient-il des noeuds isolés : {data.has_isolated_nodes()}')
    print(f'Le graph contient-il des boucles : {data.has_self_loops()}')
    print(f'Le graph est-il non-orienté : {data.is_undirected()}')
    if visualization:
        graph = to_networkx(data, to_undirected=True)
        graph_visualization(graph, color=data.x, wlabels=False)


def graph_visualization(h, color, labels=None, wlabels=False, epoch=None, loss=None):
    """
    Visualisation de graphe
    :param h:
    :param color:
    :param labels:
    :param wlabels:
    :param epoch:
    :param loss:
    :return:
    """
    plt.figure(figsize=(7, 7))
    plt.xticks([])
    plt.yticks([])

    if torch.is_tensor(h):
        h = h.detach().cpu().numpy()
        plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
        if epoch is not None and loss is not None:
            plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
    else:
        nx.draw_networkx(h, pos=nx.spring_layout(h, seed=42), labels=labels, with_labels=wlabels,
                         node_color=color)
    plt.show()


def embeddings_visualization(h, color, title: str = None):
    """
    Visualisation des embeddings
    :param h:
    :param color:
    :param title:
    :return:
    """
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    plt.figure(figsize=(10, 10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color)
    if title:
        plt.title(title)
    plt.show()
