####################################################################################################################
# Projet - An End-to-End Deep Learning Architecture for Graph Classification
####################################################################################################################

import torch
from torch.nn import Linear
import torch.nn.functional as torch_fct
from torch_geometric.nn import GCNConv, global_mean_pool, GraphConv


class GCN(torch.nn.Module):
    """
    Modèle GCN du TP du cours
    - Création de l'embedding des noeuds : 3 couches GCNConv + activation  ReLU(𝑥)=max(𝑥,0)
    - Création de l'embedding de graph : readout de type moyenne
    - Classification du graph : classifieur linéaire avec dropout
    """
    def __init__(self, dataset, hidden_channels):
        """
        Initialisation des blocs
        :param dataset:
        :param hidden_channels:
        """
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.name = "GCN"

        self.graph_conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        self.graph_conv2 = GCNConv(hidden_channels, hidden_channels)
        self.graph_conv3 = GCNConv(hidden_channels, hidden_channels)

        self.classifier = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        """
        Flux d'execution du réseau
        On empile 3 couches de convolution : on agrège l'information de voisinage jusqu'à la distance 3
        :param x:
        :param edge_index:
        :param batch:
        :return:
        """
        # 1. Calcule des embeddings de chaque noeuds, avec plusieurs étapes de passage de message
        x = self.graph_conv1(x, edge_index)
        x = x.relu()
        x = self.graph_conv2(x, edge_index)
        x = x.relu()
        x = self.graph_conv3(x, edge_index)

        # 2. Readout : Agrégation des embeddings de noeuds en un embedding de graphe unifié
        # Utilisation de la moyenne des embeddings
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Classifieur sur les embeddings de graphes
        # Pour éviter le surapprentissage:
        # certains éléments du tenseur sont aléatoirement mis à 0 avec la fonction de dropout
        x = torch_fct.dropout(x, p=0.5, training=self.training)
        x = self.classifier(x)

        return x


class GNN(torch.nn.Module):
    """
    Modèle GNN du TP du cours
    C'est le modèle GCN avec des couches GraphConv au lieu de GCNConv introduisant des raccourcis (skip-connection)
    """
    def __init__(self, dataset, hidden_channels):
        super(GNN, self).__init__()
        torch.manual_seed(12345)
        self.name = "GNN"

        self.graph_conv1 = GraphConv(dataset.num_node_features, hidden_channels)
        self.graph_conv2 = GraphConv(hidden_channels, hidden_channels)
        self.graph_conv3 = GraphConv(hidden_channels, hidden_channels)

        self.classifier = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        x = self.graph_conv1(x, edge_index)
        x = x.relu()
        x = self.graph_conv2(x, edge_index)
        x = x.relu()
        x = self.graph_conv3(x, edge_index)

        x = global_mean_pool(x, batch)

        x = torch_fct.dropout(x, p=0.5, training=self.training)
        x = self.classifier(x)

        return x
