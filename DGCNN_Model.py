####################################################################################################################
# Projet - An End-to-End Deep Learning Architecture for Graph Classification
####################################################################################################################

import torch
from torch.nn import Linear, Conv1d, MaxPool1d
from torch.nn.functional import dropout, log_softmax
from torch_geometric.nn import GCNConv, Linear
from torch_geometric.nn.aggr import SortAggregation


class DGCNN(torch.nn.Module):
    """
    Modèle DGCNN de la publication :
    An End-to-End Deep Learning Architecture for Graph Classification
    - Création de l'embedding des noeuds : 4 couches GCNConv + activation  ReLU(𝑥)=max(𝑥,0)
    - Création de l'embedding de graph avec la couche de sort pooling
    - Couches de convolution 1D
    - Classification du graph : classifieur linéaire avec dropout
    """
    def __init__(self, dataset, hidden_channels=32):
        """
        Initialisation des blocs de réseaux
        :param dataset:
        :param hidden_channels:
        """
        super(DGCNN, self).__init__()
        torch.manual_seed(12345)

        self.name = "DGCNN"
        self.dataset = dataset

        # Définition du paramètre de la couche de SortPooling en fonction du datasets
        self.k = self.define_sort_pooling_k()

        # Définition des couches de convolutoin de graphes
        self.graph_conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        self.graph_conv2 = GCNConv(hidden_channels, hidden_channels)
        self.graph_conv3 = GCNConv(hidden_channels, hidden_channels)
        self.graph_conv4 = GCNConv(hidden_channels, 1)

        # Dimension de la concaténation
        self.total_dim = hidden_channels * 3 + 1

        # Défintion des couches de convolution 1D
        self.conv1 = Conv1d(1, 16, self.total_dim, self.total_dim)
        self.conv2 = Conv1d(16, 32, 5, 1)
        self.max_pool = MaxPool1d(kernel_size=2, stride=2)

        # Calcul de la dimension d'entrée des couches denses
        dense_dim = int((self.k - 2) / 2 + 1)
        self.dense_dim = (dense_dim - 5 + 1) * 32

        # Défintion des couches denses
        self.classifier1 = Linear(self.dense_dim, 128)
        self.classifier2 = Linear(128, 2)

    def define_sort_pooling_k(self):
        """
        le k de SortPooling est fixé de telle sorte que XX% des graphiques ont des noeuds de plus de k
        - 60% pour les données avec labels
        - 90% pour les données sans labels comme les datasets de réseaux sociaux
        :param ratio:
        :return:
        """
        ratio = 0.6
        if self.dataset.name == "IMDB-BINARY":
            ratio = 0.9
        num_nodes = sorted([data.num_nodes for data in self.dataset])
        k = num_nodes[int(round(ratio * len(num_nodes))) - 1]
        k = int(max(10, k))
        print(f'La valeur de k utilisée pour le sortpooling est {k}')
        return k

    def forward(self, x, edge_index, batch):
        """
        Flux d'execution du réseau
        :param x:
        :param edge_index:
        :param batch:
        :return:
        """
        # 1. Calcul des embeddings de chaque noeuds, avec plusieurs étapes de passage de message
        x_1 = self.graph_conv1(x, edge_index)
        x_1 = x_1.tanh()
        x_2 = self.graph_conv2(x_1, edge_index)
        x_2 = x_2.tanh()
        x_3 = self.graph_conv3(x_2, edge_index)
        x_3 = x_3.tanh()
        x_4 = self.graph_conv4(x_3, edge_index)
        x_4 = x_4.tanh()

        # 2. Readout : SortPooling
        x = torch.cat([x_1, x_2, x_3, x_4], 1)
        x = SortAggregation(k=self.k).forward(x, index=batch)  # dimension [num_graphs, k * hidden]

        # 3. Classifieur sur les embeddings de graphes
        # a. Couches de convolution 1D
        x = x.unsqueeze(1)  # dimension [num_graphs, 1, k * hidden]
        x = self.conv1(x)
        x = x.relu()
        x = self.max_pool(x)
        x = self.conv2(x)
        x = x.relu()
        # b. Couches denses
        x = x.view(len(x), -1)  # Applatissement des données
        x = self.classifier1(x)
        x = x.relu()
        x = dropout(x, p=0.5, training=self.training)
        x = self.classifier2(x)
        x = log_softmax(x, dim=-1)

        return x
