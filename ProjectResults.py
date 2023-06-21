####################################################################################################################
# Projet - An End-to-End Deep Learning Architecture for Graph Classification
####################################################################################################################

from torch_geometric.datasets import TUDataset
from GraphUtil import dataset_overview
from DGCNN_Model import DGCNN
from BasicGraphModels import GCN, GNN
from TrainAndTestManager import TrainAndTestManager

# Liste des datasets à étudier
dataset_list = ["IMDB-BINARY", "PROTEINS"]


for dataset_name in dataset_list:
    # Récupération du datasets
    dataset = TUDataset(root='datasets', name=dataset_name)

    # Vue d'ensemble du datasets
    dataset_overview(dataset, shuffle=True, visualization=True)

    models_list = [DGCNN]  # GCN, GNN
    learning_rate = [0.0005, 0.0001, 0.00005, 0.00001]

    for model in models_list:
        for lr in learning_rate:
            PROTEINS_TrainAndTest = TrainAndTestManager(dataset, model, learning_rate=lr)
            PROTEINS_TrainAndTest.apply()


