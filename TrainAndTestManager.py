####################################################################################################################
# Projet - An End-to-End Deep Learning Architecture for Graph Classification
####################################################################################################################

import matplotlib.pyplot as plt
import torch
from GraphUtil import embeddings_visualization
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree


def input_features_matrix(i_data):
    """
    Définit la matrice d'entrée du modèle
    data.x est la matrice des caractéristiques des sommets du graph
    Dans le cas où le datasets ne présente pas de caractéristiques pour les sommets
    X peut être définie comme un vecteur contenant les degrés des sommets normalisés.
    :param i_data:
    :return:
    """
    if i_data.x is not None:
        output = i_data.x
    else:
        deg = degree(i_data.edge_index[1], i_data.num_nodes)
        deg = deg / deg.mean()
        deg = deg.view(-1, 1)
        output = deg
    return output


class TrainAndTestManager:
    """
    Le TrainAndTestManager permet d'entraînner et de tester un modèle de classification de graph,
    sur un datasets donnés en entrée
    """
    def __init__(self, dataset, model, hidden_channels=64, learning_rate=0.0001):
        self.test_loader = None
        self.train_loader = None
        self.dataset = dataset
        self.model = model(self.dataset, hidden_channels=hidden_channels)
        print(self.model)

        # Mise en forme des données
        self.train_loader: DataLoader
        self.test_loader: DataLoader
        self.data_pre_processing()

        # Utilisation de l'optimiseur stochastique Adam
        # paramètres testés :
        # lr=0.01 --> instabilités sur l'apprentissage
        # lr=0.0001 --> OK
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # Fonction de perte utilisée pour évaluer le modèle entropie croisée
        self.loss_criterion = torch.nn.CrossEntropyLoss()

    def data_pre_processing(self):
        """
        Mise en forme des données
        :return:
        """
        # Partitionnement du datasets pour l'entrainement
        torch.manual_seed(12345)  # pour la reproductibilité des résultats
        dataset = self.dataset.shuffle()

        # On consacre 66% des données à l'apprentissage et 33% au test
        train_len = round(len(dataset)*0.66)
        train_dataset = dataset[:train_len]
        test_dataset = dataset[train_len+1:]

        # Traitement par mini-batchs
        self.train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=50, shuffle=False)

        for step, data in enumerate(self.train_loader):
            print(f'Étape {step + 1}:')
            print('=======')
            print(f'Nombre de graphes dans ce batch : {data.num_graphs}')
            print(data)
            print()
        print(f'Nombre de graphes pour l\'apprentissage: {len(train_dataset)}')
        print(f'Nombre de graphes pour le test : {len(test_dataset)}')

    def batch_training(self, visualization_title: str = None):
        """
        Entrainnement du modèle par batch
        :param visualization_title:
        :return:
        """
        self.model.train()
        output_viz = []
        y_viz = []

        # Itération sur les batches
        for i, i_data in enumerate(self.train_loader):
            vect = input_features_matrix(i_data)
            out = self.model(vect, i_data.edge_index, i_data.batch)  # Perform a single forward pass.
            loss = self.loss_criterion(out, i_data.y)  # Compute the loss.
            loss.backward()  # Derive gradients.
            self.optimizer.step()  # Update parameters based on gradients.
            self.optimizer.zero_grad()  # Clear gradients.

            if visualization_title:
                output_viz.append(out)
                y_viz.append(i_data.y)

        if visualization_title:
            output_viz = torch.cat(output_viz)
            y_viz = torch.cat(y_viz)
            embeddings_visualization(output_viz, color=y_viz, title=visualization_title)

    def test(self, loader: DataLoader):
        """
        Evaluation du modèle
        :param loader:
        :return:
        """
        self.model.eval()

        correct = 0
        # Itération sur les batches
        for i_data in loader:
            vect = input_features_matrix(i_data)
            out = self.model(vect, i_data.edge_index, i_data.batch)
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            correct += int((pred == i_data.y).sum())  # Check against ground-truth labels.
        return correct / len(loader.dataset)  # Derive ratio of correct predictions.

    def apply(self, max_iter: int = 200):

        train_acc_viz = []
        test_acc_viz = []
        for epoch in range(0, max_iter):
            tsne_visualization = None
            if epoch == 0 or (epoch+1) % 50 == 0:
                tsne_visualization = f"Entraînnement {self.model.name} - Epoch {epoch+1}"
            self.batch_training(tsne_visualization)
            train_acc = self.test(self.train_loader)
            test_acc = self.test(self.test_loader)
            train_acc_viz.append(train_acc)
            test_acc_viz.append(test_acc)
            print(f'Epoch: {epoch:03d}, Learning accuracy: {train_acc*100:.1f}%, Test accuracy: {test_acc*100:.1f}%')

        plt.plot(train_acc_viz, label="Training")
        plt.plot(test_acc_viz, label="Test")
        plt.legend(loc='lower right')
        plt.title(f"Accuracy - Dataset: {self.dataset.name} - Modèle: {self.model.name} - lr: {self.learning_rate} ")
        plt.show()
