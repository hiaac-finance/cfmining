import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin
from tqdm import tqdm

class MLPClassifier(BaseEstimator, ClassifierMixin):
    """MLPClassifier in the Sklearn API using PyTorch.
    It mimics the MLPClassifier from Sklearn, but it uses PyTorch to train the model.
    The extra functionalities are the possibility to use class weights and sample weights.

    Parameters
    ----------
    hidden_layer_sizes : tuple, optional
            List of hidden layer sizes as a tuple with has n_layers-2 elements, by default (100,)
        batch_size : int, optional
            Size of batch for training, by default 32
        learning_rate_init : float, optional
            Initial learning rate, by default 0.1
        learning_rate_decay_rate : float, optional
            Decay rate of learning rate, equal to 1 to constant learning rate, by default 0.1
        alpha : float, optional
            Weight of L2 regularization, by default 0.0001
        epochs : int, optional
            Number of epochs to train model, by default 100
        class_weight : string, optional
            If want to use class weights in the loss, pass the value "balanced", by default None
        random_state : int, optional
            Random seed, by default None
    """

    def __init__(
        self,
        hidden_layer_sizes=(100,),
        batch_size=32,
        learning_rate_init=0.001,
        epochs=100,
        class_weight=None,
        random_state=None,
        device = None
    ):
        self._random_state = random_state
        self._seed_everything(random_state)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.batch_size = batch_size
        self.learning_rate_init = learning_rate_init
        self.epochs = epochs
        self.class_weight = class_weight
        self.device = device

    @property
    def random_state(self):
        return self._random_state

    @random_state.setter
    def random_state(self, value):
        self._random_state = value
        self._seed_everything(value)

    def _seed_everything(self, value):
        if value is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

    def set_model(self, input_dim):
        layers = []
        prev_size = input_dim
        for layer_size in self.hidden_layer_sizes:
            layers.append(nn.Linear(prev_size, layer_size))
            layers.append(nn.ReLU())
            prev_size = layer_size
        layers.append(nn.Linear(prev_size, 2))
        layers.append(nn.Softmax(dim = 1))
        model = nn.Sequential(*layers)
        return model

    def fit(self, X, y):
        if self.class_weight == "balanced":
            class_counts = np.bincount(y)
            class_weights = torch.tensor([1 / class_counts[i] for i in range(len(class_counts))], dtype=torch.float)
            if not self.device is None:
                class_weights = class_weights.to(self.device)
        else:
            class_weights = torch.tensor([1.0, 1.0])

        self.model = self.set_model(X.shape[1])

        if type(X) == pd.DataFrame:
            X = X.values
        if type(y) == pd.Series:
            y = y.values

        criterion = nn.CrossEntropyLoss(weight = class_weights)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate_init)

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(np.stack([1 - y,y]).T, dtype=torch.float32)

        if not self.device is None:
           self.model = self.model.to(self.device)
           X_tensor = X_tensor.to(self.device)
           y_tensor = y_tensor.to(self.device)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size)

        for epoch in range(self.epochs):
            self.model.train()
            for inputs, labels in dataloader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

    def predict_proba(self, X):
        self.model.eval()
        if type(X) == pd.DataFrame:
            X = X.values
        X_tensor = torch.tensor(X, dtype=torch.float32)
        if not self.device is None:
            X_tensor = X_tensor.to(self.device)
        with torch.no_grad():
            prob = self.model(X_tensor).cpu().numpy()       
        return prob

    def predict(self, X):
        if type(X) == pd.DataFrame:
            X = X.values
        prob = self.predict_proba(X)
        return prob[:, 1] > 0.5

    def score(self, X, y):
        if type(X) == pd.DataFrame:
            X = X.values
        prob = self.predict_proba(X)
        return prob[:, 1]
