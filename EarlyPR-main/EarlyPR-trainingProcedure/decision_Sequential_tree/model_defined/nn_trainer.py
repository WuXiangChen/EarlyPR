import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
class NeuralNetwork(nn.Module):
    def __init__(self, input_size=19, hidden_size=10):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(5, 7)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(7, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.criterion = nn.BCELoss()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out

    def fit(self, X_train, y_train, num_epochs=10, batch_size=32):
        X_train = torch.from_numpy(X_train.values).float()
        y_train = torch.from_numpy(y_train.values).float()
        for epoch in range(num_epochs):
            for i in range(0, X_train.shape[0], batch_size):
                inputs = X_train[i:i + batch_size]
                labels = y_train[i:i + batch_size]
                self.optimizer.zero_grad()

                outputs = self.forward(inputs)

                loss = self.criterion(outputs, labels.unsqueeze(1))
                loss.backward()
                self.optimizer.step()

    def predict(self, X_test):
        X_test = torch.from_numpy(X_test.values).float()
        with torch.no_grad():
            y_pred = self.forward(X_test).detach().numpy()
        y_pred = np.round(y_pred).flatten()
        return y_pred

    def predictall(self, X_test):
        X_test = torch.from_numpy(X_test.values).float()
        with torch.no_grad():
            y_pred = self.forward(X_test).detach().numpy()
        y_pred = np.round(y_pred).flatten()
        return y_pred

nn_trainer = NeuralNetwork()