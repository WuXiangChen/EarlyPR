import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class LSTMModel(nn.Module):
    def __init__(self, input_size=19, hidden_size=8):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.criterion = nn.BCELoss()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out

    def fit(self, X_train, y_train, num_epochs=10, batch_size=256):
        X_train = torch.from_numpy(X_train.values).float()
        y_train = torch.from_numpy(y_train.values).float()
        for epoch in range(num_epochs):
            for i in range(0, X_train.shape[0], batch_size):
                inputs = X_train[i:i + batch_size].unsqueeze(1)
                labels = y_train[i:i + batch_size]
                self.optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = self.criterion(outputs, labels.unsqueeze(1))
                loss.backward()
                self.optimizer.step()

    def predict(self, X_test):
        X_test = torch.from_numpy(X_test.values).unsqueeze(1).float()
        with torch.no_grad():
            y_pred = self.forward(X_test).detach().numpy()
        y_pred = np.round(y_pred).flatten()
        return y_pred

    def predictall(self, X_test):
        X_test = torch.from_numpy(X_test.values).unsqueeze(1).float()
        with torch.no_grad():
            y_pred = self.forward(X_test).detach().numpy()
        y_pred = np.round(y_pred).flatten().astype(np.int32)
        return y_pred

lstm_trainer = LSTMModel()