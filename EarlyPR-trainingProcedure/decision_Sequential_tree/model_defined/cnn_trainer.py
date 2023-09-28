import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt


class ResNetBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += identity
        out = self.relu(out)
        return out

class ConvolutionalNetwork(nn.Module):
    def __init__(self, input_size=19, num_resnet_blocks=2):
        super(ConvolutionalNetwork, self).__init__()
        self.input_size = input_size
        self.criterion = nn.BCELoss()
        self.conv1 = nn.Conv1d(self.input_size, 8, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(8, 12)
        # 实例化卷积结构
        self.resnet_blocks = nn.ModuleList([ResNetBlock(8) for _ in range(num_resnet_blocks)])
        self.fc2 = nn.Linear(12, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.sigmoid = nn.Sigmoid()
        self.losses = []

    def forward(self, x):
        out = x.permute(0, 2, 1)
        out = self.conv1(out)
        out = self.relu(out)
        for resnet_block in self.resnet_blocks:
            out = resnet_block(out)
        out = self.pool(out)
        out = self.flatten(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

    def fit(self, X_train, y_train, num_epochs=10, batch_size=128):
        # 输入数据是DataFrame格式的，我需要将其numpy化
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
                self.losses.append(loss.item())

    def predict(self, X_test):
        X_test = X_test.values
        X_test = torch.from_numpy(X_test).float().unsqueeze(1)
        with torch.no_grad():
            y_pred = self.forward(X_test).detach().numpy()
        y_pred = np.array([int(np.round(y_pred).flatten()[0])])
        return y_pred

    # tmp 非主要方法
    def predictall(self, X_test):
        X_test = X_test.values
        X_test = torch.from_numpy(X_test).float().unsqueeze(1)
        with torch.no_grad():
            y_pred = self.forward(X_test).detach().numpy()
        y_pred = np.array(np.round(y_pred).reshape(-1))
        return y_pred

    def plot_loss_curve(self):
        plt.plot(self.losses)
        plt.xlabel('Batch Iteration')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.show()

cnn_trainer = ConvolutionalNetwork()
cnn_trainer_mergedPR = ConvolutionalNetwork()