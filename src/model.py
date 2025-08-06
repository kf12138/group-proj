# src/model.py
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class ConvNet(nn.Module):
    """
    增强版卷积网络：2 层卷积 + 池化 + 2 层全连接
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)  # 28→14
        self.fc1   = nn.Linear(14*14*64, 128)
        self.fc2   = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))      # [B,32,28,28]
        x = self.pool(F.relu(self.conv2(x)))  # [B,64,14,14]
        x = x.view(-1, 14*14*64)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# 方便外部 import
_models = {
    'mlp': MLP,
    'conv': ConvNet
}

def get_model(name: str):
    return _models[name]
