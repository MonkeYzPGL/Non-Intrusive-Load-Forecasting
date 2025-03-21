# KAN.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class InterpolationLayer(nn.Module):
    """
    Simuleaza o funcție f(x) invatabila per feature, similar cu KAN.
    Foloseste o mică rețea pentru a aproxima comportamentul neliniar.
    """
    def __init__(self, input_dim, hidden_dim=16):
        super(InterpolationLayer, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # x shape: (batch, features)
        x = torch.sin(self.fc1(x))  # funcție neliniara smooth, asemanatoare unei interpolari
        x = self.fc2(x)
        return x


class KANModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        super(KANModel, self).__init__()
        self.interpolation = InterpolationLayer(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.interpolation(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
