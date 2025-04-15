import torch
import torch.nn as nn


class Attention(nn.Module):
    """Scaled Dot-Product Attention pentru imbunatatirea ponderii"""
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.scale = 1.0 / (hidden_size ** 0.5)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, lstm_out):
        attn_scores = torch.matmul(lstm_out, lstm_out.transpose(1, 2)) * self.scale
        attn_weights = self.softmax(attn_scores)
        context = torch.matmul(attn_weights, lstm_out)
        return torch.mean(context, dim=1)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3, dropout=0.5):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Strat LSTM Bidirecțional pentru a captura mai multe relatii temporale
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=True)

        # Attention Layer optimizat
        self.attention = Attention(hidden_size * 2)  # *2 pentru bidirecțional

        # Normalizare pentru stabilizare
        self.batch_norm = nn.BatchNorm1d(hidden_size * 2)

        # Straturi fully connected imbunatatite
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)

        self.leaky_relu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context = self.attention(lstm_out)
        context = self.dropout(context)

        if context.shape[0] > 1:
            context = self.batch_norm(context)

        x = self.leaky_relu(self.fc1(context))
        x = self.dropout(x)
        x = self.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x