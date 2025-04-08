import torch
from torch import nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, lstm_output):
        attn_weights = self.attn(lstm_output)               # [batch, seq_len, 1]
        attn_weights = F.softmax(attn_weights, dim=1)
        context = torch.sum(attn_weights * lstm_output, dim=1)  # [batch, hidden*2]
        return context


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=256, output_size=1, num_layers=2, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        self.attention = Attention(hidden_size * 2)
        self.fc1 = nn.Linear(hidden_size * 2, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context = self.attention(lstm_out)
        x = self.fc1(context)
        x = self.bn1(x)
        x = self.dropout1(F.relu(x))
        return self.fc2(x)

