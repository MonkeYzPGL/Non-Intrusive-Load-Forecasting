import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    """Mecanism de Atenție pentru LSTM."""
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, lstm_out):
        attn_weights = F.softmax(self.v(torch.tanh(self.attn(lstm_out))), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        return context

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)

        self.attention = Attention(hidden_size)

        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)

        self.leaky_relu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)

        context = self.attention(lstm_out)

        x = self.leaky_relu(self.fc1(context))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
