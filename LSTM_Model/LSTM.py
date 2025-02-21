import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    """Mecanism de Atenție pentru LSTM."""
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        attn_weights = torch.tanh(self.attn(encoder_outputs))
        attn_weights = self.v(attn_weights).squeeze(2)
        attn_weights = F.softmax(attn_weights, dim=1)
        context = torch.sum(attn_weights.unsqueeze(2) * encoder_outputs, dim=1)
        return context

class LSTMModel(nn.Module):
    def __init__(self, input_size=9, hidden_size=256, output_size=1, num_layers=3, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.activation = nn.Identity()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context = self.attention(lstm_out, lstm_out)
        out = self.fc(context)
        return self.activation(out)