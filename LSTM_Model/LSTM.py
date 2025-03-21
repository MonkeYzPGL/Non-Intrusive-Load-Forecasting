import torch
import torch.nn as nn


class Attention(nn.Module):
    """Scaled Dot-Product Attention pentru îmbunătățirea ponderii"""
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.scale = 1.0 / (hidden_size ** 0.5)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, lstm_out):
        attn_scores = torch.matmul(lstm_out, lstm_out.transpose(1, 2)) * self.scale
        attn_weights = self.softmax(attn_scores)
        context = torch.matmul(attn_weights, lstm_out)
        return context[:, -1, :]  # Luăm ultima valoare din secvență


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Strat LSTM Bidirectional pentru a captura mai multe relații temporale
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout, bidirectional=True)

        # Layer de atenție optimizat
        self.attention = Attention(hidden_size * 2)  # *2 pentru bidirecțional

        # Normalizare pentru stabilizare
        self.batch_norm = nn.BatchNorm1d(hidden_size * 2)

        # Strat fully connected final pentru predicția pe toate canalele
        self.fc = nn.Linear(hidden_size * 2, output_size)

        self.leaky_relu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context = self.attention(lstm_out)

        # Aplicam Batch Norm doar dacă dimensiunea permite
        if context.shape[0] > 1:
            context = self.batch_norm(context)

        x = self.fc(context)  # Predictie pentru toate canalele
        return x
