from torch import nn

class LSTMDecomposer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)  #output_size = nr. aparate
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.norm(out)
        out = self.fc(out)
        out = self.activation(out)
        return out
