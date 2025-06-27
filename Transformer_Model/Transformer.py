import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model=128, nhead=4, num_layers=3, dropout=0.1, output_size=24, window_size=168):
        super(TransformerModel, self).__init__()
        self.input_size = input_size
        self.window_size = window_size
        self.output_size = output_size
        self.d_model = d_model

        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = self._generate_positional_encoding(window_size, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_layer = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),  # activare interna robusta
            nn.Dropout(dropout),
            nn.Linear(128, output_size),
            nn.Sigmoid()  # activare finala – output ∈ [0, 1]
        )

    def _generate_positional_encoding(self, seq_len, d_model):
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # shape: (1, seq_len, d_model)

    def forward(self, x):
        # x shape: (batch_size, window_size, input_size)
        batch_size = x.size(0)
        x = self.input_projection(x)  # (B, T, d_model)
        pe = self.positional_encoding[:, :x.size(1), :].to(x.device)
        x = x + pe * 0.1
        x = self.transformer_encoder(x)  # (B, T, d_model)
        x = x[:, -1, :]
        out = self.output_layer(x)
        return out
