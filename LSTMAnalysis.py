import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class EnergyDataset(Dataset):
    """Dataset pentru datele energetice pentru LSTM."""
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + self.seq_length]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class LSTMModel(nn.Module):
    """Modelul LSTM."""
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

class LSTMAnalyzer:
    """Clasă pentru implementarea și antrenarea LSTM."""
    def __init__(self, data, seq_length=10, batch_size=64, hidden_size=50, num_layers=2, learning_rate=0.001, epochs=20, device=None):
        self.data = data
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LSTMModel(hidden_size=hidden_size, num_layers=num_layers).to(self.device)
        print(f"Using device: {torch.cuda.get_device_name(0)}")

    def preprocess_data(self):
        """Preprocesare date pentru antrenament."""
        self.mean = np.mean(self.data)
        self.std = np.std(self.data)
        self.data = (self.data - self.mean) / self.std  # Normalizare
        dataset = EnergyDataset(self.data, self.seq_length)
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def train(self):
        """Antrenarea modelului LSTM."""
        self.preprocess_data()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_history = []  # Lista pentru a stoca pierderea

        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0
            for x, y in self.dataloader:
                x, y = x.to(self.device), y.to(self.device)
                x = x.unsqueeze(-1)  # Adăugăm dimensiunea de intrare
                optimizer.zero_grad()
                predictions = self.model(x)
                loss = criterion(predictions, y.unsqueeze(-1))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(self.dataloader)
            self.loss_history.append(avg_loss)
            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}")

    def plot_loss(self):
        """Plotarea pierderii (loss) pentru fiecare epocă."""
        if not hasattr(self, 'loss_history') or not self.loss_history:
            print("Loss history is empty. Train the model first.")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.loss_history) + 1), self.loss_history, marker='o', label='Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss over Epochs')
        plt.legend()
        plt.grid()
        plt.show()

    def denormalize(self, value):
        """Denormalizează valorile folosind media și abaterea standard a datelor."""
        return value * self.std + self.mean

    def predict(self, data):
        """Realizare predicții pe baza datelor de intrare."""
        self.model.eval()
        with torch.no_grad():
            data = (data - self.mean) / self.std  # Normalizare
            input_seq = torch.tensor(data[-self.seq_length:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(self.device)
            prediction = self.model(input_seq)
        return self.denormalize(prediction.item())

    def predict_next_day(self):
        """Realizare predicții pentru următoarele 24 de ore (valori la 1 oră)."""
        self.model.eval()
        predictions = []
        input_seq = self.data[-self.seq_length:]  # Ultima secvență de date
        with torch.no_grad():
            for _ in range(24):  # 24 de valori pentru o zi
                input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(self.device)
                pred = self.model(input_tensor).item()
                pred = self.denormalize(pred)  # Denormalizare
                predictions.append(pred)
                input_seq = np.append(input_seq[1:], (pred - self.mean) / self.std)  # Actualizăm secvența de intrare

        return predictions
