import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=(dropout if num_layers > 1 else 0))
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Output LSTM și ultima stare ascunsă
        lstm_out, _ = self.lstm(x)
        # Doar ultima ieșire pentru previziune
        out = self.fc(lstm_out[:, -1, :])
        return out


class LSTMTrainer:
    def __init__(self, model, device, learning_rate=0.001):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def train(self, train_loader, val_loader, epochs=10):
        self.model.train()
        print(f"Training on device: {self.device}")
        print(f"Model is on device: {next(self.model.parameters()).device}")

        for epoch in range(epochs):
            train_loss = 0.0
            for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                if epoch == 0 and batch_idx == 0:
                    print(f"X_batch is on device: {X_batch.device}, y_batch is on device: {y_batch.device}")

                self.optimizer.zero_grad()
                y_pred = self.model(X_batch)
                loss = self.criterion(y_pred.squeeze(-1), y_batch)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            val_loss = self.evaluate(val_loader)
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss / len(train_loader):.4f}, "
                  f"Val Loss: {val_loss:.4f}")

            # GPU Memory Usage
            if self.device.type == 'cuda':
                print(f"CUDA Memory Allocated: {torch.cuda.memory_allocated(self.device)} bytes")
                print(f"CUDA Memory Cached: {torch.cuda.memory_reserved(self.device)} bytes")

    def evaluate(self, val_loader):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                y_pred = self.model(X_batch)
                loss = self.criterion(y_pred.squeeze(-1), y_batch)
                val_loss += loss.item()
        return val_loss / len(val_loader)

    def predict(self, test_loader):
        self.model.eval()
        predictions = []
        actuals = []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                y_pred = self.model(X_batch)
                predictions.extend(y_pred.cpu().numpy())
                actuals.extend(y_batch.cpu().numpy())
        return predictions, actuals

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)
        print(f"Model loaded from {path}")


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.clone().detach().float()
        self.y = y.clone().detach().float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
