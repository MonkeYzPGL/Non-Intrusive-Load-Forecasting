import os
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from LSTM_Model.LSTM import LSTMModel


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMAnalyzer:
    def __init__(self, csv_path, window_size=168, batch_size=64, hidden_size=256, learning_rate=0.001):
        self.csv_path = csv_path
        self.window_size = window_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        #  Preprocesam datele inainte de a initializa modelul
        self.preprocess_data()

        #  Dupa preprocesare, definim selected_features
        self.model = LSTMModel(
            input_size=len(self.selected_features),
            hidden_size=256,
            output_size=1,
            num_layers=2,
            dropout=0.3
        ).to(self.device)

        #  Functia de cost si optimizer
        self.criterion = nn.SmoothL1Loss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.9, patience=3, min_lr=5e-5)

    def preprocess_data(self):
        # Citirea datelor
        data = pd.read_csv(self.csv_path)
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data.set_index('timestamp', inplace=True)
        data['day_of_week'] = data.index.dayofweek
        data['hour_of_day'] = data.index.hour
        data["is_weekend"] = data.index.dayofweek >= 5  # 1 daca e sambata/duminica
        data["month"] = data.index.month
        data["season"] = data["month"] % 12 // 3  # 0: iarna, 1: primavara, 2: vara, 3: toamna

        data['power'] = pd.to_numeric(data['power'], errors='coerce').fillna(0)  # Conversie în numeric

        # Creare caracteristici suplimentare
        data["hour_sin"] = np.sin(2 * np.pi * data["hour_of_day"] / 24)
        data["hour_cos"] = np.cos(2 * np.pi * data["hour_of_day"] / 24)
        data["day_sin"] = np.sin(2 * np.pi * data["day_of_week"] / 7)
        data["day_cos"] = np.cos(2 * np.pi * data["day_of_week"] / 7)

        # Aplicare lag-uri
        lag_features = {
            f'lag_{lag}h': data['power'].shift(lag)
            for lag in range(1, self.window_size + 1)
        }
        data = pd.concat([data, pd.DataFrame(lag_features, index=data.index)], axis=1)

        # Creare caracteristici temporale avansate
        data['delta_power'] = data['power'].diff().shift(1)
        data['rolling_mean_12h'] = data['power'].rolling('12h').mean().shift(1)
        data['rolling_std_12h'] = data['power'].rolling('12h').std().shift(1)
        data['rolling_max_12h'] = data['power'].rolling('12h').max().shift(1)
        data['rolling_mean_24h'] = data['power'].rolling('24h').mean().shift(1)
        data['rolling_max_24h'] = data['power'].rolling('24h').max().shift(1)
        data['rolling_median_12h'] = data['power'].rolling('12h').median().shift(1)

        # Umplem valorile lipsa
        data = data.interpolate(method='linear', limit_direction='both')

        # Impartirea datelor inainte de scalare
        train_size = int(0.8 * len(data))
        val_size = int(0.1 * len(data))

        train_data = data.iloc[:train_size]
        val_data = data.iloc[train_size:train_size + val_size]
        test_data = data.iloc[train_size + val_size:]

        # Selectam caracteristicile pentru scalare
        self.selected_features = [
            'power', 'delta_power',
            'day_of_week', 'hour_of_day', 'is_weekend', 'month', 'season',
            'lag_1h', 'lag_6h', 'lag_24h', 'lag_168h',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'rolling_mean_12h', 'rolling_std_12h', 'rolling_max_12h',
            'rolling_mean_24h', 'rolling_max_24h'
        ]

        #   ,'rolling_mean_12h', 'rolling_std_12h', 'rolling_max_12h', 'rolling_mean_24h',
                              #  'rolling_median_12h']


        # Aplicăm scalarea DOAR pe setul de train
        self.scaler = MinMaxScaler(feature_range=(0,10))
        self.scaler.fit(train_data[self.selected_features])  # Se antreneaza scaler-ul doar pe train

        # Transformăm fiecare subset folosind scaler-ul antrenat pe train
        train_scaled = self.scaler.transform(train_data[self.selected_features])
        val_scaled = self.scaler.transform(val_data[self.selected_features])
        test_scaled = self.scaler.transform(test_data[self.selected_features])

        # Aplicăm create_sequences separat pentru fiecare subset
        X_train, y_train = self.create_sequences(train_scaled)
        X_val, y_val = self.create_sequences(val_scaled)
        X_test, y_test = self.create_sequences(test_scaled)

        # Creăm DataLoaders
        self.train_loader = DataLoader(TimeSeriesDataset(X_train, y_train), batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(TimeSeriesDataset(X_val, y_val), batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(TimeSeriesDataset(X_test, y_test), batch_size=self.batch_size, shuffle=False)

    def create_sequences(self, data):
        seq = []
        labels = []
        for i in range(len(data) - self.window_size - 24):
            seq.append(data[i:i + self.window_size])
            labels.append(data[i + self.window_size, 0])

        return torch.tensor(np.array(seq), dtype=torch.float32), torch.tensor(np.array(labels), dtype=torch.float32)

    def train(self, epochs=100, patience=8, model_path = None):
        """
        Antreneaza modelul LSTM , folosind si Early Stopping + LEarning Scheduler
        """
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0

            for X_batch, y_batch in self.train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad()
                y_pred = self.model(X_batch)
                y_batch = y_batch.view(-1, 1)
                loss = self.criterion(y_pred, y_batch)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in self.val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    y_batch = y_batch.view(-1, 1)
                    y_pred = self.model(X_batch)
                    val_loss += self.criterion(y_pred, y_batch).item()

            train_losses.append(train_loss / len(self.train_loader))
            val_losses.append(val_loss / len(self.val_loader))

            for param_group in self.optimizer.param_groups:
                print(f" LR actual: {param_group['lr']:.6f}")

            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

            # EARLY STOPPING
            if val_losses[-1] < best_val_loss:
                if model_path is not None:
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                    model_save_path = model_path
                else:
                    model_save_path = "saved_lstm_model.pth"

                torch.save(self.model.state_dict(), model_save_path)
                print(f"✅ Model salvat la: {model_save_path} (epoch {epoch + 1}) cu val_loss: {val_losses[-1]:.4f}")
                best_val_loss = val_losses[-1]
                patience_counter = 0  # Resetare counter daca loss-ul scade
            else:
                patience_counter += 1  # Resetare counter daca loss-ul NU scade

            if patience_counter >= patience:
                print(f"🔴 Early stopping activat! Antrenarea se opreste la epoch {epoch + 1}.")
                break  # Iesire din train loop

            self.scheduler.step(val_losses[-1])


        # Plot Train vs Validation Loss
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.legend()
        plt.title("Train vs Validation Loss")
        plt.show()

    def predict(self):
        """
        Genereaza predictii pentru output_size=24 si le denormalizeaza.
        """
        self.model.eval()
        predictions, actuals = [], []

        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                X_batch = X_batch.to(self.device)
                y_pred = self.model(X_batch).cpu().numpy()
                y_batch_np = y_batch.cpu().numpy()

                for pred_val, actual_val in zip(y_pred.flatten(), y_batch_np.flatten()):
                    # Pregatim un array gol pentru reconversie
                    dummy_input = np.zeros((1, len(self.selected_features)))

                    # Denormalizam actualul
                    dummy_input[0, 0] = actual_val
                    denorm_actual = self.scaler.inverse_transform(dummy_input)[0, 0]

                    # Denormalizam predictia
                    dummy_input[0, 0] = pred_val
                    denorm_pred = self.scaler.inverse_transform(dummy_input)[0, 0]

                    # Fortam predictia la 0 daca actualul este 0
                    if denorm_actual == 0:
                        denorm_pred = 0

                    # Clipping ca sa nu prezicem valori negative
                    predictions.append(max(0, denorm_pred))
                    actuals.append(max(0, denorm_actual))

        # Aplatizam ca sa putem salva in csv si compara
        flat_preds = np.array(predictions).flatten()
        flat_targets = np.array(actuals).flatten()

        df_results = pd.DataFrame({
            "prediction": flat_preds,
            "actual": flat_targets
        })
        return predictions, actuals, df_results
