# KANAnalysis.py
import os
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from KAN_Model.KAN import KANTimeSeriesModel
import matplotlib.pyplot as plt


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class KANAnalyzer:
    def __init__(self, csv_path, window_size=24, batch_size=128, learning_rate=0.001, hidden_size=512):
        self.csv_path = csv_path
        self.window_size = window_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = None  # va fi initializat dupa scalare
        self.scaler = None

    def preprocess_data(self):
        # Citim si procesam fisierul CSV
        data = pd.read_csv(self.csv_path)
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data.set_index('timestamp', inplace=True)
        data['power'] = pd.to_numeric(data['power'], errors='coerce').fillna(0)

        # Adaugam feature-uri temporale
        data['day_of_week'] = data.index.dayofweek
        data['hour_of_day'] = data.index.hour
        data['hour_sin'] = np.sin(2 * np.pi * data['hour_of_day'] / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour_of_day'] / 24)
        data['day_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
        data['day_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
        data['delta_power'] = data['power'].diff().shift(1)

        # Rolling features
        data['rolling_mean_12h'] = data['power'].rolling('12h').mean().shift(1)
        data['rolling_std_12h'] = data['power'].rolling('12h').std().shift(1)
        data['rolling_max_12h'] = data['power'].rolling('12h').max().shift(1)
        data['rolling_mean_24h'] = data['power'].rolling('24h').mean().shift(1)
        data['rolling_min_12h'] = data['power'].rolling('12h').min().shift(1)
        data['rolling_median_12h'] = data['power'].rolling('12h').median().shift(1)

        # Lag features
        for lag in [1, 3, 6, 12, 24, 48]:
            data[f'lag_{lag}h'] = data['power'].shift(lag)

        # Interpolare si curatare
        data = data.interpolate(method='linear', limit_direction='both')
        data.dropna(inplace=True)

        data = data.sample(frac=1.0, random_state=42).reset_index(drop=True)

        if data['power'].std() < 0.01:
            print(f" Variabilitate extrem de mică în {self.csv_path}. Modelul va performa slab.")

        # Feature selection
        self.selected_features = ['power', 'delta_power', 'day_of_week', 'hour_of_day',
                             'lag_1h', 'lag_3h', 'lag_6h', 'lag_12h', 'lag_24h',
                             'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
                             'rolling_mean_12h', 'rolling_std_12h', 'rolling_max_12h', 'rolling_mean_24h',
                             'rolling_min_12h', 'rolling_median_12h']

        # Filtrare pe baza corelatiei cu 'power'
        corr = data[self.selected_features].corr()['power'].abs().sort_values(ascending=False)
        self.selected_features = corr.head(20).index.tolist()  # pastram cei mai relevanti

        # Split + scalare
        train_size = int(0.8 * len(data))
        val_size = int(0.1 * len(data))
        train, val, test = data[:train_size], data[train_size:train_size+val_size], data[train_size+val_size:]

        self.scaler = MinMaxScaler(feature_range=(0, 10))
        self.scaler.fit(train[self.selected_features])
        train_scaled = self.scaler.transform(train[self.selected_features])
        val_scaled = self.scaler.transform(val[self.selected_features])
        test_scaled = self.scaler.transform(test[self.selected_features])

        # Secventializare
        self.X_train, self.y_train = self.create_sequences(train_scaled)
        self.X_val, self.y_val = self.create_sequences(val_scaled)
        self.X_test, self.y_test = self.create_sequences(test_scaled)

        self.train_loader = DataLoader(TimeSeriesDataset(self.X_train, self.y_train), batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(TimeSeriesDataset(self.X_val, self.y_val), batch_size=self.batch_size)
        self.test_loader = DataLoader(TimeSeriesDataset(self.X_test, self.y_test), batch_size=self.batch_size)

        # Initializam modelul dupa ce stim input_dim
        input_dim = len(self.selected_features)
        self.model = KANTimeSeriesModel(input_dim=input_dim, hidden_dims=[256, 128, 64], output_dim=1).to(self.device)

    def create_sequences(self, data):
        seq = [data[i:i + self.window_size] for i in range(len(data) - self.window_size)]
        labels = [data[i + self.window_size, 0] for i in range(len(data) - self.window_size)]
        return torch.tensor(np.array(seq), dtype=torch.float32), torch.tensor(np.array(labels), dtype=torch.float32)

    def train(self, epochs=100, patience=8, model_path=None):
        criterion = nn.SmoothL1Loss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3,
                                                               min_lr=1e-5)

        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for X_batch, y_batch in self.train_loader:
                X_batch, y_batch = X_batch[:, -1, :].to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                y_pred = self.model(X_batch, update_grid=(epoch < 5)).squeeze()
                loss = criterion(y_pred, y_batch)
                loss += 1e-3 * self.model.regularization_loss()  # regularizare KAN
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in self.val_loader:
                    X_batch, y_batch = X_batch[:, -1, :].to(self.device), y_batch.to(self.device)
                    y_pred = self.model(X_batch).squeeze()
                    loss = criterion(y_pred, y_batch)
                    val_loss += loss.item()

            train_loss_avg = train_loss / len(self.train_loader)
            val_loss_avg = val_loss / len(self.val_loader)
            train_losses.append(train_loss_avg)
            val_losses.append(val_loss_avg)

            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss_avg:.4f}, Val Loss: {val_loss_avg:.4f}")

            scheduler.step(val_loss_avg)

            if val_loss_avg < best_val_loss:
                best_val_loss = val_loss_avg
                patience_counter = 0
                if model_path:
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                    torch.save(self.model.state_dict(), model_path)
                    print(f"✅ Model salvat la: {model_path}")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print("🔴 Early stopping activat.")
                break

        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Val Loss")
        plt.legend()
        plt.title("KAN Train vs Val Loss")
        plt.show()

    def predict(self):
        self.model.eval()
        predictions, actuals = [], []

        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                X_batch = X_batch[:, -1, :].to(self.device)
                y_pred = self.model(X_batch).squeeze().cpu().numpy()
                y_pred = np.atleast_1d(y_pred)  # 👈 Adăugat pentru a preveni eroarea la len()

                # Denormalizare
                y_pred_exp = np.zeros((len(y_pred), len(self.selected_features)))
                y_pred_exp[:, 0] = y_pred
                y_pred = self.scaler.inverse_transform(y_pred_exp)[:, 0]
                y_pred = np.clip(y_pred, 0, None)

                y_pred = np.where(y_batch == 0, 0, y_pred)

                y_batch_np = y_batch.cpu().numpy()
                y_batch_np = np.atleast_1d(y_batch_np)  # 👈 deja ai adăugat corect
                y_batch_exp = np.zeros_like(y_pred_exp)
                y_batch_exp[:, 0] = y_batch_np
                y_batch = self.scaler.inverse_transform(y_batch_exp)[:, 0]

                predictions.extend(y_pred)
                actuals.extend(y_batch)

        df_results = pd.DataFrame({
            "KAN_Prediction": predictions,
            "Actual_Value": actuals
        })

        df_results["abs_error"] = np.abs(df_results["KAN_Prediction"] - df_results["Actual_Value"])
        print("🔍 Top 5 erori mari:\n", df_results.sort_values("abs_error", ascending=False).head())

        return predictions, actuals, df_results


