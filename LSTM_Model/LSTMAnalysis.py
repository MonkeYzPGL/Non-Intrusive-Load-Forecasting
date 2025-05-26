import os
import time

import joblib
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from matplotlib import pyplot as plt
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tsa.stattools import acf, pacf
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from LSTM_Model.LSTM import LSTMModel

import seaborn as sns

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMAnalyzer:
    timing_csv = "training_timing_lstm.csv"
    def __init__(self, csv_path, window_size=168, batch_size=64, hidden_size=512, learning_rate=0.001, scaler_dir = None, channel_number = 0):
        torch.manual_seed(42)
        np.random.seed(42)
        self.csv_path = csv_path
        self.window_size = window_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.scaler_dir = scaler_dir
        self.channel_number = channel_number
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Cream path-urile pentru scalere
        if self.scaler_dir is not None:
            os.makedirs(self.scaler_dir, exist_ok=True)
            self.scaler_path_X = os.path.join(self.scaler_dir, f"channel_{self.channel_number}_X_scaler.pkl")
            self.scaler_path_y = os.path.join(self.scaler_dir, f"channel_{self.channel_number}_y_scaler.pkl")
        else:
            self.scaler_path_X = None
            self.scaler_path_y = None

        self.preprocess_data()

        self.model = LSTMModel(
            input_size=len(self.selected_features),
            hidden_size=self.hidden_size,
            output_size=24
        ).to(self.device)

        self.criterion = nn.SmoothL1Loss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.9, patience=3, min_lr=0.00005)


    def calculate_spike_threshold(self,df, method="std", k=3, percentile=95):
        if "power" not in df.columns:
            raise ValueError("DataFrame-ul trebuie sa aiba o coloana 'power'.")

        if method == "std":
            mean_power = df['power'].mean()
            std_power = df['power'].std()
            threshold = mean_power + k * std_power
        elif method == "percentile":
            threshold = np.percentile(df['power'], percentile)
        else:
            raise ValueError("Metoda trebuie sa fie 'std' sau 'percentile'.")

        return threshold

    def custom_loss(self, y_pred, y_true, alpha=3):
        base_loss = self.criterion(y_pred, y_true)
        spike_mask = (torch.abs(y_true - y_pred) > self.threshold).float()
        spike_loss = (spike_mask * torch.abs(y_true - y_pred)).mean()
        return base_loss + alpha * spike_loss

    def remove_highly_correlated_features(self, data, features, threshold=0.95):
        corr_matrix = data[features].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        print(f"[INFO] Caracteristici eliminate din cauza corelatiei > {threshold}: {to_drop}")
        data = data.drop(columns=to_drop)
        features = [f for f in features if f not in to_drop]
        return data, features

    def preprocess_data(self):
        # Citirea datelor
        data = pd.read_csv(self.csv_path)
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data.set_index('timestamp', inplace=True)
        data['day_of_week'] = data.index.dayofweek
        data['hour_of_day'] = data.index.hour
        data['power'] = pd.to_numeric(data['power'], errors='coerce').fillna(0)

        data['is_weekend'] = (data.index.dayofweek >= 5).astype(int)
        data['month'] = data.index.month
        data['season'] = data['month'] % 12 // 3

        # Creare caracteristici suplimentare
        data["hour_sin"] = np.sin(2 * np.pi * data["hour_of_day"] / 24)
        data["hour_cos"] = np.cos(2 * np.pi * data["hour_of_day"] / 24)
        data["day_sin"] = np.sin(2 * np.pi * data["day_of_week"] / 7)
        data["day_cos"] = np.cos(2 * np.pi * data["day_of_week"] / 7)

        # Aplicare lag-uri
        lags = [1, 2, 3, 6, 12, 24, 48,72, 168, 336, 672]
        for lag in lags:
            data[f'lag_{lag}h'] = data['power'].shift(lag)

        data['roc_1h'] = data['power'].diff(1)
        data['roc_3h'] = data['power'].diff(3)
        data['roc_6h'] = data['power'].diff(6)
        data['roc_12h'] = data['power'].diff(12)
        data['roc_24h'] = data['power'].diff(24)

        window = 24
        data['zscore_24h'] = (data['power'] - data['power'].rolling(window).mean()) / data['power'].rolling(window).std()
        data['spike_flag'] = (data['zscore_24h'].abs() > 2).astype(int)

        data['rolling_skew_24h'] = data['power'].rolling(24).skew()

        data['grad_3h'] = data['power'].rolling(3).apply(lambda x: x.iloc[-1] - x.iloc[0])
        data['grad_6h'] = data['power'].rolling(6).apply(lambda x: x.iloc[-1] - x.iloc[0])

        # Creare caracteristici temporale avansate
        data['delta_power'] = data['power'].diff().shift(1)
        data['rolling_mean_12h'] = data['power'].rolling('12h').mean().shift(1)
        data['rolling_std_12h'] = data['power'].rolling('12h').std().shift(1)
        data['rolling_mean_24h'] = data['power'].rolling('24h').mean().shift(1)
        data['rolling_min_12h'] = data['power'].rolling('12h').min().shift(1)
        data['rolling_median_12h'] = data['power'].rolling('12h').median().shift(1)
        data['rolling_max_24h'] = data['power'].rolling('24h').max().shift(1)
        data['rolling_min_24h'] = data['power'].rolling('24h').min().shift(1)
        data['rolling_std_24h'] = data['power'].rolling('24h').std().shift(1)

        data["power_diff_24h"] = data["power"] - data["power"].shift(24)

        # Threshold auto pe baza std dev
        diff_std = data['power_diff_24h'].std()
        self.spike_event_threshold = 2 * diff_std

        print(f"Threshold auto pentru event_spike/drop: {self.spike_event_threshold}")

        data['event_spike'] = (data['power_diff_24h'] > self.spike_event_threshold).astype(int)
        data['event_drop'] = (data['power_diff_24h'] < -self.spike_event_threshold).astype(int)
        data['is_spike_context'] = data['event_spike'].rolling(3, center=True).max().fillna(0)

        data['acf_1h'] = data['power'].rolling(24).apply(lambda x: acf(x, nlags=1, fft=True)[1] if len(x.dropna()) == 24 else 0)

        # Umplem valorile lipsa
        data = data.interpolate(method='linear', limit_direction='both')

        spike_data = data[data['event_spike'] == 1]
        drop_data = data[data['event_drop'] == 1]

        # Adauga-le de 2x (experimenteaza cu 2 sau 3)
        data = pd.concat([data, spike_data, drop_data])
        data = data.sort_index()

        # Împărțirea datelor înainte de scalare
        train_size = int(0.8 * len(data))
        val_size = int(0.1 * len(data))

        train_data = data.iloc[:train_size]
        val_data = data.iloc[train_size:train_size + val_size]
        test_data = data.iloc[train_size + val_size:]

        # Selectam caracteristicile pentru scalare
        self.selected_features = ['power', 'day_of_week', 'hour_of_day', 'is_weekend', 'month', 'season',
                                   'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'lag_1h', 'lag_2h',
                                   'lag_3h', 'lag_6h', 'lag_12h', 'lag_24h', 'lag_48h','lag_72h', 'lag_168h', 'lag_336h', 'lag_672h',
                                   'roc_1h', 'roc_3h', 'roc_6h', 'roc_12h', 'roc_24h', 'zscore_24h',
                                   'spike_flag', 'rolling_skew_24h', 'grad_3h',
                                   'grad_6h', 'delta_power', 'rolling_mean_12h', 'rolling_std_12h',
                                   'rolling_mean_24h', 'rolling_min_12h',
                                   'rolling_median_12h', 'rolling_max_24h', 'rolling_min_24h',
                                   'rolling_std_24h', 'event_spike',
                                   'event_drop', 'is_spike_context', 'acf_1h'
                                  ]

        #data, self.selected_features = self.remove_highly_correlated_features(data, self.selected_features)

        # Aplicăm scalarea DOAR pe setul de train pt. a evita data leakage
        self.scaler = MinMaxScaler(feature_range=(0, 10))
        self.scaler.fit(train_data[self.selected_features])

        self.scaler_y = MinMaxScaler(feature_range=(0, 10))
        self.scaler_y.fit(train_data[['power']])

        # Transformăm fiecare subset folosind scaler-ul antrenat pe train
        train_scaled = self.scaler.transform(train_data[self.selected_features])
        val_scaled = self.scaler.transform(val_data[self.selected_features])
        test_scaled = self.scaler.transform(test_data[self.selected_features])

        X_train, y_train = self.create_sequences(train_scaled)
        X_val, y_val = self.create_sequences(val_scaled)
        X_test, y_test = self.create_sequences(test_scaled)

        # Creăm DataLoaders
        self.train_loader = DataLoader(TimeSeriesDataset(X_train, y_train), batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(TimeSeriesDataset(X_val, y_val), batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(TimeSeriesDataset(X_test, y_test), batch_size=self.batch_size, shuffle=False)

        self.threshold = self.calculate_spike_threshold(train_data, method="std", k=1)
        print(f"Threshold spike auto-calculat: {self.threshold}")

        self.test_data = test_data.copy()

        if self.scaler_path_X is not None and self.scaler_path_y is not None:
            joblib.dump(self.scaler, self.scaler_path_X)
            joblib.dump(self.scaler_y, self.scaler_path_y)
            print(f" Scalere salvate pentru channel_{self.channel_number}: {self.scaler_path_X}, {self.scaler_path_y}")


        # # Compute the correlation matrix
        # corr_matrix = data[self.selected_features].corr()
        #
        # # Plot the correlation matrix as a heatmap
        # plt.figure(figsize=(12, 10))
        # sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', fmt=".2f")
        # plt.title("Correlation Matrix of Engineered Features")
        # plt.show()

    def create_sequences(self, data, horizon=24):
        X, y = [], []
        for i in range(len(data) - self.window_size - horizon):
            X.append(data[i:i + self.window_size])
            y.append(data[i + self.window_size: i + self.window_size + horizon, 0])
        return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.float32)

    def train(self, epochs=100, patience=15, model_path=None):
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0

        start_time = time.time()

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0

            for X_batch, y_batch in self.train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad()
                y_pred = self.model(X_batch)
                loss = self.custom_loss(y_pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5)
                self.optimizer.step()
                train_loss += loss.item()

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in self.val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    y_pred = self.model(X_batch)
                    val_loss += self.custom_loss(y_pred, y_batch).item()

            train_losses.append(train_loss / len(self.train_loader))
            val_losses.append(val_loss / len(self.val_loader))

            current_lr = self.optimizer.param_groups[0]['lr']
            print(
                f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, LR: {current_lr:.6f}")

            if val_losses[-1] < best_val_loss:
                if model_path is not None:
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                    model_save_path = model_path
                else:
                    model_save_path = "saved_lstm_model.pth"

                torch.save(self.model.state_dict(), model_save_path)
                print(f" Model salvat la: {model_save_path} (epoch {epoch + 1}) cu val_loss: {val_losses[-1]:.4f}")
                best_val_loss = val_losses[-1]
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f" Early stopping activat! Antrenarea se opreste la epoch {epoch + 1}.")
                break

            self.scheduler.step(val_losses[-1])

        # End timer
        end_time = time.time()
        training_duration = end_time - start_time
        print(f"\n Timp total pentru antrenare canal {self.channel_number}: {training_duration:.2f} secunde")

        # Salvare timp intr-un CSV global
        timing_data = pd.DataFrame([{
                "channel_number": self.channel_number,
                "training_time_seconds": training_duration
        }])

        # Verifica daca fisierul exista, daca nu, adauga header
        if not os.path.exists(self.timing_csv):
            timing_data.to_csv(self.timing_csv, index=False, mode='w')
        else:
            timing_data.to_csv(self.timing_csv, index=False, mode='a', header=False)

        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.legend()
        plt.title("Train vs Validation Loss")
        plt.show()

    def predict(self):
        self.model.eval()
        rows = []

        with torch.no_grad():
            for batch_idx, (X_batch, y_batch) in enumerate(self.test_loader):
                X_batch = X_batch.to(self.device).float()
                y_batch = y_batch.to(self.device)

                # Predictii
                y_pred = self.model(X_batch).cpu().numpy()  # (batch, 24)
                y_batch = y_batch.cpu().numpy()  # (batch, 24)

                for i in range(len(y_pred)):
                    pred_fill = np.zeros((24, len(self.selected_features)))
                    actual_fill = np.zeros((24, len(self.selected_features)))

                    pred_fill[:, 0] = y_pred[i]
                    actual_fill[:, 0] = y_batch[i]

                    y_pred_inv = self.scaler.inverse_transform(pred_fill)[:, 0]
                    y_pred_inv = np.clip(y_pred_inv, 0, None)
                    y_actual_inv = self.scaler.inverse_transform(actual_fill)[:, 0]

                    timestamp = self.test_data.index[self.window_size + batch_idx * self.batch_size + i]

                    row = {
                        "timestamp": timestamp,
                        "prediction": y_pred_inv[0],  # prima ora
                        "actual": y_actual_inv[0]  # prima ora
                    }

                    rows.append(row)

        df_results = pd.DataFrame(rows)

        return df_results