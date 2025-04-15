import os
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
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
    def __init__(self, csv_path, window_size=168, batch_size=64, hidden_size=512, learning_rate=0.001):
        self.csv_path = csv_path
        self.window_size = window_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.preprocess_data()

        self.model = LSTMModel(
            input_size=len(self.selected_features),
            hidden_size=self.hidden_size,
            output_size=1
        ).to(self.device)

        self.criterion = nn.SmoothL1Loss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.9, patience=5, min_lr=0.00005)

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

    def custom_loss(self, y_pred, y_true, alpha=2.5):
        base_loss = self.criterion(y_pred, y_true)
        spike_mask = (torch.abs(y_true - y_pred) > self.threshold).float()
        spike_loss = (spike_mask * torch.abs(y_true - y_pred)).mean()
        return base_loss + alpha * spike_loss

    def preprocess_data(self):
        # Citirea datelor
        data = pd.read_csv(self.csv_path)
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data.set_index('timestamp', inplace=True)
        data['day_of_week'] = data.index.dayofweek
        data['hour_of_day'] = data.index.hour
        data['power'] = pd.to_numeric(data['power'], errors='coerce').fillna(0)  # Conversie în numeric

        data['is_weekend'] = (data.index.dayofweek >= 5).astype(int)
        data['month'] = data.index.month
        data['season'] = data['month'] % 12 // 3

        # Creare caracteristici suplimentare
        data["hour_sin"] = np.sin(2 * np.pi * data["hour_of_day"] / 24)
        data["hour_cos"] = np.cos(2 * np.pi * data["hour_of_day"] / 24)
        data["day_sin"] = np.sin(2 * np.pi * data["day_of_week"] / 7)
        data["day_cos"] = np.cos(2 * np.pi * data["day_of_week"] / 7)

        # Aplicare lag-uri
        lags = [1, 2, 3, 6, 12, 24, 48, 168]
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
        data['rolling_kurt_24h'] = data['power'].rolling(24).kurt()

        data['grad_3h'] = data['power'].rolling(3).apply(lambda x: x.iloc[-1] - x.iloc[0])
        data['grad_6h'] = data['power'].rolling(6).apply(lambda x: x.iloc[-1] - x.iloc[0])

        # Creare caracteristici temporale avansate
        data['delta_power'] = data['power'].diff().shift(1)
        data['rolling_mean_12h'] = data['power'].rolling('12h').mean().shift(1)
        data['rolling_std_12h'] = data['power'].rolling('12h').std().shift(1)
        data['rolling_max_12h'] = data['power'].rolling('12h').max().shift(1)
        data['rolling_mean_24h'] = data['power'].rolling('24h').mean().shift(1)
        data['rolling_min_12h'] = data['power'].rolling('12h').min().shift(1)
        data['rolling_median_12h'] = data['power'].rolling('12h').median().shift(1)
        data['rolling_max_24h'] = data['power'].rolling('24h').max().shift(1)
        data['rolling_min_24h'] = data['power'].rolling('24h').min().shift(1)
        data['rolling_std_24h'] = data['power'].rolling('24h').std().shift(1)
        data['rolling_sum_24h'] = data['power'].rolling(24).sum().shift(1)

        data["power_diff_24h"] = data["power"] - data["power"].shift(24)

        # Threshold auto pe baza std dev
        diff_std = data['power_diff_24h'].std()
        self.spike_event_threshold = 2 * diff_std

        print(f"Threshold auto pentru event_spike/drop: {self.spike_event_threshold}")

        data['event_spike'] = (data['power_diff_24h'] > self.spike_event_threshold).astype(int)
        data['event_drop'] = (data['power_diff_24h'] < -self.spike_event_threshold).astype(int)
        data['is_spike_context'] = data['event_spike'].rolling(3, center=True).max().fillna(0)

        data['acf_1h'] = data['power'].rolling(24).apply(
            lambda x: acf(x, nlags=1, fft=True)[1] if len(x.dropna()) == 24 else 0)
        data['pacf_1h'] = data['power'].rolling(24).apply(lambda x: pacf(x, nlags=1)[1] if len(x.dropna()) == 24 else 0)

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

        # Selectăm caracteristicile pentru scalare
        self.selected_features = ['power', 'delta_power', 'day_of_week', 'hour_of_day', 'is_weekend', 'month', 'season',
                                  'lag_1h', 'lag_2h', 'lag_3h', 'lag_6h', 'lag_12h', 'lag_24h', 'lag_48h', 'lag_168h',
                                  'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
                                  'roc_1h', 'roc_3h', 'roc_6h', 'roc_12h', 'roc_24h', 'zscore_24h', 'spike_flag',
                                  'rolling_skew_24h', 'rolling_kurt_24h', 'grad_3h', 'grad_6h',
                                  'rolling_mean_12h', 'rolling_std_12h', 'rolling_max_12h', 'rolling_mean_24h',
                                  'rolling_min_12h', 'rolling_max_24h', 'rolling_std_24h', 'power_diff_24h',
                                  'acf_1h', 'pacf_1h', 'event_spike', 'event_drop',
                                  'rolling_sum_24h',
                                  'rolling_min_24h',
                                  'rolling_max_24h',
                                  'is_spike_context'
                                  ]

        # Aplicăm scalarea DOAR pe setul de train pt. a evita data leakage
        self.scaler = MinMaxScaler(feature_range=(0, 10))
        self.scaler.fit(train_data[self.selected_features])  # Se antrenează scaler-ul doar pe train

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

        self.threshold = self.calculate_spike_threshold(train_data, method="std", k=2)
        print(f"Threshold spike auto-calculat: {self.threshold}")

    def create_sequences(self, data):
        sequences = [data[i:i + self.window_size] for i in range(len(data) - self.window_size)]
        labels = [data[i + self.window_size, 0] for i in range(len(data) - self.window_size)]
        return torch.tensor(np.array(sequences), dtype=torch.float32), torch.tensor(np.array(labels), dtype=torch.float32)

    def train(self, epochs=100, patience=10, model_path=None):
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
                loss = self.custom_loss(y_pred.squeeze(), y_batch)
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
                    val_loss += self.custom_loss(y_pred.squeeze(), y_batch).item()

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

        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.legend()
        plt.title("Train vs Validation Loss")
        plt.show()

    def predict(self):
        self.model.eval()
        predictions, actuals = [], []

        # Incarcam timestamp-urile pentru sincronizare cu predictiile
        original_data = pd.read_csv(self.csv_path)
        original_data['timestamp'] = pd.to_datetime(original_data['timestamp'])
        original_data = original_data.interpolate(method='linear', limit_direction='both')
        timestamps = original_data['timestamp'].iloc[self.window_size:]

        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                X_batch = X_batch.float()

                y_pred = self.model(X_batch).squeeze().cpu().numpy()
                y_batch = y_batch.cpu().numpy()

                # Pregatim inversarea scaler-ului pentru predictii
                zeros_pred = np.zeros((len(y_pred), len(self.selected_features)))
                zeros_pred[:, 0] = y_pred  # doar target log_power
                y_pred = self.scaler.inverse_transform(zeros_pred)[:, 0]
                y_pred = np.clip(y_pred, 0, None)

                # Pregatim inversarea scaler-ului pentru valorile reale
                zeros_actual = np.zeros((len(y_batch), len(self.selected_features)))
                zeros_actual[:, 0] = y_batch  # doar target log_power
                y_batch = self.scaler.inverse_transform(zeros_actual)[:, 0]

                predictions.extend(y_pred.tolist())
                actuals.extend(y_batch.tolist())

        df_results = pd.DataFrame({
            "timestamp": timestamps[:len(predictions)],
            "prediction": predictions,
            "actual": actuals
        })

        # Optional: smoothing pentru raportare/vizualizare
        df_results['prediction_smooth'] = df_results['prediction'].rolling(window=3, center=True).mean().fillna(
            method='bfill').fillna(method='ffill')

        return predictions, actuals, df_results

    def plot_predictions_vs_actuals(self, df_results):
        plt.figure(figsize=(20, 6))

        # Plot actual values
        plt.plot(df_results['timestamp'], df_results['actual'], label='Actual', linewidth=1.5)

        # Plot predicted values
        plt.plot(df_results['timestamp'], df_results['prediction'], label='Predicted', linewidth=1.5)


        plt.xlabel('Timp')
        plt.ylabel('Consum (Power)')
        plt.title('Predictii LSTM vs Valori Reale')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()