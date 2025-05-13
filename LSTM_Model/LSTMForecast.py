import joblib
import torch
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta

from statsmodels.tsa.stattools import acf, pacf

from LSTM_Model.LSTM import LSTMModel  # Asigură-te că path-ul e corect

class LSTMForecaster:
    def __init__(self, model_path, csv_path, window_size=168, hidden_size=512, scaler_dir = None, channel_number = 0):
        self.model_path = model_path
        self.csv_path = csv_path
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.scaler_dir = scaler_dir
        self.channel_number = channel_number
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.scaler = None
        self.selected_features = None
        self.context_df = None  # ultimele N ore (fereastra initiala)

    def load_model_and_scalers(self):
        """
        Incarca modelul salvat si scalerele asociate pentru forecast.
        """
        # Incarcam modelul
        self.model = LSTMModel(
            input_size=len(self.selected_features),
            hidden_size=self.hidden_size,
            output_size=24
        ).to(self.device)

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Modelul nu a fost gasit: {self.model_path}")

        state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # Incarcam scalerele
        scaler_x_path = os.path.join(self.scaler_dir, f"channel_{self.channel_number}_X_scaler.pkl")
        scaler_y_path = os.path.join(self.scaler_dir, f"channel_{self.channel_number}_y_scaler.pkl")

        if not os.path.exists(scaler_x_path) or not os.path.exists(scaler_y_path):
            raise FileNotFoundError(
                f"Scalerele pentru channel_{self.channel_number} nu au fost gasite in {self.scaler_dir}")

        self.scaler_X = joblib.load(scaler_x_path)
        self.scaler_y = joblib.load(scaler_y_path)
        print(f"Model incarcat cu {len(self.selected_features)} caracteristici pentru channel_{self.channel_number}.")

    def generate_features(self, data):
        """
        Genereaza toate coloanele necesare pentru forecast, la fel ca in timpul antrenarii.
        """
        data = data.copy()
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
        lags = [1, 2, 3, 6, 12, 24, 48, 168]
        for lag in lags:
            data[f'lag_{lag}h'] = data['power'].shift(lag)

        data['roc_1h'] = data['power'].diff(1)
        data['roc_3h'] = data['power'].diff(3)
        data['roc_6h'] = data['power'].diff(6)
        data['roc_12h'] = data['power'].diff(12)
        data['roc_24h'] = data['power'].diff(24)

        window = 24
        data['zscore_24h'] = (data['power'] - data['power'].rolling(window).mean()) / data['power'].rolling(
            window).std()
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

        data['event_spike'] = (data['power_diff_24h'] > self.spike_event_threshold).astype(int)
        data['event_drop'] = (data['power_diff_24h'] < -self.spike_event_threshold).astype(int)
        data['is_spike_context'] = data['event_spike'].rolling(3, center=True).max().fillna(0)

        data['acf_1h'] = data['power'].rolling(24).apply(
            lambda x: acf(x, nlags=1, fft=True)[1] if len(x.dropna()) == 24 else 0)
        data['pacf_1h'] = data['power'].rolling(24).apply(lambda x: pacf(x, nlags=1)[1] if len(x.dropna()) == 24 else 0)

        data = data.interpolate(method='linear', limit_direction='both')
        data.fillna(0, inplace=True)

        return data

    def load_recent_data(self):
        df = pd.read_csv(self.csv_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df = self.generate_features(df)
        self.context_df = df.tail(self.window_size).copy()
        self.selected_features = list(self.context_df.columns)
        print(f"Context de forecast pregatit din ultimele {self.window_size} valori cu {len(self.selected_features)} caracteristici.")

    def predict_day(self, target_day):
        self.load_model_and_scalers()

        # Incarcam datele pentru ziua tinta
        df = pd.read_csv(self.csv_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df = self.generate_features(df)
        day_df = df.loc[target_day]

        if day_df.shape[0] != 24:
            raise ValueError(f"Ziua {target_day} nu are 24 de puncte de date.")

        # Scalez input-ul
        input_features = day_df[self.selected_features].fillna(0)
        input_scaled = self.scaler_X.transform(input_features)
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Predictie
        with torch.no_grad():
            predictions = self.model(input_tensor).cpu().numpy().flatten()

        # Denormalizare
        predictions_real = self.scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()

        # Organizare rezultate
        df_results = pd.DataFrame({
            "timestamp": day_df.index,
            "actual_power": day_df['power'].values,
            "predicted_power": predictions_real
        })

        return df_results
