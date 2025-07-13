import os
import joblib
import numpy as np
import pandas as pd
import torch
from statsmodels.tsa.stattools import acf
from services.auxiliarClasses.KAN_Model.KAN import KAN

class KANForecaster:
    def __init__(self, model_path, csv_path, window_size=168, hidden_size=512, scaler_dir=None, channel_number=0):
        self.model_path = model_path
        self.csv_path = csv_path
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.scaler_dir = scaler_dir
        self.channel_number = channel_number
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.selected_features = self.get_selected_features()

        self.load_model_and_scalers()
        self.load_classifier_and_dominant()

    def get_selected_features(self):
        return ['power', 'day_of_week', 'hour_of_day', 'is_weekend', 'month', 'season',
                'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'lag_1h', 'lag_2h',
                'lag_3h', 'lag_6h', 'lag_12h', 'lag_24h', 'lag_48h', 'lag_72h', 'lag_168h', 'lag_336h', 'lag_672h',
                'roc_1h', 'roc_3h', 'roc_6h', 'roc_12h', 'roc_24h', 'zscore_24h', 'spike_flag', 'rolling_skew_24h',
                'grad_3h', 'grad_6h', 'delta_power', 'rolling_mean_12h', 'rolling_std_12h',
                'rolling_mean_24h', 'rolling_min_12h', 'rolling_median_12h', 'rolling_max_24h',
                'rolling_min_24h', 'rolling_std_24h', 'event_spike', 'event_drop', 'is_spike_context',
                'acf_1h', 'is_on', 'event_on', 'context_on_window']

    def generate_features(self, df):
        df = df.copy()
        df['day_of_week'] = df.index.dayofweek
        df['hour_of_day'] = df.index.hour
        df['power'] = pd.to_numeric(df['power'], errors='coerce').fillna(0)
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        df['month'] = df.index.month
        df['season'] = df['month'] % 12 // 3
        df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        for lag in [1, 2, 3, 6, 12, 24, 48, 72, 168, 336, 672]:
            df[f'lag_{lag}h'] = df['power'].shift(lag)

        df['roc_1h'] = df['power'].diff(1)
        df['roc_3h'] = df['power'].diff(3)
        df['roc_6h'] = df['power'].diff(6)
        df['roc_12h'] = df['power'].diff(12)
        df['roc_24h'] = df['power'].diff(24)

        df['zscore_24h'] = (df['power'] - df['power'].rolling(24).mean()) / df['power'].rolling(24).std()
        df['spike_flag'] = (df['zscore_24h'].abs() > 2).astype(int)
        df['rolling_skew_24h'] = df['power'].rolling(24).skew()
        df['grad_3h'] = df['power'].rolling(3).apply(lambda x: x.iloc[-1] - x.iloc[0])
        df['grad_6h'] = df['power'].rolling(6).apply(lambda x: x.iloc[-1] - x.iloc[0])

        df['delta_power'] = df['power'].diff().shift(1)
        df['rolling_mean_12h'] = df['power'].rolling(12).mean().shift(1)
        df['rolling_std_12h'] = df['power'].rolling(12).std().shift(1)
        df['rolling_mean_24h'] = df['power'].rolling(24).mean().shift(1)
        df['rolling_min_12h'] = df['power'].rolling(12).min().shift(1)
        df['rolling_median_12h'] = df['power'].rolling(12).median().shift(1)
        df['rolling_max_24h'] = df['power'].rolling(24).max().shift(1)
        df['rolling_min_24h'] = df['power'].rolling(24).min().shift(1)
        df['rolling_std_24h'] = df['power'].rolling(24).std().shift(1)

        df['power_diff_24h'] = df['power'] - df['power'].shift(24)
        diff_std = df['power_diff_24h'].std()
        spike_threshold = 5 * diff_std
        df['event_spike'] = (df['power_diff_24h'] > spike_threshold).astype(int)
        df['event_drop'] = (df['power_diff_24h'] < -spike_threshold).astype(int)
        df['is_spike_context'] = df['event_spike'].rolling(3, center=True).max().fillna(0)

        df['acf_1h'] = df['power'].rolling(24).apply(lambda x: acf(x, nlags=1, fft=True)[1] if len(x.dropna()) == 24 else 0)

        threshold = np.percentile(df['power'], 90)
        df['is_on'] = (df['power'] > threshold).astype(int)
        df['event_on'] = ((df['is_on'] == 1) & (df['is_on'].shift(1) == 0)).astype(int)
        df['context_on_window'] = df['is_on'].rolling(3, center=True).max().fillna(0).astype(int)

        df = df.interpolate(method='linear', limit_direction='both')
        df.fillna(0, inplace=True)
        return df

    def load_model_and_scalers(self):
        #incarcam scalere
        self.scaler_X_path = os.path.join(self.scaler_dir, f"channel_{self.channel_number}_X_scaler.pkl")
        self.scaler_y_path = os.path.join(self.scaler_dir, f"channel_{self.channel_number}_y_scaler.pkl")
        if not os.path.exists(self.scaler_X_path) or not os.path.exists(self.scaler_y_path):
            raise FileNotFoundError(f"Scalers not found for channel {self.channel_number}")

        self.scaler_X = joblib.load(self.scaler_X_path)
        self.scaler_y = joblib.load(self.scaler_y_path)

        #incarca modelul
        input_size = self.window_size * len(self.selected_features)
        self.model = KAN(layers_hidden=[input_size, self.hidden_size, 24])
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def load_classifier_and_dominant(self):
        classifier_path = os.path.join(os.path.dirname(self.scaler_dir), "classifier",
                                       f"channel_{self.channel_number}_classifier.pkl")
        dominant_path = os.path.join(os.path.dirname(self.scaler_dir), "dominant_value",
                                     f"channel_{self.channel_number}_dominant_value.pkl")

        self.classifier = joblib.load(classifier_path) if os.path.exists(classifier_path) else None
        self.dominant_value = joblib.load(dominant_path) if os.path.exists(dominant_path) else None

    def rolling_forecast_day(self, target_day):
        df = pd.read_csv(self.csv_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df = self.generate_features(df)

        forecasts = []

        for hour in range(24):
            forecast_time = pd.to_datetime(target_day) + pd.Timedelta(hours=hour)
            start = forecast_time - pd.Timedelta(hours=self.window_size)
            end = forecast_time + pd.Timedelta(hours=0)
            window_df = df.loc[start:end]

            if window_df.shape[0] != self.window_size + 1:
                continue

            input_features = window_df[self.selected_features].iloc[:self.window_size]
            actual_value = window_df['power'].iloc[-1]

            input_scaled = self.scaler_X.transform(input_features)
            input_tensor = torch.tensor(input_scaled, dtype=torch.float32).reshape(1, -1).to(self.device)

            with torch.no_grad():
                prediction = self.model(input_tensor).cpu().numpy().flatten()[0]

            prediction_real = self.scaler_y.inverse_transform([[prediction]])[0][0]
            prediction_real = max(prediction_real, 0)

            forecasts.append({
                "timestamp": forecast_time,
                "actual_power": actual_value,
                "predicted_power": prediction_real
            })

        return pd.DataFrame(forecasts)
