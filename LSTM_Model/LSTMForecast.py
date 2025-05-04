import torch
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta
from LSTM_Model.LSTM import LSTMModel  # Asigură-te că path-ul e corect

class LSTMForecaster:
    def __init__(self, model_path, csv_path, window_size=168, hidden_size=512):
        self.model_path = model_path
        self.csv_path = csv_path
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.scaler = None
        self.selected_features = None
        self.context_df = None  # ultimele N ore (fereastra initiala)

    def load_model(self):
        """
        Incarca modelul salvat si il pregateste pentru forecast.
        """
        if not self.selected_features:
            raise ValueError("selected_features trebuie setat inainte de a incarca modelul.")

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

        print(f" Model incarcat din: {self.model_path}")


    def generate_features(self, df):
        """
        Genereaza toate coloanele necesare pentru forecast, la fel ca in timpul antrenarii.
        """
        df = df.copy()
        df['day_of_week'] = df.index.dayofweek
        df['hour_of_day'] = df.index.hour
        df['power'] = pd.to_numeric(df['power'], errors='coerce').fillna(0)

        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        df['month'] = df.index.month
        df['season'] = df['month'] % 12 // 3

        # sin/cos pentru ora si zi
        df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
        df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

        # laguri
        for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
            df[f'lag_{lag}h'] = df['power'].shift(lag)

        # rate of change
        df['roc_1h'] = df['power'].diff(1)
        df['roc_3h'] = df['power'].diff(3)
        df['roc_6h'] = df['power'].diff(6)
        df['roc_12h'] = df['power'].diff(12)
        df['roc_24h'] = df['power'].diff(24)

        # zscore + spike flag
        window = 24
        df['zscore_24h'] = (df['power'] - df['power'].rolling(window).mean()) / df['power'].rolling(window).std()
        df['spike_flag'] = (df['zscore_24h'].abs() > 2).astype(int)

        df['rolling_skew_24h'] = df['power'].rolling(24).skew()
        df['rolling_kurt_24h'] = df['power'].rolling(24).kurt()

        df['grad_3h'] = df['power'].rolling(3).apply(lambda x: x.iloc[-1] - x.iloc[0])
        df['grad_6h'] = df['power'].rolling(6).apply(lambda x: x.iloc[-1] - x.iloc[0])

        df['delta_power'] = df['power'].diff().shift(1)
        df['rolling_mean_12h'] = df['power'].rolling('12h').mean().shift(1)
        df['rolling_std_12h'] = df['power'].rolling('12h').std().shift(1)
        df['rolling_max_12h'] = df['power'].rolling('12h').max().shift(1)
        df['rolling_mean_24h'] = df['power'].rolling('24h').mean().shift(1)
        df['rolling_min_12h'] = df['power'].rolling('12h').min().shift(1)
        df['rolling_median_12h'] = df['power'].rolling('12h').median().shift(1)
        df['rolling_max_24h'] = df['power'].rolling('24h').max().shift(1)
        df['rolling_min_24h'] = df['power'].rolling('24h').min().shift(1)
        df['rolling_std_24h'] = df['power'].rolling('24h').std().shift(1)
        df['rolling_sum_24h'] = df['power'].rolling(24).sum().shift(1)
        df['rolling_mean_7d'] = df['power'].rolling(168).mean().shift(1)

        df["power_diff_24h"] = df["power"] - df["power"].shift(24)

        # spike/drop events
        diff_std = df['power_diff_24h'].std()
        spike_event_threshold = 2 * diff_std

        df['event_spike'] = (df['power_diff_24h'] > spike_event_threshold).astype(int)
        df['event_drop'] = (df['power_diff_24h'] < -spike_event_threshold).astype(int)
        df['is_spike_context'] = df['event_spike'].rolling(3, center=True).max().fillna(0)

        # ACF si PACF
        from statsmodels.tsa.stattools import acf, pacf
        df['acf_1h'] = df['power'].rolling(24).apply(
            lambda x: acf(x, nlags=1, fft=True)[1] if x.nunique() > 1 else 0)

        df['pacf_1h'] = df['power'].rolling(24).apply(
            lambda x: pacf(x, nlags=1)[1] if x.nunique() > 1 else 0)

        df = df.interpolate(method='linear', limit_direction='both')
        df.dropna(inplace=True)

        return df

    def load_recent_data(self):
        """
        Incarca ultimele date reale si le preproceseaza pentru a construi contextul initial de forecast.
        """
        df = pd.read_csv(self.csv_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df['power'] = pd.to_numeric(df['power'], errors='coerce').fillna(0)

        # Genereaza toate feature-urile
        df = self.generate_features(df)

        # Pastreaza ultimele window_size randuri ca seed
        self.context_df = df.tail(self.window_size).copy()

        # Lista de feature-uri folosite (la fel ca in training)
        self.selected_features = [
            'power', 'delta_power', 'day_of_week', 'hour_of_day', 'is_weekend', 'month', 'season',
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

        # Init scaler si scalare context
        self.scaler = MinMaxScaler(feature_range=(0, 10))
        self.scaler.fit(self.context_df[self.selected_features])

        print(f" Context de forecast pregatit din ultimele {self.window_size} valori.")

    def predict_day(self, target_day):
        """
        Testeaza modelul pe o zi existenta din dataset.
        target_day: string format 'YYYY-MM-DD'
        """
        df = pd.read_csv(self.csv_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df['power'] = pd.to_numeric(df['power'], errors='coerce').fillna(0)

        df = self.generate_features(df)

        min_date = df.index.min().date()
        max_date = df.index.max().date()
        target_date = pd.to_datetime(target_day).date()

        if target_date < min_date or target_date > max_date:
            raise ValueError(
                f"Data selectata ({target_day}) este in afara intervalului datasetului ({min_date} - {max_date}).")

        # Selectam ziua dorita
        day_df = df.loc[target_day]
        if len(day_df) != 24:
            raise ValueError(f"Ziua {target_day} nu are 24 de puncte de date (posibil lipsuri in date).")

        input_features = day_df[self.selected_features].copy()
        input_scaled = self.scaler.transform(input_features)

        input_tensor = torch.tensor(input_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Predictie
        with torch.no_grad():
            predictions = self.model(input_tensor).cpu().numpy().flatten()

        # Denormalizare
        predictions_real = []
        for pred in predictions:
            pred_array = np.zeros((1, len(self.selected_features)))
            pred_array[0][self.selected_features.index('power')] = pred
            prediction_real = self.scaler.inverse_transform(pred_array)[0][self.selected_features.index('power')]
            prediction_real = max(prediction_real, 0)
            predictions_real.append(prediction_real)

        # Organizare rezultate
        df_results = pd.DataFrame({
            "timestamp": day_df.index,
            "actual_power": day_df['power'].values,
            "predicted_power": predictions_real
        })

        return df_results
