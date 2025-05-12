import torch
from KAN_Model.KAN import KAN
import pandas as pd
import numpy as np

def load_kan_model(model_path, input_size, hidden_size):
    model = KAN(layers_hidden=[input_size, hidden_size, 1])  # Output size = 1
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def preprocess_for_forecast(csv_path, selected_features, scaler):
    # Citirea datelor
    data = pd.read_csv(csv_path)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data.set_index('timestamp', inplace=True)
    data['power'] = pd.to_numeric(data['power'], errors='coerce').fillna(0)

    # Caracteristici temporale
    data['day_of_week'] = data.index.dayofweek
    data['hour_of_day'] = data.index.hour
    data['is_weekend'] = (data.index.dayofweek >= 5).astype(int)
    data['month'] = data.index.month
    data['season'] = data['month'] % 12 // 3

    # Transformari periodice (sinusoide)
    data["hour_sin"] = np.sin(2 * np.pi * data["hour_of_day"] / 24)
    data["hour_cos"] = np.cos(2 * np.pi * data["hour_of_day"] / 24)
    data["day_sin"] = np.sin(2 * np.pi * data["day_of_week"] / 7)
    data["day_cos"] = np.cos(2 * np.pi * data["day_of_week"] / 7)

    # Aplicare lag-uri
    lags = [1, 2, 3, 6, 12, 24, 48, 168]
    for lag in lags:
        data[f'lag_{lag}h'] = data['power'].shift(lag)

    # Rate of Change (ROC)
    data['roc_1h'] = data['power'].diff(1)
    data['roc_3h'] = data['power'].diff(3)
    data['roc_6h'] = data['power'].diff(6)
    data['roc_12h'] = data['power'].diff(12)
    data['roc_24h'] = data['power'].diff(24)

    # Rolling statistics
    data['zscore_24h'] = (data['power'] - data['power'].rolling(24).mean()) / data['power'].rolling(24).std()
    data['rolling_skew_24h'] = data['power'].rolling(24).skew()
    data['rolling_kurt_24h'] = data['power'].rolling(24).kurt()

    # Interpolare date lipsa
    data = data.interpolate(method='linear', limit_direction='both')

    # Eliminam randurile cu valori lipsa dupa lag-uri
    data = data.dropna()

    # Selectam doar features folosite la antrenament
    X = scaler.transform(data[selected_features].values)

    # Conversie la tensor
    X = torch.tensor(X, dtype=torch.float32)

    return X, data.index

def load_scalers(channel_name):
    scaler_X_path = os.path.join(scalers_dir, f"{channel_name}_scaler_X.pkl")
    scaler_y_path = os.path.join(scalers_dir, f"{channel_name}_scaler_y.pkl")

    if not os.path.isfile(scaler_X_path) or not os.path.isfile(scaler_y_path):
        print(f"⚠️ Scalerii lipsesc pentru {channel_name}")
        return None, None

    scaler_X = joblib.load(scaler_X_path)
    scaler_y = joblib.load(scaler_y_path)

    print(f"✅ Scalerii incarcati pentru {channel_name}")
    return scaler_X, scaler_y