import torch
from imblearn.over_sampling import SMOTE
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import matplotlib.pyplot as plt
from LSTM_Model.LSTM import LSTMModel
import torch.nn as nn
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch.optim.lr_scheduler as lr_scheduler
import time
import os

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMAnalyzer:
    def __init__(self, house_dir, csv_path, window_size=35, batch_size=128, hidden_size=512, learning_rate=0.001):
        self.house_dir = house_dir
        self.csv_path = csv_path
        self.window_size = window_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Utlizare dispozivitivul : {torch.cuda.get_device_name(0)}")

        # Modelul LSTM
        self.model = LSTMModel(input_size=21, hidden_size=hidden_size, output_size=1).to(self.device)
        self.model.to(self.device)

        # Functia de cost si optimizer si scheduler-ul pentru Learning Rate
        self.criterion = nn.SmoothL1Loss() #MAE
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5, min_lr=0.0005)

    def preprocess_data(self):
        # Citirea tuturor canalelor de date
        channels = ['channel_1.dat_downsampled_1T.csv', 'channel_2.dat_downsampled_1T.csv',
                    'channel_3.dat_downsampled_1T.csv', 'channel_4.dat_downsampled_1T.csv',
                    'channel_5.dat_downsampled_1T.csv']

        data_list = []

        for channel in channels:
            file_path = os.path.join(self.house_dir, channel)
            if not os.path.exists(file_path):
                print(f"❌ Fisier lipsa: {file_path}")
                continue
            df = pd.read_csv(file_path)
            df.rename(columns={'power': channel.replace('.dat_downsampled_1T.csv', '')}, inplace=True)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            data_list.append(df)

        # Combinam toate canalele intr-un singur DataFrame
        data = data_list[0]
        for i in range(1, len(data_list)):
            data = data.merge(data_list[i], on='timestamp', how='inner')

        data['power_total'] = data['channel_1']  # Folosim direct consumul real masurat
        data['delta_power'] = data['power_total'].diff().shift(1)

        # Extragere caracteristici avansate
        data['day_of_week'] = data['timestamp'].dt.dayofweek
        data['hour_of_day'] = data['timestamp'].dt.hour
        data['minute_of_hour'] = data['timestamp'].dt.minute

        # Feature-uri ciclice pentru a modela timpul
        data["hour_sin"] = np.sin(2 * np.pi * data["hour_of_day"] / 24)
        data["hour_cos"] = np.cos(2 * np.pi * data["hour_of_day"] / 24)
        data["day_sin"] = np.sin(2 * np.pi * data["day_of_week"] / 7)
        data["day_cos"] = np.cos(2 * np.pi * data["day_of_week"] / 7)
        data["minute_sin"] = np.sin(2 * np.pi * data["minute_of_hour"] / 60)
        data["minute_cos"] = np.cos(2 * np.pi * data["minute_of_hour"] / 60)

        #  Creare lag-uri pentru a captura dependente temporale
        lags = [1, 3, 6, 12, 24, 48]
        for lag in lags:
            data[f'lag_{lag}h'] = data['power_total'].shift(lag)

        data.set_index('timestamp', inplace=True)  # Setăm indexul la timestamp

        # Medii mobile pentru tendinte pe termen lung
        data['rolling_mean_12h'] = data['power_total'].rolling('12h').mean().shift(1)
        data['rolling_std_12h'] = data['power_total'].rolling('12h').std().shift(1)
        data['rolling_max_12h'] = data['power_total'].rolling('12h').max().shift(1)
        data['rolling_mean_24h'] = data['power_total'].rolling('24h').mean().shift(1)
        data['rolling_min_12h'] = data['power_total'].rolling('12h').min().shift(1)

        #Eliminare valori lipsa
        data = data.interpolate(method='linear', limit_direction='both')

        # Impartirea in train/val/test
        train_size = int(0.8 * len(data))
        val_size = int(0.1 * len(data))

        train_data = data.iloc[:train_size]
        val_data = data.iloc[train_size:train_size + val_size]
        test_data = data.iloc[train_size + val_size:]

        #  Aplicam scalarea
        selected_features = [
            'power_total', 'delta_power', 'day_of_week', 'hour_of_day',
            'lag_1h', 'lag_3h', 'lag_6h', 'lag_12h', 'lag_24h', 'lag_48h',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'minute_sin', 'minute_cos',
            'rolling_mean_12h', 'rolling_std_12h', 'rolling_max_12h', 'rolling_mean_24h', 'rolling_min_12h']

        self.scaler = MinMaxScaler(feature_range=(0, 10))
        self.scaler.fit(train_data[selected_features])

        train_scaled = self.scaler.transform(train_data[selected_features])
        val_scaled = self.scaler.transform(val_data[selected_features])
        test_scaled = self.scaler.transform(test_data[selected_features])

        #  Generare secvente pentru LSTM
        X_train, y_train = self.create_sequences(train_scaled)
        X_val, y_val = self.create_sequences(val_scaled)
        X_test, y_test = self.create_sequences(test_scaled)

        # Creare DataLoaders
        self.train_loader = DataLoader(TimeSeriesDataset(X_train, y_train), batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(TimeSeriesDataset(X_val, y_val), batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(TimeSeriesDataset(X_test, y_test), batch_size=self.batch_size, shuffle=False)

    def create_sequences(self, data):

        sequences = [data[i:i + self.window_size] for i in range(len(data) - self.window_size)]
        labels = [data[i + self.window_size, 0] for i in range(len(data) - self.window_size)]
        return torch.tensor(np.array(sequences), dtype=torch.float32), torch.tensor(np.array(labels), dtype=torch.float32)

    def train(self, epochs=100, patience=5, model_path = None):
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
                loss = self.criterion(y_pred.squeeze(), y_batch)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in self.val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    y_pred = self.model(X_batch)
                    val_loss += self.criterion(y_pred.squeeze(), y_batch).item()

            train_losses.append(train_loss / len(self.train_loader))
            val_losses.append(val_loss / len(self.val_loader))

            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

            if epoch > 8:
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] *= 0.9  # 🔹 Reducem LR cu 50%
                    print(f"🔽 Learning Rate redus la {param_group['lr']:.6f}")

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
        Generează predicții și le denormalizează.
        """
        self.model.eval()
        predictions, actuals = [], []

        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                X_batch = X_batch.to(self.device)
                y_pred = self.model(X_batch).squeeze().cpu().numpy()

                # Evitam erori legate de valori zero in y_batch
                y_pred = np.where(y_batch.cpu().numpy() == 0, 0, y_pred)

                # Aplicam denormalizarea
                y_pred_expanded = np.zeros((len(y_pred), len(self.scaler.feature_names_in_)))  # Pregatim structura
                y_pred_expanded[:, 0] = y_pred  # Setam doar prima coloana (target)
                y_pred = self.scaler.inverse_transform(y_pred_expanded)[:, 0]  # Inversam doar prima coloana

                # Eliminăm valori negative într-un mod mai elegant
                y_pred = np.clip(y_pred, 0, None)

                # Aplicam aceeaSi denormalizare si pe y_batch
                y_batch_expanded = np.zeros((len(y_batch.cpu().numpy()), len(self.scaler.feature_names_in_)))
                y_batch_expanded[:, 0] = y_batch.cpu().numpy()
                y_batch = self.scaler.inverse_transform(y_batch_expanded)[:, 0]

                predictions.extend(y_pred)
                actuals.extend(y_batch)

        return predictions, actuals

    def load_model(self, model_path="saved_lstm_model.pth"):
        """
        Incarca un model antrenat anterior daca exista.
        """
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
            self.model.to(self.device)
            self.model.eval()
            print(f"✅ Model incarcat din {model_path}")
        else:
            print("❌ Modelul nu exista! Trebuie antrenat inainte de a putea fi folosit.")

    def predict_future(self, future_steps):
        """
        Genereaza predicții pentru un numar specific de pasi in viitor.
        """
        self.model.eval()
        predictions = []

        # Folosim ultimele `window_size` valori ca punct de start
        last_sequence = self.test_loader.dataset.X[-1].unsqueeze(0).to(self.device)

        with torch.no_grad():
            for _ in range(future_steps):
                y_pred = self.model(last_sequence).squeeze().cpu().numpy()

                # Obtinem numele caracteristicilor originale folosite de scaler
                feature_names = self.scaler.feature_names_in_ if hasattr(self.scaler, "feature_names_in_") else [
                    f"feature_{i}" for i in range(18)]

                # Construim un DataFrame cu numele corecte pentru MinMaxScaler
                y_pred_expanded = pd.DataFrame(np.zeros((1, 18)), columns=feature_names)
                y_pred_expanded.iloc[:, 0] = y_pred  # Setam doar prima coloana cu predictia

                # Aplicam denormalizarea
                y_pred = self.scaler.inverse_transform(y_pred_expanded)[:, 0][0]

                # Salvare predictie
                predictions.append(y_pred)

                # Construim urmatoarea secventa, eliminand prima valoare si adăugand noua predictie
                new_input = last_sequence.squeeze(0).cpu().numpy()

                # Construim un DataFrame pentru MinMaxScaler
                new_input_df = pd.DataFrame([[y_pred] + [0] * 17], columns=feature_names)

                # Aplicam scalarea corecta
                new_input[-1, 0] = self.scaler.transform(new_input_df)[0][0]  # Adaugam predictia scalata

                last_sequence = torch.tensor(new_input, dtype=torch.float32).unsqueeze(0).to(self.device)

        return predictions
