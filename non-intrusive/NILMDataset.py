import os
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
import numpy as np
from LSTMNilm import LSTMDecomposer
from tqdm import tqdm
from services.ErrorMetricsService import ErrorMetricsAnalyzer

class NILMDataset(Dataset):
    def __init__(self, x_path, y_path):
        self.X = np.load(x_path)
        self.Y = np.load(y_path)

        assert self.X.shape[0] == self.Y.shape[0], "X si Y trebuie sa aiba acelasi nr de exemple"

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)  # [168, 1]
        y = torch.tensor(self.Y[idx], dtype=torch.float32)  # [168, n_aparate]
        return x, y

class NILMTrainer:
    def __init__(self, x_path, y_path, model, batch_size=64, lr=1e-3, num_epochs=30,
                     save_path="model_nilm_lstm.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.batch_size = batch_size
        self.learning_rate = lr
        self.num_epochs = num_epochs
        self.save_path = save_path

        dataset = NILMDataset(x_path, y_path)
        total_len = len(dataset)

        train_len = int(0.8 * total_len)
        val_len = int(0.1 * total_len)
        test_len = total_len - train_len - val_len

        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(42)
        )

        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.9,
                                                                        patience=3, min_lr=1e-5)

    def train(self):
        best_loss = float("inf")
        patience = 10
        patience_counter = 0

        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0

            for x_batch, y_batch in tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.num_epochs}"):
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(x_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5)
                self.optimizer.step()

                running_loss += loss.item()

            epoch_loss = running_loss / len(self.train_loader)
            self.scheduler.step(epoch_loss)
            print(f"epoch {epoch + 1} Loss: {epoch_loss:.4f}")

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(self.model.state_dict(), self.save_path)
                print(f"model salvat la: {self.save_path}")
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"early stopping activat la epoca {epoch + 1}.")
                break

    def predict(self, appliance_cols=None, save_dir=None):
        self.model.eval()
        predictions, targets = [], []

        with torch.no_grad():
            for x_batch, y_batch in self.test_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                y_pred = self.model(x_batch)
                predictions.append(y_pred.cpu().numpy())
                targets.append(y_batch.cpu().numpy())

        y_preds = np.concatenate(predictions, axis=0)
        y_trues = np.concatenate(targets, axis=0)

        print(f"predictii - Shape: {y_preds.shape}")

        if save_dir and appliance_cols:
            os.makedirs(save_dir, exist_ok=True)
            timestamps = np.load(os.path.join(nilm_dir, "timestamps.npy"), allow_pickle=True)
            test_timestamps = timestamps[-len(y_preds):]

            for i, name in enumerate(appliance_cols):
                records = []
                index = 0
                for s in range(y_preds.shape[0]):
                    for t in range(y_preds.shape[1]):
                        if index >= len(test_timestamps):
                            continue
                        records.append({
                            "timestamp": test_timestamps[index],
                            "prediction": y_preds[s, t, i],
                            "actual": y_trues[s, t, i]
                        })
                        index += 1
                df = pd.DataFrame(records)
                file_path = os.path.join(save_dir, f"{name}_test_predictions.csv")
                df.to_csv(file_path, index=False)
                print(f"salvata pred. pentru {name}: {file_path}")

        return y_preds, y_trues

    import os

    def evaluate_all_appliances(self, predictions_dir, save_csv_path):
        results = []
        metrics_folder = os.path.join(predictions_dir, "metrics")
        os.makedirs(metrics_folder, exist_ok=True)

        for filename in os.listdir(predictions_dir):
            if not filename.endswith("_test_predictions.csv"):
                continue

            appliance = filename.replace("_test_predictions.csv", "")
            path = os.path.join(predictions_dir, filename)

            df = pd.read_csv(path)
            predictions = df["prediction"].values
            actuals = df["actual"].values

            analyzer = ErrorMetricsAnalyzer(predictions=predictions, actuals=actuals, output_path=None)
            metrics = analyzer.compute_all_metrics()
            metrics["Appliance"] = appliance
            results.append(metrics)

            #salvam metrici per canal
            df_channel = pd.DataFrame([metrics])
            df_channel = df_channel[["Appliance", "MAE", "RMSE", "MSE", "R2 Score", "MAPE", "SMAPE"]]
            channel_path = os.path.join(metrics_folder, f"{appliance}_metrics.csv")
            df_channel.to_csv(channel_path, index=False)
            print(f"Metrici salvate pentru {appliance}: {channel_path}")

        df_all = pd.DataFrame(results)
        df_all = df_all[["Appliance", "MAE", "RMSE", "MSE", "R2 Score", "MAPE", "SMAPE"]]
        df_all.to_csv(save_csv_path, index=False)
        print(f"Metrici agregate salvate la: {save_csv_path}")
        return df_all

    def plot_predictions_per_appliance(self, y_preds, y_trues, appliance_cols, save_dir, max_samples=3):
        os.makedirs(save_dir, exist_ok=True)

        start_times = pd.date_range("2013-10-01", periods=max_samples, freq="7D")
        timestamps_all = [pd.date_range(start=start, periods=y_preds.shape[1], freq="H") for start in start_times]

        for i, appliance in enumerate(appliance_cols):
            plt.figure(figsize=(18, 6))

            for sample_id in range(min(max_samples, y_preds.shape[0])):
                pred = y_preds[sample_id, :, i]
                true = y_trues[sample_id, :, i]
                time = timestamps_all[sample_id]

                # Actual cu albastru, Prediction cu portocaliu
                plt.plot(time, true, color='tab:blue', linewidth=2, label="Actual" if sample_id == 0 else "")
                plt.plot(time, pred, color='tab:orange', linestyle='--', label="Predicted" if sample_id == 0 else "")

            plt.title(f"Predictii vs Valori Reale - {appliance}")
            plt.xlabel("Timp")
            plt.ylabel("Consum (Power)")
            plt.legend()
            plt.tight_layout()

            plot_path = os.path.join(save_dir, f"{appliance}_combined_samples.png")
            plt.savefig(plot_path)
            plt.close()
            print(f"plot salvat la: {plot_path}")


def create_nilm_sequences( csv_path, window_size=168):
        df = pd.read_csv(csv_path, parse_dates=['timestamp'])
        df.set_index('timestamp', inplace=True)

        df['hour_of_day'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek

        df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        for lag in [1, 2, 6, 24]:
            df[f'lag_{lag}h'] = df['channel_1'].shift(lag)

        df['roc_1h'] = df['channel_1'].diff(1)
        df['roc_3h'] = df['channel_1'].diff(3)

        df['rolling_mean_24h'] = df['channel_1'].rolling(window=24).mean()
        df['rolling_std_24h'] = df['channel_1'].rolling(window=24).std()
        df['rolling_max_24h'] = df['channel_1'].rolling(window=24).max()

        #eliminam valorile lipsa
        df = df.dropna()

        #separam X si Y
        appliance_cols = [col for col in df.columns if col.startswith("channel_") and col != "channel_1"]
        selected_features = [
            'channel_1', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'lag_1h', 'lag_2h', 'lag_6h', 'lag_24h',
            'roc_1h', 'roc_3h',
            'rolling_mean_24h', 'rolling_std_24h', 'rolling_max_24h'
        ]

        X_list, Y_list = [], []

        for i in range(len(df) - window_size):
            y_window = df.iloc[i:i + window_size][appliance_cols].values
            if np.sum(y_window) < 1e-3:
                continue

            x_window = df.iloc[i:i + window_size][selected_features].values
            X_list.append(x_window)
            Y_list.append(y_window)

        X = np.array(X_list)
        Y = np.array(Y_list)

        X_mean = X.mean()
        X_std = X.std() + 1e-6
        X = (X - X_mean) / X_std

        return X, Y, appliance_cols, df.index[-len(X):]


if __name__ == "__main__":
    nilm_dir = "C:\\Users\\elecf\\Desktop\\Licenta\\Date\\UK-DALE-disaggregated\\house_1\\downsampled\\1H\\NILM"
    csv_dataset_path = os.path.join(nilm_dir, "nilm_dataset.csv")  # CSV cu canale sincronizate

    X, Y, appliance_cols, timestamps = create_nilm_sequences(csv_dataset_path, window_size=168)
    np.save(os.path.join(nilm_dir, "timestamps.npy"), timestamps)

    np.save(os.path.join(nilm_dir, "X_total.npy"), X)
    np.save(os.path.join(nilm_dir, "Y_appliances.npy"), Y)

    print(f"Shape X: {X.shape}, Shape Y: {Y.shape}")
    path_model = os.path.join(nilm_dir, "model_nilm_lstm.pt")
    csv_dir = os.path.join(nilm_dir, "predictions_per_channel")
    plot_dir = os.path.join(nilm_dir, "prediction_plots")

    input_size = X.shape[2]
    num_appliances = Y.shape[2]
    model = LSTMDecomposer(input_size=input_size, hidden_size=512, output_size=num_appliances)

    x_path = os.path.join(nilm_dir, "X_total.npy")
    y_path = os.path.join(nilm_dir, "Y_appliances.npy")

    trainer = NILMTrainer(
        x_path=x_path,
        y_path=y_path,
        model=model,
        batch_size=64,
        lr=1e-3,
        num_epochs=200,
        save_path=path_model
    )

    trainer.train()

    appliance_cols = [f"channel_{i}" for i in range(2, 2 + num_appliances)]

    # predictie + salvare CSV
    y_preds, y_trues = trainer.predict(
        appliance_cols=appliance_cols,
        save_dir=csv_dir
    )

    trainer.plot_predictions_per_appliance(
        y_preds=y_preds,
        y_trues=y_trues,
        appliance_cols=appliance_cols,
        save_dir=plot_dir,
        max_samples=3
    )

    np.save(os.path.join(nilm_dir, "y_preds.npy"), y_preds)
    np.save(os.path.join(nilm_dir, "y_trues.npy"), y_trues)
    print(f"Predictii salvate in {nilm_dir}")

    csv_metrics_path = os.path.join(nilm_dir, "nilm_evaluation.csv")

    trainer.evaluate_all_appliances(
        predictions_dir=csv_dir,
        save_csv_path=csv_metrics_path
    )
