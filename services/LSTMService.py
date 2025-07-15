import matplotlib.pyplot as plt
import numpy as np

from services.auxiliarClasses.LSTM_Model.LSTMForecast import LSTMForecaster
import os
import pandas as pd
from services.ErrorMetricsService import ErrorMetricsAnalyzer

def run_total_forecast_for_day_lstm(target_day, base_dir):

    downsampled_dir = os.path.join(base_dir, "downsampled", "1H")
    lstm_model_dir = os.path.join(base_dir, "modele_salvate", "LSTM")
    scalers_dir_LSTM = os.path.join(lstm_model_dir, "scalers")
    base_output_path = os.path.join(base_dir, "predictii_viitor")
    target_folder = os.path.join(base_output_path, target_day)

    subdirs = {
        "combinat": os.path.join(target_folder, "combinat"),
        "csv_LSTM": os.path.join(target_folder, "csv", "LSTM"),
        "plots_LSTM": os.path.join(target_folder, "plots", "LSTM"),
        "metrics_LSTM": os.path.join(target_folder, "metrics", "LSTM")
    }
    for path in subdirs.values():
        os.makedirs(path, exist_ok=True)

    lstm_combinat_dir = os.path.join(subdirs["combinat"], "LSTM")
    os.makedirs(lstm_combinat_dir, exist_ok=True)
    output_csv = os.path.join(lstm_combinat_dir, "forecast_total_lstm.csv")
    output_channel_csv_LSTM = subdirs["csv_LSTM"]
    plots_output_LSTM = subdirs["plots_LSTM"]

    channel_1_path = os.path.join(downsampled_dir, "channel_1_downsampled_1H.csv")
    channel_1_df = pd.read_csv(channel_1_path)
    channel_1_df['timestamp'] = pd.to_datetime(channel_1_df['timestamp'])
    channel_1_df.set_index('timestamp', inplace=True)
    actual_total = channel_1_df.loc[target_day]['power'].reset_index(drop=True)
    timestamps = channel_1_df.loc[target_day].index

    combined_df = pd.DataFrame({'timestamp': timestamps})
    total_pred = []
    canale_valide = []
    total_pred = []

    for i in range(2, 54):
        try:
            channel_name = f"channel_{i}"
            channel_csv_path = os.path.join(downsampled_dir, f"{channel_name}_downsampled_1H.csv")
            model_path = os.path.join(lstm_model_dir, f"lstm_model_{channel_name}.pth")

            forecaster = LSTMForecaster(
                model_path=model_path,
                csv_path=channel_csv_path,
                window_size=168,
                scaler_dir=scalers_dir_LSTM,
                channel_number=i
            )

            df_forecast = forecaster.rolling_forecast_day(target_day)

            #ignora channel-urile incomplete
            if df_forecast.shape[0] != 24:
                print(f"{channel_name} are doar {df_forecast.shape[0]} ore, ignorat.")
                continue

            combined_df[f"{channel_name}_predicted"] = df_forecast["predicted_power"]
            total_pred.append(df_forecast["predicted_power"].values)

            channel_output_csv = os.path.join(output_channel_csv_LSTM, f"forecast_{channel_name}.csv")
            df_forecast.to_csv(channel_output_csv, index=False)

            plot_path = os.path.join(plots_output_LSTM, f"plot_{channel_name}.png")
            plt.figure(figsize=(10, 4))
            plt.plot(df_forecast["timestamp"], df_forecast["actual_power"], label="Actual", color="red")
            plt.plot(df_forecast["timestamp"], df_forecast["predicted_power"], label="Predicted", color="blue")
            plt.title(f"Forecast vs Actual - {channel_name}")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()

        except Exception as e:
            print(f"Eroare la {channel_name}: {e}")

    combined_df["total_predicted"] = np.sum(total_pred, axis=0)
    combined_df["total_actual"] = actual_total.values
    combined_df.to_csv(output_csv, index=False)

def run_forecast_for_single_channel_lstm(target_day, channel_id, base_dir):

    downsampled_dir = os.path.join(base_dir, "downsampled", "1H")
    lstm_model_dir = os.path.join(base_dir, "modele_salvate", "LSTM")
    scalers_dir = os.path.join(lstm_model_dir, "scalers")

    forecast_root = os.path.join(base_dir, "predictii_viitor", target_day)
    csv_output = os.path.join(forecast_root, "csv", "LSTM")
    plot_output = os.path.join(forecast_root, "plots", "LSTM")
    metrics_output = os.path.join(forecast_root, "metrics", "LSTM")

    os.makedirs(metrics_output, exist_ok=True)
    os.makedirs(csv_output, exist_ok=True)
    os.makedirs(plot_output, exist_ok=True)

    channel_name = f"channel_{channel_id}"
    csv_path = os.path.join(downsampled_dir, f"{channel_name}_downsampled_1H.csv")
    model_path = os.path.join(lstm_model_dir, f"lstm_model_{channel_name}.pth")
    output_csv = os.path.join(csv_output, f"forecast_{channel_name}.csv")
    plot_path = os.path.join(plot_output, f"plot_{channel_name}.png")

    forecaster = LSTMForecaster(
        model_path=model_path,
        csv_path=csv_path,
        window_size=168,
        scaler_dir=scalers_dir,
        channel_number=channel_id
    )

    df_forecast = forecaster.rolling_forecast_day(target_day)
    df_forecast.to_csv(output_csv, index=False)

    plt.figure(figsize=(10, 4))
    plt.plot(df_forecast["timestamp"], df_forecast["actual_power"], label="Actual", color="red")
    plt.plot(df_forecast["timestamp"], df_forecast["predicted_power"], label="Predicted", color="blue")
    plt.title(f"Forecast vs Actual - {channel_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    return df_forecast

def calculate_forecast_metrics_lstm(channel_id, date_str, base_dir):

    if channel_id == 1:
        forecast_path = os.path.join(base_dir, "predictii_viitor", date_str, "combinat", "LSTM", "forecast_total_lstm.csv")
        output_path = os.path.join(base_dir, "predictii_viitor", date_str, "metrics", "LSTM", "forecast_total_metrics.csv")
        actual_col = "total_actual"
        predicted_col = "total_predicted"
    else:
        forecast_path = os.path.join(base_dir, "predictii_viitor", date_str, "csv", "LSTM", f"forecast_channel_{channel_id}.csv")
        output_path = os.path.join(base_dir, "predictii_viitor", date_str, "metrics", "LSTM", f"forecast_channel_{channel_id}_metrics.csv")
        actual_col = "actual_power"
        predicted_col = "predicted_power"

    if not os.path.exists(forecast_path):
        raise FileNotFoundError(f"Fisierul de forecast lipseste: {forecast_path}")

    df = pd.read_csv(forecast_path)

    if actual_col not in df.columns or predicted_col not in df.columns:
        raise ValueError(f"Coloanele necesare lipsesc din fisier: {forecast_path}")

    analyzer = ErrorMetricsAnalyzer(
        predictions=df[predicted_col].values,
        actuals=df[actual_col].values,
        output_path=output_path
    )

    analyzer.save_metrics()
    return analyzer.compute_all_metrics()
