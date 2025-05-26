import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from Analysis.DataAnalysis import DataAnalyzer
from Analysis.PlotAnalysis import PlotAnalyzer
from KAN_Model.KANAnalysis import KANAnalyzer
from Metrics.Metrics import MetricsAnalyzer
from Analysis.AggregationAnalysis import AggregationAnalyzer
from LSTM_Model.LSTMAnalysis import LSTMAnalyzer
from Metrics.ErrorMetrics import ErrorMetricsAnalyzer
import os
import pandas as pd
from Analysis.DeltaCalculation import calculate_delta
#from KAN_Model.KANAnalysis import KANAnalyzer
from LSTM_Model.LSTMForecast import LSTMForecaster


if __name__ == "__main__":
    # 📌 Setăm directorul de bază (modifică-l dacă e necesar)
    base_dir = r'C:\Users\elecf\Desktop\Licenta\Date\UK-DALE-disaggregated\house_1'
    # labels_file = os.path.join(base_dir, 'labels.dat')
    #
    # # 📌 Detectăm automat canalele din fișierele .dat
    # valid_channels = []
    # for f in os.listdir(base_dir):
    #     if f.endswith(".dat") and "labels" not in f.lower():
    #         valid_channels.append(f)
    # channels = valid_channels

    # 📌 Cream sub-directoarele pentru organizarea fișierelor
    aggregated_dir = os.path.join(base_dir, "aggregated")
    downsampled_dir = os.path.join(base_dir, "downsampled")
    metrics_dir = os.path.join(base_dir, "metrics")
    predictii_dir = os.path.join(base_dir, "predictii")
    models_dir = os.path.join(base_dir, "modele_salvate")
    predictii_viitor_dir = os.path.join(base_dir, "predictii_viitor")  # 📌 Director pentru predicții viitoare
    plots_dir = os.path.join(base_dir, "plots")

    os.makedirs(aggregated_dir, exist_ok=True)
    os.makedirs(downsampled_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(predictii_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(predictii_viitor_dir, exist_ok=True)

    # 📌 Inițializăm și preprocesăm datele
    # analyzer = DataAnalyzer(house_dir=base_dir, labels_file=labels_file, channels=channels)
    # analyzer.load_labels()
    # analyzer.load_data()

    # for channel in channels:
    #analyzer.plot_acf_pacf(r'C:\Users\elecf\Desktop\Licenta\Date\UK-DALE-disaggregated\house_1\downsampled\1H\channel_1_downsampled_1H.csv')

    # Vizualizam datele pentru fiecare canal
    # analyzer.plot_time_series()

    # 📌 Calculăm și salvăm metricile generale
    # metrics_analyzer = MetricsAnalyzer(data_dict=analyzer.data_dict, labels=analyzer.labels)
    # general_metrics_path = os.path.join(metrics_dir, "general_metrics.csv")
    #
    # daily_avg = metrics_analyzer.calculate_daily_average()
    # peaks = metrics_analyzer.identify_peaks()
    # correlation = metrics_analyzer.calculate_correlation()
    #
    # metrics_analyzer.metrics_df = {
    #     'Daily Averages': daily_avg,
    #     'Peaks': peaks,
    #     'Correlations': correlation
    # }
    # metrics_analyzer.save_metrics(output_path=general_metrics_path)

    # Initializam PlotAnalyzer pentru vizualizari suplimentare
    # plot_analyzer = PlotAnalyzer(data_dict=analyzer.data_dict, labels=analyzer.labels)

    # plot_analzyzer.plot_histograms()  # Histogramele pentru distributia consumului
    # plot_analyzer.plot_correlograms()  # Corelograme pentru analiza corelatiilor

    #  Salvăm datele agregate și reducem granularitatea
    # aggregation_analyzer = AggregationAnalyzer(data_dict=analyzer.data_dict, labels=analyzer.labels)
    downsampled_dir = os.path.join(downsampled_dir, "1H")
    # aggregation_analyzer.save_downsampled_data(freq='1h', output_dir=downsampled_dir)
    # aggregation_analyzer.generate_aggregated(downsampled_dir)
    # # Calculăm diferențele între canale (delta)
    # # calculate_delta(downsampled_dir)

    predictii_dir_lstm = os.path.join(base_dir, "predictii")
    predictii_dir_lstm = os.path.join(predictii_dir_lstm, "LSTM")

    scalers_dir_LSTM = os.path.join(base_dir, "modele_salvate", "LSTM", "scalers")

    metrics_dir_lstm = os.path.join(base_dir, "metrics")
    metrics_dir_lstm = os.path.join(metrics_dir_lstm, "LSTM")
    plots_dir_lstm = os.path.join(plots_dir, "LSTM")
    lstm_model_dir = os.path.join(models_dir, "LSTM")

    """TEST LSTM"""
    for i in range(2, 2):
         channel_name = f"channel_{i}"

         channel_csv_path = os.path.join(downsampled_dir, f"{channel_name}_downsampled_1H.csv")
         lstm_model_path = os.path.join(lstm_model_dir, f"lstm_model_{channel_name}.pth")
         lstm_prediction_path = os.path.join(predictii_dir_lstm, f"lstm_predictions_{channel_name}.csv")
         lstm_metrics_path = os.path.join(metrics_dir_lstm, f"lstm_metrics_{channel_name}.csv")
         plot_save_path = os.path.join(plots_dir_lstm, f"plot_{channel_name}.png")

         print(f"\n📌 Rulare LSTM: {channel_name}")

         try:
             # Initializare obiect
             lstm_analyzer = LSTMAnalyzer(csv_path=channel_csv_path, scaler_dir=scalers_dir_LSTM,channel_number=i)

             # Antrenare model
             lstm_analyzer.train(model_path=lstm_model_path)

             # Predictii
             df_results = lstm_analyzer.predict()

             # Salvare predictii
             df_results.to_csv(lstm_prediction_path, index=False)
             print(f"✅ Predictii salvate: {lstm_prediction_path}")

             metrics_analyzer = ErrorMetricsAnalyzer(
                 predictions=df_results["prediction"].values,
                 actuals=df_results["actual"].values,
                 output_path=lstm_metrics_path,
             )
             metrics_analyzer.save_metrics()

             print(f"📊 Metrici salvate: {lstm_metrics_path}")

             # Salvare plot
             plt.figure(figsize=(20, 6))
             plt.plot(df_results["timestamp"], df_results["actual"], label="Actual", linewidth=1.5)
             plt.plot(df_results["timestamp"], df_results["prediction"], label="Predicted", linewidth=1.5)
             plt.xlabel("Timp")
             plt.ylabel("Consum (Power)")
             plt.title(f"Predictii LSTM vs Valori Reale - {channel_name}")
             plt.legend()
             plt.grid(True)
             plt.tight_layout()
             plt.savefig(plot_save_path)
             plt.close()

             print(f"🖼️ Plot salvat: {plot_save_path}")

         except Exception as e:
             print(f"❌ Eroare la {channel_name}: {str(e)}")

    """FORECAST"""
    # === Config de baza ===
    target_day = "2014-11-20"
    window_size = 168

    # === Folder root pentru predictii ===
    base_output_path = r"C:\Users\elecf\Desktop\Licenta\Date\UK-DALE-disaggregated\house_1\predictii_viitor"
    target_folder = os.path.join(base_output_path, target_day)

    # === Structura directoare ===
    subdirs = {
        "combinat": os.path.join(target_folder, "combinat"),
        "csv_LSTM": os.path.join(target_folder, "csv", "LSTM"),
        "plots_LSTM": os.path.join(target_folder, "plots", "LSTM"),
        "metrics_LSTM": os.path.join(target_folder, "metrics", "LSTM")
    }

    # Creaza toate folderele necesare
    for path in subdirs.values():
        os.makedirs(path, exist_ok=True)

    # === Fisiere output ===
    output_csv = os.path.join(subdirs["combinat"], "forecast_total.csv")
    output_channel_csv_LSTM = subdirs["csv_LSTM"]
    plots_output_LSTM = subdirs["plots_LSTM"]
    metrics_output_LSTM = subdirs["metrics_LSTM"]  # optional

    # === Date reale pentru canalul 1 ===
    channel_1_path = os.path.join(downsampled_dir, "channel_1_downsampled_1H.csv")
    channel_1_df = pd.read_csv(channel_1_path)
    channel_1_df['timestamp'] = pd.to_datetime(channel_1_df['timestamp'])
    channel_1_df.set_index('timestamp', inplace=True)
    actual_total = channel_1_df.loc[target_day]['power'].reset_index(drop=True)
    timestamps = channel_1_df.loc[target_day].index

    # === Predictii combinate ===
    combined_df = pd.DataFrame({'timestamp': timestamps})
    total_pred = []
    total_actual = []

    for i in range(2, 54):
        channel_name = f"channel_{i}"
        channel_csv_path = os.path.join(downsampled_dir, f"{channel_name}_downsampled_1H.csv")
        lstm_model_path = os.path.join(lstm_model_dir, f"lstm_model_{channel_name}.pth")

        print(f"📌 Forecast pentru {channel_name}")

        try:
            forecaster = LSTMForecaster(
                model_path=lstm_model_path,
                csv_path=channel_csv_path,
                window_size=window_size,
                scaler_dir=scalers_dir_LSTM,
                channel_number=i,
            )

            forecaster.load_recent_data()
            forecaster.load_model_and_scalers()
            df_forecast = forecaster.predict_day(target_day)

            combined_df[f"{channel_name}_predicted"] = df_forecast["predicted_power"]
            combined_df[f"{channel_name}_actual"] = df_forecast["actual_power"]

            total_pred.append(df_forecast["predicted_power"].values)
            total_actual.append(df_forecast["actual_power"].values)

            # Salvare CSV
            channel_output_csv = os.path.join(output_channel_csv_LSTM, f"forecast_{channel_name}.csv")
            df_forecast.to_csv(channel_output_csv, index=False)

            # Salvare grafic individual
            plt.figure(figsize=(10, 4))
            plt.plot(df_forecast["timestamp"], df_forecast["actual_power"], label="Actual", color="red")
            plt.plot(df_forecast["timestamp"], df_forecast["predicted_power"], label="Predicted", color="blue")
            plt.title(f"Forecast vs Actual - {channel_name}")
            plt.xlabel("Timestamp")
            plt.ylabel("Power")
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.grid(True)

            plot_path = os.path.join(plots_output_LSTM, f"plot_{channel_name}.png")
            plt.savefig(plot_path)
            plt.close()

            print(f"✅ Salvat pentru {channel_name}")

        except Exception as e:
            print(f"⚠️ Eroare la {channel_name}: {str(e)}")

    # Totaluri
    combined_df["total_predicted"] = sum(total_pred)
    combined_df["total_actual"] = actual_total.values

    # Salvare finala
    combined_df.to_csv(output_csv, index=False)
    print(f"\n✅ Fisier final salvat: {output_csv}")

    # Grafic total
    plt.figure(figsize=(12, 5))
    plt.plot(combined_df["timestamp"], combined_df["total_actual"], label="Total Actual", color="red")
    plt.plot(combined_df["timestamp"], combined_df["total_predicted"], label="Total Predicted", color="blue")
    plt.title("Total Predicted vs Total Actual")
    plt.xlabel("Timestamp")
    plt.ylabel("Power")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(True)

    # Salvare grafic total
    total_plot_path = os.path.join(subdirs["combinat"], "total_forecast_plot.png")
    plt.savefig(total_plot_path)
    plt.close()

    """ KAN """
    predictii_dir_kan = os.path.join(base_dir, "predictii")
    predictii_dir_kan = os.path.join(predictii_dir_kan, "KAN")

    metrics_dir_kan = os.path.join(base_dir, "metrics")
    metrics_dir_kan = os.path.join(metrics_dir_kan, "KAN")
    models_dir_kan = os.path.join(base_dir, "modele_salvate", "KAN")
    plots_dir_kan = os.path.join(base_dir, "plots", "KAN")
    scalers_dir = os.path.join(base_dir, "modele_salvate", "KAN", "scalers")
    os.makedirs(scalers_dir, exist_ok=True)
    #  Iteram prin toate canalele
    for i in range(2, 2):
        channel_name = f"channel_{i}"
        print(f"\n📌 Procesare KAN pentru: {channel_name}")

        try:
            # Fisierele pentru acest canal
            channel_csv_path = os.path.join(downsampled_dir, f"{channel_name}_downsampled_1H.csv")

            # Verificam daca fisierul exista
            if not os.path.isfile(channel_csv_path):
                print(f"⚠️ Fisierul lipseste: {channel_csv_path}")
                continue

            kan_model_path = os.path.join(models_dir, f"kan_model_{channel_name}.pth")
            kan_prediction_path = os.path.join(predictii_dir_kan, f"kan_predictions_{channel_name}.csv")
            kan_metrics_path = os.path.join(metrics_dir_kan, f"kan_metrics_{channel_name}.csv")
            plot_save_path = os.path.join(plots_dir_kan, f"plot_{channel_name}_KAN.png")

            # Initializam si rulam modelul
            kan_analyzer = KANAnalyzer(csv_path=channel_csv_path, channel_number = i)
            kan_analyzer.preprocess_data(scalers_dir=scalers_dir)
            kan_analyzer.train(model_path=kan_model_path)

            # Predictii
            df_results = kan_analyzer.predict()
            df_results.to_csv(kan_prediction_path, index=False)
            print(f"✅ Predictii salvate: {kan_prediction_path}")

            # Metrici
            error_analyzer = ErrorMetricsAnalyzer(
                predictions=df_results["prediction"].values,
                actuals=df_results["actual"].values,
                output_path=kan_metrics_path
            )
            error_analyzer.save_metrics()
            print(f"📊 Metrici salvate: {kan_metrics_path}")

            # Plot
            plt.figure(figsize=(20, 6))
            plt.plot(df_results["timestamp"], df_results["actual"], label="Actual", linewidth=1.5)
            plt.plot(df_results["timestamp"], df_results["prediction"], label="Predicted", linewidth=1.5)
            plt.xlabel("Timp")
            plt.ylabel("Consum (Power)")
            plt.title(f"Predictii KAN vs Valori Reale - {channel_name}")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(plot_save_path)
            plt.close()
            print(f"🖼️ Plot salvat: {plot_save_path}")

        except Exception as e:
            print(f"❌ Eroare la {channel_name}: {str(e)}")