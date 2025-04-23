import torch
from matplotlib import pyplot as plt

from Analysis.DataAnalysis import DataAnalyzer
from Analysis.PlotAnalysis import PlotAnalyzer
from Metrics.Metrics import MetricsAnalyzer
from Analysis.AggregationAnalysis import AggregationAnalyzer
from LSTM_Model.LSTMAnalysis import LSTMAnalyzer
from Metrics.ErrorMetrics import ErrorMetricsAnalyzer
import os
import pandas as pd
from Analysis.DeltaCalculation import calculate_delta
from KAN_Model.KANAnalysis import KANAnalyzer
from LSTM_Model.LSTMForecast import LSTMForecaster


if __name__ == "__main__":
    # 📌 Setăm directorul de bază (modifică-l dacă e necesar)
    base_dir = r'C:\Users\elecf\Desktop\Licenta\Date\UK-DALE-disaggregated\house_1'
    labels_file = os.path.join(base_dir, 'labels.dat')

    # 📌 Detectăm automat canalele din fișierele .dat
    valid_channels = []
    for f in os.listdir(base_dir):
        if f.endswith(".dat") and "labels" not in f.lower():
            valid_channels.append(f)
    channels = valid_channels

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
    #analyzer = DataAnalyzer(house_dir=base_dir, labels_file=labels_file, channels=channels)
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
    #  Calculăm diferențele între canale (delta)
    # calculate_delta(downsampled_dir)

    predictii_dir_lstm = os.path.join(base_dir, "predictii")
    predictii_dir_lstm = os.path.join(predictii_dir_lstm, "LSTM")

    metrics_dir_lstm = os.path.join(base_dir, "metrics")
    metrics_dir_lstm = os.path.join(metrics_dir_lstm, "LSTM")

    """TEST LSTM"""
    for i in range(1, 2):
         channel_name = f"channel_{i}"

         channel_csv_path = os.path.join(downsampled_dir, f"{channel_name}_downsampled_1H.csv")
         lstm_model_path = os.path.join(models_dir, f"lstm_model_{channel_name}.pth")
         lstm_ae_model_path = os.path.join(models_dir, f"ae_model_{channel_name}.pth")  # optional
         lstm_prediction_path = os.path.join(predictii_dir_lstm, f"lstm_predictions_{channel_name}.csv")
         lstm_metrics_path = os.path.join(metrics_dir_lstm, f"lstm_metrics_{channel_name}.csv")
         plot_save_path = os.path.join(plots_dir, f"plot_{channel_name}.png")

         print(f"\n📌 Rulare LSTM: {channel_name}")

         try:
             # Initializare obiect
             lstm_analyzer = LSTMAnalyzer(csv_path=channel_csv_path)

             # Antrenare model
             lstm_analyzer.train(model_path=lstm_model_path)

             # Predictii
             predictions, actuals, df_results = lstm_analyzer.predict()

             # Salvare predictii
             df_results.to_csv(lstm_prediction_path, index=False)
             print(f"✅ Predictii salvate: {lstm_prediction_path}")

             # Salvare metrici
             error_analyzer = ErrorMetricsAnalyzer(
                 predictions=df_results['prediction'],
                 actuals=df_results['actual'],
                 output_path=lstm_metrics_path
             )
             error_analyzer.save_metrics()
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

    # for i in range(1, 2):
    #     channel_name = f"channel_{i}"
    #
    #     channel_csv_path = os.path.join(downsampled_dir, f"{channel_name}_downsampled_1H.csv")
    #     lstm_model_path = os.path.join(models_dir, f"lstm_model_{channel_name}.pth")
    #     forecast_output_lstm = os.path.join(predictii_viitor_dir, "LSTM")
    #     forecast_output_lstm = os.path.join(forecast_output_lstm, f"forecast_{channel_name}.csv")
    #
    #     print(f"\n📌 Forecast LSTM pentru: {channel_name}")
    #
    #     try:
    #         # Initializare obiect forecaster
    #         forecaster = LSTMForecaster(
    #             model_path=lstm_model_path,
    #             csv_path=channel_csv_path,
    #             window_size=168
    #         )
    #
    #         # Incarcam ultimele date si setam scalerul + feature-urile
    #         forecaster.load_recent_data()
    #
    #         # Acum stim input_size => incarcam modelul corect
    #         forecaster.load_model()
    #
    #         # Generam forecast pe urmatoarele 48 de ore
    #         df_forecast = forecaster.forecast(num_steps=168)
    #
    #         # Salvam forecast-ul in CSV
    #         df_forecast.to_csv(forecast_output_lstm, index=False)
    #         print(f"✅ Forecast salvat: {forecast_output_lstm}")
    #
    #     except Exception as e:
    #         print(f"❌ Eroare la {channel_name}: {str(e)}")

    """ KAN """
    # predictii_dir_kan = os.path.join(base_dir, "predictii")
    # predictii_dir_kan = os.path.join(predictii_dir_kan, "KAN")
    #
    # metrics_dir_kan = os.path.join(base_dir, "metrics")
    # metrics_dir_kan = os.path.join(metrics_dir_kan, "KAN")
    #
    # #  Iteram prin toate canalele
    # for f in os.listdir(downsampled_dir):
    #     if not f.endswith("1H.csv"):
    #         continue
    #
    #     channel_name = f.replace("_downsampled_1H.csv", "")
    #     print(f"\n📌 Procesare KAN pentru: {channel_name}")
    #
    #     # 🔹 Fisierele pentru acest canal
    #     channel_csv_path = os.path.join(downsampled_dir, f)
    #     kan_model_path = os.path.join(models_dir, f"kan_model_{channel_name}.pth")
    #     kan_prediction_path = os.path.join(predictii_dir_kan, f"kan_predictions_{channel_name}.csv")
    #     kan_metrics_path = os.path.join(metrics_dir_kan, f"kan_metrics_{channel_name}.csv")
    #
    #     # 🔹 Initializam si rulam modelul
    #     kan_analyzer = KANAnalyzer(csv_path=channel_csv_path)
    #     kan_analyzer.preprocess_data()
    #     kan_analyzer.train(model_path=kan_model_path)
    #
    #     # 🔹 Predictii
    #     predictions, actuals, df_results = kan_analyzer.predict()
    #     df_results.to_csv(kan_prediction_path, index=False)
    #     print(f" Predictii salvate: {kan_prediction_path}" )
    #
    #     # 🔹 Metrice
    #     error_analyzer = ErrorMetricsAnalyzer(predictions=predictions, actuals=actuals, output_path=kan_metrics_path)
    #     error_analyzer.save_metrics()
    #     print(f" Metrici salvate: {kan_metrics_path}")

