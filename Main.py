from Analysis.DataAnalysis import DataAnalyzer
from Analysis.PlotAnalysis import PlotAnalyzer
from Metrics.Metrics import MetricsAnalyzer
from Analysis.AggregationAnalysis import AggregationAnalyzer
from LSTM_Model.LSTMAnalysis import LSTMAnalyzer
from Metrics.ErrorMetrics import ErrorMetricsAnalyzer
import os
import pandas as pd
from Analysis.DeltaCalculation import calculate_delta

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

    os.makedirs(aggregated_dir, exist_ok=True)
    os.makedirs(downsampled_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(predictii_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(predictii_viitor_dir, exist_ok=True)

    # 📌 Inițializăm și preprocesăm datele
    analyzer = DataAnalyzer(house_dir=base_dir, labels_file=labels_file, channels=channels)
    analyzer.load_labels()
    analyzer.load_data()

    # for channel in channels:
    # analyzer.plot_acf_pacf(channel)

    # Vizualizam datele pentru fiecare canal
    # analyzer.plot_time_series()

    # 📌 Calculăm și salvăm metricile generale
    metrics_analyzer = MetricsAnalyzer(data_dict=analyzer.data_dict, labels=analyzer.labels)
    general_metrics_path = os.path.join(metrics_dir, "general_metrics.csv")

    daily_avg = metrics_analyzer.calculate_daily_average()
    peaks = metrics_analyzer.identify_peaks()
    correlation = metrics_analyzer.calculate_correlation()

    metrics_analyzer.metrics_df = {
        'Daily Averages': daily_avg,
        'Peaks': peaks,
        'Correlations': correlation
    }
    metrics_analyzer.save_metrics(output_path=general_metrics_path)

    # Initializam PlotAnalyzer pentru vizualizari suplimentare
    # plot_analyzer = PlotAnalyzer(data_dict=analyzer.data_dict, labels=analyzer.labels)

    # plot_analzyzer.plot_histograms()  # Histogramele pentru distributia consumului
    # plot_analyzer.plot_correlograms()  # Corelograme pentru analiza corelatiilor

    #  Salvăm datele agregate și reducem granularitatea
    aggregation_analyzer = AggregationAnalyzer(data_dict=analyzer.data_dict, labels=analyzer.labels)
    aggregation_analyzer.save_downsampled_data(freq='1T', output_dir=downsampled_dir)

    #  Calculăm diferențele între canale (delta)
    calculate_delta(downsampled_dir)

    #  Inițializăm și preprocesăm datele pentru LSTM
    lstm_analyzer = LSTMAnalyzer(house_dir=downsampled_dir)

    #  Antrenăm modelul
    lstm_model_path = os.path.join(models_dir, 'lstm_model_total.pth')
    lstm_analyzer.train(model_path=lstm_model_path)

    #  Generăm predicții pentru consumul total
    predictions, actuals = lstm_analyzer.predict()

    #  Salvăm predicțiile
    prediction_output_path = os.path.join(predictii_dir, 'power_total_predictions.csv')
    prediction_df = pd.DataFrame(predictions, columns=[f"Channel_{i+1}" for i in range(lstm_analyzer.num_channels)])
    prediction_df.to_csv(prediction_output_path, index=False)
    print(f"✅ Predictions saved in: {prediction_output_path}")

    #  Calculăm și salvăm metricile de eroare pentru predictii
    error_metrics_path = os.path.join(metrics_dir, "power_total_lstm_error_metrics.csv")
    error_metrics_analyzer = ErrorMetricsAnalyzer(predictions=predictions, actuals=actuals, output_path=error_metrics_path)
    error_metrics_analyzer.save_metrics()
    print(f"✅ Error metrics saved in: {error_metrics_path}")

    # **🔹 Generăm predicții în viitor pentru NILF**
    future_steps = 60  #  Prezicem consumul pentru următoarele 60 de minute
    future_predictions_path = os.path.join(predictii_viitor_dir, 'future_predictions.csv')
    lstm_analyzer.save_future_predictions(future_steps=future_steps, output_path=future_predictions_path)
    print(f"✅ Future predictions saved in: {future_predictions_path}")

    # **🔹 Vizualizăm predicțiile NILF**
    lstm_analyzer.plot_future_predictions(future_steps=future_steps)

    print("✅ NILF complet: Modelul a fost antrenat, predicțiile viitoare au fost salvate și vizualizate.")
