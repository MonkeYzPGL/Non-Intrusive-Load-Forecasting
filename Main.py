from Analysis.DataAnalysis import DataAnalyzer
from Analysis.PlotAnalysis import PlotAnalyzer
from Metrics.Metrics import MetricsAnalyzer
from Analysis.AggregationAnalysis import AggregationAnalyzer
from LSTM_Model.LSTMAnalysis import LSTMAnalyzer
from Metrics.ErrorMetrics import ErrorMetricsAnalyzer
import os
import pandas as pd

if __name__ == "__main__":
    # Definim directoarele principale
    base_dir = r'C:\Users\elecf\Desktop\Licenta\Date\UK-DALE-disaggregated\house_3'
    labels_file = os.path.join(base_dir, 'labels.dat')
    channels = ['channel_1.dat', 'channel_2.dat', 'channel_3.dat', 'channel_4.dat', 'channel_5.dat']

    # Cream sub-directoarele pentru organizarea fisierelor
    aggregated_dir = os.path.join(base_dir, "aggregated")
    downsampled_dir = os.path.join(base_dir, "downsampled")
    metrics_dir = os.path.join(base_dir, "metrics")
    predictii_dir = os.path.join(base_dir, "predictii")
    models_dir = os.path.join(base_dir, "modele_salvate")
    predictii_viitor_dir = os.path.join(base_dir, "predictii_viitor")

    os.makedirs(aggregated_dir, exist_ok=True)
    os.makedirs(downsampled_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(predictii_dir, exist_ok=True)

    # Initializam DataAnalyzer
    analyzer = DataAnalyzer(house_dir=base_dir, labels_file=labels_file, channels=channels)

    # Incarcam etichetele si datele
    analyzer.load_labels()
    analyzer.load_data()

    #for channel in channels:
       # analyzer.plot_acf_pacf(channel)

    # Vizualizam datele pentru fiecare canal
    #analyzer.plot_time_series()

    # Calculam si afisam metricile generale pentru fiecare canal
    metrics_analyzer = MetricsAnalyzer(data_dict=analyzer.data_dict, labels=analyzer.labels)

    # Salvam metricile generale in `metrics/`
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
    #plot_analyzer = PlotAnalyzer(data_dict=analyzer.data_dict, labels=analyzer.labels)

    #plot_analzyzer.plot_histograms()  # Histogramele pentru distributia consumului
    #plot_analyzer.plot_correlograms()  # Corelograme pentru analiza corelatiilor

    # Analizam agregarea datelor
    aggregation_analyzer = AggregationAnalyzer(data_dict=analyzer.data_dict, labels=analyzer.labels)

    # Salvam datele agregate si cele cu granularitate redusa in directoarele respective
    #aggregation_analyzer.save_aggregated_data(freq='D', output_dir=aggregated_dir)
    aggregation_analyzer.save_downsampled_data(freq='1T', output_dir=downsampled_dir)

    # Verificam daca exista fisierul cu datele downsampled pentru canalul 5
    #channel_5_downsampled_path = os.path.join(downsampled_dir, 'channel_4.dat_downsampled_1T.csv')

    lstm_analyzer = LSTMAnalyzer(house_dir = downsampled_dir, csv_path=None)
    lstm_analyzer.preprocess_data()

    # Antrenam modelul
    lstm_model_path = os.path.join(models_dir, 'lstm_model_total.pth')
    lstm_analyzer.train(model_path=lstm_model_path)

    # Generam predictii pentru consumul total
    predictions, actuals = lstm_analyzer.predict()

    # Salvam predictiile
    prediction_output_path = os.path.join(predictii_dir, 'power_total_predictions.csv')
    prediction_df = pd.DataFrame({'Predictions': predictions, 'Actuals': actuals})
    prediction_df.to_csv(prediction_output_path, index=False)
    print(f"✅ Predictions saved in: {prediction_output_path}")

    # Calculam și salvam metricile de eroare pentru predictia consumului total
    error_metrics_path = os.path.join(metrics_dir, "power_total_lstm_error_metrics.csv")
    error_metrics_analyzer = ErrorMetricsAnalyzer(predictions=predictions, actuals=actuals,
                                                  output_path=error_metrics_path)
    error_metrics_analyzer.save_metrics()
    print(f"✅ Error metrics saved in: {error_metrics_path}")
