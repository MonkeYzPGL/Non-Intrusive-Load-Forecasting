from matplotlib import pyplot as plt
import pandas as pd

from services.DetailsService import decomposition_plot
from services.ErrorMetricsService import ErrorMetricsAnalyzer
import os
from services.BaselineService import BaselineGenerator

if __name__ == "__main__":
    base_dir = r'C:\Users\elecf\Desktop\Licenta\Date\UK-DALE-disaggregated\house_1'
    # labels_file = os.path.join(base_dir, 'labels.dat')
    #
    # # detectam automat canalele din fisierele .dat
    # valid_channels = []
    # for f in os.listdir(base_dir):
    #     if f.endswith(".dat") and "labels" not in f.lower():
    #         valid_channels.append(f)
    # channels = valid_channels

    #cream sub-directoarele pentru organizarea fisierelor
    downsampled_dir = os.path.join(base_dir, "downsampled")
    metrics_dir = os.path.join(base_dir, "metrics")
    predictii_dir = os.path.join(base_dir, "predictii")
    models_dir = os.path.join(base_dir, "modele_salvate")
    predictii_viitor_dir = os.path.join(base_dir, "predictii_viitor")
    plots_dir = os.path.join(base_dir, "plots")
    details_dir = os.path.join(base_dir, "details")
    baseline_dir = os.path.join(base_dir, "baseline")

    os.makedirs(downsampled_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(predictii_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(predictii_viitor_dir, exist_ok=True)
    os.makedirs(baseline_dir, exist_ok=True)

    #initializam si preprocesare date
    # analyzer = DataAnalyzer(house_dir=base_dir, labels_file=labels_file, channels=channels)
    # analyzer.load_labels()
    # analyzer.load_data()

    # for channel in channels:
    # analyzer.plot_acf_pacf(r'C:\Users\elecf\Desktop\Licenta\Date\UK-DALE-disaggregated\house_1\downsampled\1H\channel_1_downsampled_1H.csv')

    #vizualizam datele pentru fiecare channel
    # analyzer.plot_time_series()

    #calculam si salvam metricile generale
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


    # plot_analyzer = PlotAnalyzer(data_dict=analyzer.data_dict, labels=analyzer.labels)

    # plot_analzyzer.plot_histograms()  # Histogramele pentru distributia consumului
    # plot_analyzer.plot_correlograms()  # Corelograme pentru analiza corelatiilor

    #salvam datele agregate si reducem granularitatea
    #aggregation_analyzer = AggregationAnalyzer(data_dict=analyzer.data_dict, labels=analyzer.labels)
    downsampled_dir = os.path.join(downsampled_dir, "1H")
    #aggregation_analyzer.save_downsampled_data(freq='1H', output_dir=downsampled_dir)
    #aggregation_analyzer.generate_aggregated(downsampled_dir)

    #DECOMPOSITION
    decomposition_dir = os.path.join(base_dir, "decomposition")
    os.makedirs(decomposition_dir, exist_ok=True)

    for i in range(1, 1):
        channel_name = f"channel_{i}"
        csv_path = os.path.join(downsampled_dir, f"{channel_name}_downsampled_1H.csv")

        try:
            decomposition_plot(
                input_path=csv_path,
                output_dir=decomposition_dir,
                channel_name=channel_name
            )
        except Exception as e:
            print(f" Eroare la decompunere pentru {channel_name}: {str(e)}")

    baseline_day_dir = os.path.join(baseline_dir, "last_day")
    baseline_week_dir = os.path.join(baseline_dir, "last_week")
    baseline_seasonal_dir = os.path.join(baseline_dir, "seasonal")
    os.makedirs(baseline_day_dir, exist_ok=True)
    os.makedirs(baseline_week_dir, exist_ok=True)
    os.makedirs(baseline_seasonal_dir, exist_ok=True)

    #BASELINE
    for i in range(1,1):
        channel_name = f"channel_{i}"
        input_csv = os.path.join(downsampled_dir, f"{channel_name}_downsampled_1H.csv")

        try:
            #MOVING AVERAGE
            output_csv_ma = os.path.join(baseline_day_dir, f"baseline_movingavg_{channel_name}.csv")
            metrics_csv_ma = os.path.join(baseline_day_dir, f"metrics_movingavg_{channel_name}.csv")
            BaselineGenerator.moving_average(input_csv, output_csv_ma)
            df_ma = pd.read_csv(output_csv_ma)
            ErrorMetricsAnalyzer(df_ma['prediction'], df_ma['actual'], metrics_csv_ma).save_metrics()

            #LAST WEEK
            output_csv_lw = os.path.join(baseline_week_dir, f"baseline_lastweek_{channel_name}.csv")
            metrics_csv_lw = os.path.join(baseline_week_dir, f"metrics_lastweek_{channel_name}.csv")
            BaselineGenerator.last_week(input_csv, output_csv_lw)
            df_lw = pd.read_csv(output_csv_lw)
            ErrorMetricsAnalyzer(df_lw['prediction'], df_lw['actual'], metrics_csv_lw).save_metrics()

            #SEASONAL HOURLY
            output_csv_s = os.path.join(baseline_seasonal_dir, f"baseline_seasonal_{channel_name}.csv")
            metrics_csv_s = os.path.join(baseline_seasonal_dir, f"metrics_seasonal_{channel_name}.csv")
            BaselineGenerator.seasonal_hourly(input_csv, output_csv_s)
            df_s = pd.read_csv(output_csv_s)
            ErrorMetricsAnalyzer(df_s['prediction'], df_s['actual'], metrics_csv_s).save_metrics()

            print(f"toate baseline-urile calculate pentru {channel_name}")

        except Exception as e:
            print(f"eroare la baseline pentru {channel_name}: {str(e)}")

    output_comparatie = os.path.join(baseline_dir, "comparatie_baseline.csv")

    metrics_to_extract = ["MAE", "RMSE", "MAPE", "SMAPE", "R2 Score"]
    rows = []

    for i in range(1, 1):
        channel = f"channel_{i}"
        row = {"Channel": channel}

        for name, folder, prefix in [
            ("MovingAvg", baseline_day_dir, "metrics_movingavg"),
            ("LastWeek", baseline_week_dir, "metrics_lastweek"),
            ("Seasonal", baseline_seasonal_dir, "metrics_seasonal"),
        ]:
            filepath = os.path.join(folder, f"{prefix}_{channel}.csv")
            try:
                df = pd.read_csv(filepath)
                for metric in metrics_to_extract:
                    val = df[df["Metric"] == metric]["Value"].values[0]
                    row[f"{metric}_{name}"] = val
            except Exception as e:
                print(f"Eroare la {filepath}: {e}")
                for metric in metrics_to_extract:
                    row[f"{metric}_{name}"] = None

        rows.append(row)

    output_path = os.path.join(baseline_dir, "comparatie_baseline.csv")
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"comparatie completa salvata: {output_path}")