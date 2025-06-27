import traceback

from matplotlib import pyplot as plt
import pandas as pd
from Services.DetailsService import decomposition_plot
from ModelsAnalysis.KANAnalysis import KANAnalyzer
from Services.AuxiliarClasses.KAN_Model.KANForecast import KANForecaster
from Services.ErrorMetricsService import ErrorMetricsAnalyzer
import os
from Services.BaselineService import BaselineGenerator

#from KAN_Model.KANAnalysis import KANAnalyzer
from Transformer_Model.TransformerAnalysis import TransformerAnalyzer

if __name__ == "__main__":
    # Setam directorul de baza
    base_dir = r'C:\Users\elecf\Desktop\Licenta\Date\UK-DALE-disaggregated\house_1'
    #labels_file = os.path.join(base_dir, 'labels.dat')

    #  Detectam automat canalele din fisierele .dat
    valid_channels = []
    for f in os.listdir(base_dir):
        if f.endswith(".dat") and "labels" not in f.lower():
            valid_channels.append(f)
    channels = valid_channels

    #  Cream sub-directoarele pentru organizarea fisierelor
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

    # Inițializam si preprocesam datele
    # analyzer = DataAnalyzer(house_dir=base_dir, labels_file=labels_file, channels=channels)
    # analyzer.load_labels()
    # analyzer.load_data()

    # for channel in channels:
    #analyzer.plot_acf_pacf(r'C:\Users\elecf\Desktop\Licenta\Date\UK-DALE-disaggregated\house_1\downsampled\1H\channel_1_downsampled_1H.csv')

    # Vizualizam datele pentru fiecare canal
    # analyzer.plot_time_series()

    # Calculam si salvam metricile generale
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

    #generare metrici fiecare aparat
    #metrics_channels(input_dir=downsampled_dir, output_dir=details_dir)

    # === DECOMPOSITION
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

    # === BASELINE: Moving Average (24h) si Last Week (168h)
    for i in range(1, 1):
        channel_name = f"channel_{i}"
        input_csv = os.path.join(downsampled_dir, f"{channel_name}_downsampled_1H.csv")

        try:
            # === MOVING AVERAGE ===
            output_csv_ma = os.path.join(baseline_day_dir, f"baseline_movingavg_{channel_name}.csv")
            metrics_csv_ma = os.path.join(baseline_day_dir, f"metrics_movingavg_{channel_name}.csv")
            BaselineGenerator.moving_average(input_csv, output_csv_ma)
            df_ma = pd.read_csv(output_csv_ma)
            ErrorMetricsAnalyzer(df_ma['prediction'], df_ma['actual'], metrics_csv_ma).save_metrics()

            # === LAST WEEK ===
            output_csv_lw = os.path.join(baseline_week_dir, f"baseline_lastweek_{channel_name}.csv")
            metrics_csv_lw = os.path.join(baseline_week_dir, f"metrics_lastweek_{channel_name}.csv")
            BaselineGenerator.last_week(input_csv, output_csv_lw)
            df_lw = pd.read_csv(output_csv_lw)
            ErrorMetricsAnalyzer(df_lw['prediction'], df_lw['actual'], metrics_csv_lw).save_metrics()

            # === SEASONAL HOURLY ===
            output_csv_s = os.path.join(baseline_seasonal_dir, f"baseline_seasonal_{channel_name}.csv")
            metrics_csv_s = os.path.join(baseline_seasonal_dir, f"metrics_seasonal_{channel_name}.csv")
            BaselineGenerator.seasonal_hourly(input_csv, output_csv_s)
            df_s = pd.read_csv(output_csv_s)
            ErrorMetricsAnalyzer(df_s['prediction'], df_s['actual'], metrics_csv_s).save_metrics()

            print(f" Toate baseline-urile calculate pentru {channel_name}")

        except Exception as e:
            print(f" Eroare la baseline pentru {channel_name}: {str(e)}")

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
    print(f"Comparatie completa salvata: {output_path}")

    """ KAN """
    predictii_dir_kan = os.path.join(base_dir, "predictii")
    predictii_dir_kan = os.path.join(predictii_dir_kan, "KAN")

    metrics_dir_kan = os.path.join(base_dir, "metrics")
    metrics_dir_kan = os.path.join(metrics_dir_kan, "KAN")
    models_dir_kan = os.path.join(base_dir, "modele_salvate", "KAN")
    plots_dir_kan = os.path.join(base_dir, "plots", "KAN")
    scalers_dir_KAN = os.path.join(base_dir, "modele_salvate", "KAN", "scalers")
    os.makedirs(scalers_dir_KAN, exist_ok=True)
    #  Iteram prin toate canalele
    for i in range(1, 1):
        channel_name = f"channel_{i}"
        print(f"\n Procesare KAN pentru: {channel_name}")

        try:
            # Fisierele pentru acest canal
            channel_csv_path = os.path.join(downsampled_dir, f"{channel_name}_downsampled_1H.csv")

            # Verificam daca fisierul exista
            if not os.path.isfile(channel_csv_path):
                print(f" Fisierul lipseste: {channel_csv_path}")
                continue

            kan_model_path = os.path.join(models_dir_kan, f"kan_model_{channel_name}.pth")
            kan_prediction_path = os.path.join(predictii_dir_kan, f"kan_predictions_{channel_name}.csv")
            kan_metrics_path = os.path.join(metrics_dir_kan, f"kan_metrics_{channel_name}.csv")
            plot_save_path = os.path.join(plots_dir_kan, f"plot_{channel_name}_KAN.png")

            # Initializam si rulam modelul
            kan_analyzer = KANAnalyzer(csv_path=channel_csv_path, channel_number = i, scaler_dir=scalers_dir_KAN,)
            kan_analyzer.train(model_path=kan_model_path)

            # Predictii
            df_results = kan_analyzer.predict()
            df_results.to_csv(kan_prediction_path, index=False)
            print(f" Predictii salvate: {kan_prediction_path}")

            # Metrici
            error_analyzer = ErrorMetricsAnalyzer(
                predictions=df_results["prediction"].values,
                actuals=df_results["actual"].values,
                output_path=kan_metrics_path
            )
            error_analyzer.save_metrics()
            print(f" Metrici salvate: {kan_metrics_path}")

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
            print(f" Plot salvat: {plot_save_path}")

        except Exception as e:
            print(f" Eroare la {channel_name}: {str(e)}")

    """FORECAST KAN"""

    # === Config de baza ===
    target_day = "2014-11-22"
    window_size = 168

    # === Folder root pentru predictii ===
    base_output_path = r"C:\Users\elecf\Desktop\Licenta\Date\UK-DALE-disaggregated\house_1\predictii_viitor"
    target_folder = os.path.join(base_output_path, target_day)

    # === Structura directoare ===
    subdirs = {
        "combinat": os.path.join(target_folder, "KAN/combinat"),
        "csv_KAN": os.path.join(target_folder, "csv", "KAN"),
        "plots_KAN": os.path.join(target_folder, "plots", "KAN"),
        "metrics_KAN": os.path.join(target_folder, "metrics", "KAN")
    }

    # Creaza toate folderele necesare
    for path in subdirs.values():
        os.makedirs(path, exist_ok=True)

    # === Fisiere output ===
    output_csv = os.path.join(subdirs["combinat"], "forecast_total.csv")
    output_channel_csv_KAN = subdirs["csv_KAN"]
    plots_output_KAN = subdirs["plots_KAN"]
    metrics_output_KAN = subdirs["metrics_KAN"]

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
        kan_model_path = os.path.join(models_dir_kan, f"kan_model_{channel_name}.pth")

        print(f" Forecast pentru {channel_name}")

        try:
            forecaster = KANForecaster(
                model_path=kan_model_path,
                csv_path=channel_csv_path,
                window_size=168,
                scaler_dir=scalers_dir_KAN,
                channel_number=i,
            )

            df_forecast = forecaster.rolling_forecast_day(target_day)

            combined_df[f"{channel_name}_predicted"] = df_forecast["predicted_power"]
            combined_df[f"{channel_name}_actual"] = df_forecast["actual_power"]

            total_pred.append(df_forecast["predicted_power"].values)
            total_actual.append(df_forecast["actual_power"].values)

            # Salvare CSV
            channel_output_csv = os.path.join(output_channel_csv_KAN, f"forecast_{channel_name}.csv")
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

            plot_path = os.path.join(plots_output_KAN, f"plot_{channel_name}.png")
            plt.savefig(plot_path)
            plt.close()

            print(f" Salvat pentru {channel_name}")

        except Exception as e:
            print(f"️ Eroare la {channel_name}: {str(e)}")
            traceback.print_exc()

    # Totaluri
    combined_df["total_predicted"] = sum(total_pred)
    combined_df["total_actual"] = actual_total.values

    # Salvare finala
    combined_df.to_csv(output_csv, index=False)
    print(f"\n Fisier final salvat: {output_csv}")

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


    """ Transformer """
    predictii_dir_transformer = os.path.join(base_dir, "predictii")
    predictii_dir_transformer = os.path.join(predictii_dir_transformer, "Transformer")

    metrics_dir_transformer = os.path.join(base_dir, "metrics")
    metrics_dir_transformer = os.path.join(metrics_dir_transformer, "Transformer")
    models_dir_transformer = os.path.join(base_dir, "modele_salvate", "Transformer")
    plots_dir_transformer = os.path.join(base_dir, "plots", "Transformer")
    scalers_dir_transformer = os.path.join(base_dir, "modele_salvate", "Transformer", "scalers")
    os.makedirs(scalers_dir_transformer, exist_ok=True)

    #  Iteram prin toate aparatele
    for i in range(1, 1):
        channel_name = f"channel_{i}"
        print(f"\n Procesare Transformer pentru: {channel_name}")

        try:
            channel_csv_path = os.path.join(downsampled_dir, f"{channel_name}_downsampled_1H.csv")

            if not os.path.isfile(channel_csv_path):
                print(f" Fisierul lipseste: {channel_csv_path}")
                continue

            transformer_model_path = os.path.join(models_dir_transformer, f"transformer_model_{channel_name}.pth")
            transformer_prediction_path = os.path.join(predictii_dir_transformer, f"transformer_predictions_{channel_name}.csv")
            transformer_metrics_path = os.path.join(metrics_dir_transformer, f"transformer_metrics_{channel_name}.csv")
            plot_save_path = os.path.join(plots_dir_transformer, f"plot_{channel_name}_Transformer.png")

            # Initializam si rulam modelul
            transformer_analyzer = TransformerAnalyzer(
                csv_path=channel_csv_path,
                channel_number=i,
                scaler_dir=scalers_dir_transformer
            )
            transformer_analyzer.train(model_path=transformer_model_path)

            # Predictii
            df_results = transformer_analyzer.predict()
            df_results.to_csv(transformer_prediction_path, index=False)
            print(f" Predictii salvate: {transformer_prediction_path}")

            # Metrici
            error_analyzer = ErrorMetricsAnalyzer(
                predictions=df_results["prediction"].values,
                actuals=df_results["actual"].values,
                output_path=transformer_metrics_path
            )
            error_analyzer.save_metrics()
            print(f" Metrici salvate: {transformer_metrics_path}")

            # Plot
            plt.figure(figsize=(20, 6))
            plt.plot(df_results["timestamp"], df_results["actual"], label="Actual", linewidth=1.5)
            plt.plot(df_results["timestamp"], df_results["prediction"], label="Predicted", linewidth=1.5)
            plt.xlabel("Timp")
            plt.ylabel("Consum (Power)")
            plt.title(f"Predictii Transformer vs Valori Reale - {channel_name}")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(plot_save_path)
            plt.close()
            print(f" Plot salvat: {plot_save_path}")

        except Exception as e:
            print(f" Eroare la {channel_name}: {str(e)}")