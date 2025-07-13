""" KAN """
import os
import traceback

import pandas as pd
from matplotlib import pyplot as plt

from modelsAnalysis.KANAnalysis import KANAnalyzer
from services.auxiliarClasses.KAN_Model.KANForecast import KANForecaster
from services.ErrorMetricsService import ErrorMetricsAnalyzer

if __name__ == "__main__":
    base_dir = r'C:\Users\elecf\Desktop\Licenta\Date\UK-DALE-disaggregated\house_1'
    downsampled_dir = os.path.join(base_dir, "downsampled/1H")
    metrics_dir = os.path.join(base_dir, "metrics")
    predictii_dir = os.path.join(base_dir, "predictii")
    models_dir = os.path.join(base_dir, "modele_salvate")
    predictii_viitor_dir = os.path.join(base_dir, "predictii_viitor")
    plots_dir = os.path.join(base_dir, "plots")
    details_dir = os.path.join(base_dir, "details")
    baseline_dir = os.path.join(base_dir, "baseline")
    predictii_dir_kan = os.path.join(base_dir, "predictii")
    predictii_dir_kan = os.path.join(predictii_dir_kan, "KAN")

    metrics_dir_kan = os.path.join(base_dir, "metrics")
    metrics_dir_kan = os.path.join(metrics_dir_kan, "KAN")
    models_dir_kan = os.path.join(base_dir, "modele_salvate", "KAN")
    plots_dir_kan = os.path.join(base_dir, "plots", "KAN")
    scalers_dir_KAN = os.path.join(base_dir, "modele_salvate", "KAN", "scalers")
    os.makedirs(scalers_dir_KAN, exist_ok=True)

    for i in range(1, 1):
        channel_name = f"channel_{i}"
        print(f"\n Procesare KAN pentru: {channel_name}")

        try:
            channel_csv_path = os.path.join(downsampled_dir, f"{channel_name}_downsampled_1H.csv")

            if not os.path.isfile(channel_csv_path):
                print(f" Fisierul lipseste: {channel_csv_path}")
                continue

            kan_model_path = os.path.join(models_dir_kan, f"kan_model_{channel_name}.pth")
            kan_prediction_path = os.path.join(predictii_dir_kan, f"kan_predictions_{channel_name}.csv")
            kan_metrics_path = os.path.join(metrics_dir_kan, f"kan_metrics_{channel_name}.csv")
            plot_save_path = os.path.join(plots_dir_kan, f"plot_{channel_name}_KAN.png")

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

    target_day = "2014-12-4"
    window_size = 168

    base_output_path = r"C:\Users\elecf\Desktop\Licenta\Date\UK-DALE-disaggregated\house_1\predictii_viitor"
    target_folder = os.path.join(base_output_path, target_day)

    subdirs = {
        "combinat": os.path.join(target_folder, "combinat"),
        "csv_KAN": os.path.join(target_folder, "csv", "KAN"),
        "plots_KAN": os.path.join(target_folder, "plots", "KAN"),
        "metrics_KAN": os.path.join(target_folder, "metrics", "KAN")
    }

    for path in subdirs.values():
        os.makedirs(path, exist_ok=True)

    #fisiere output
    kan_combinat_dir = os.path.join(subdirs["combinat"], "KAN")
    os.makedirs(kan_combinat_dir, exist_ok=True)
    output_csv = os.path.join(kan_combinat_dir, "forecast_total_kan.csv")
    output_channel_csv_KAN = subdirs["csv_KAN"]
    plots_output_KAN = subdirs["plots_KAN"]
    metrics_output_KAN = subdirs["metrics_KAN"]

    #date reale pentru canalul 1
    channel_1_path = os.path.join(downsampled_dir, "channel_1_downsampled_1H.csv")
    channel_1_df = pd.read_csv(channel_1_path)
    channel_1_df['timestamp'] = pd.to_datetime(channel_1_df['timestamp'])
    channel_1_df.set_index('timestamp', inplace=True)
    actual_total = channel_1_df.loc[target_day]['power'].reset_index(drop=True)
    timestamps = channel_1_df.loc[target_day].index

    #predictii combinate
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

            channel_output_csv = os.path.join(output_channel_csv_KAN, f"forecast_{channel_name}.csv")
            df_forecast.to_csv(channel_output_csv, index=False)

            metrics_path_channel = os.path.join(metrics_output_KAN, f"forecast_{channel_name}_metrics.csv")

            analyzer = ErrorMetricsAnalyzer(
                predictions=df_forecast["predicted_power"].values,
                actuals=df_forecast["actual_power"].values,
                output_path=metrics_path_channel
            )
            analyzer.save_metrics()
            print(f"metrici salvate pentru {channel_name}: {metrics_path_channel}")

            #salvare grafic individual
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

            print(f"salvat pentru {channel_name}")

        except Exception as e:
            print(f"️eroare la {channel_name}: {str(e)}")
            traceback.print_exc()

    combined_df["total_predicted"] = sum(total_pred)
    combined_df["total_actual"] = actual_total.values

    combined_df.to_csv(output_csv, index=False)
    print(f"\n Fisier final salvat: {output_csv}")

    metrics_total_path = os.path.join(metrics_output_KAN, "forecast_total_metrics.csv")

    analyzer_total = ErrorMetricsAnalyzer(
        predictions=combined_df["total_predicted"].values,
        actuals=combined_df["total_actual"].values,
        output_path=metrics_total_path
    )
    analyzer_total.save_metrics()
    print(f"metrici totale KAN salvate: {metrics_total_path}")

    #grafic total
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

    total_plot_path = os.path.join(kan_combinat_dir, "total_forecast_plot_kan.png")
    plt.savefig(total_plot_path)
    plt.close()