from flask import Blueprint, jsonify
import os
import pandas as pd

from services.KANService import calculate_forecast_metrics_kan, run_forecast_for_single_channel_kan, \
    run_total_forecast_for_day_kan
from services.LSTMService import run_total_forecast_for_day_lstm, run_forecast_for_single_channel_lstm, \
    calculate_forecast_metrics_lstm

forecast_bp = Blueprint("forecast", __name__)

BASELINE_SEASONAL_DIR = r'C:\Users\elecf\Desktop\Licenta\Date\UK-DALE-disaggregated\house_1\baseline\seasonal'
BASE_DIR = r"C:\Users\elecf\Desktop\Licenta\Date\UK-DALE-disaggregated\house_1"

@forecast_bp.route("/baseline/seasonal/preview/<int:channel_id>", methods=["GET"])
def preview_seasonal_baseline(channel_id):
    channel_name = f"channel_{channel_id}"
    file_path = os.path.join(BASELINE_SEASONAL_DIR, f"baseline_seasonal_{channel_name}.csv")

    if not os.path.exists(file_path):
        return jsonify({"error": f"Fisierul baseline pentru {channel_name} nu exista."}), 404

    try:
        df = pd.read_csv(file_path, parse_dates=["timestamp"])

        count_10_percent = max(1, int(len(df) * 0.1))
        preview_df = df.tail(count_10_percent)

        return jsonify(preview_df.to_dict(orient="records"))

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@forecast_bp.route("/predictieLSTM/<int:channel_id>/<date_str>", methods=["GET"])
def get_prediction_for_day_lstm(channel_id, date_str):

    forecast_root = os.path.join(BASE_DIR, "predictii_viitor", date_str)

    path_total = os.path.join(forecast_root, "combinat", "LSTM", "forecast_total_lstm.csv")
    path_single = os.path.join(forecast_root, "csv", "LSTM", f"forecast_channel_{channel_id}.csv")

    try:
        if channel_id == 1:
            if not os.path.exists(path_total):
                run_total_forecast_for_day_lstm(date_str, BASE_DIR)

            df = pd.read_csv(path_total, parse_dates=['timestamp'])
            return jsonify(df[["timestamp", "total_predicted"]].to_dict(orient="records"))
        else:
            if not os.path.exists(path_single):
                df_forecast = run_forecast_for_single_channel_lstm(date_str, channel_id, BASE_DIR)
            else:
                df_forecast = pd.read_csv(path_single, parse_dates=['timestamp'])

            return jsonify(df_forecast[["timestamp", "predicted_power"]].to_dict(orient="records"))

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@forecast_bp.route("/metricsLSTM/<int:channel_id>/<date_str>", methods=["GET"])
def get_forecast_metrics_lstm(channel_id, date_str):
    try:
        BASE_DIR = r"C:\Users\elecf\Desktop\Licenta\Date\UK-DALE-disaggregated\house_1"
        metrics = calculate_forecast_metrics_lstm(channel_id, date_str, BASE_DIR)
        return jsonify(metrics)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@forecast_bp.route("/predictieKAN/<int:channel_id>/<date_str>", methods=["GET"])
def get_prediction_for_day_kan(channel_id, date_str):

    forecast_root = os.path.join(BASE_DIR, "predictii_viitor", date_str)

    path_total = os.path.join(forecast_root, "combinat", "KAN", "forecast_total_kan.csv")
    path_single = os.path.join(forecast_root, "csv", "KAN", f"forecast_channel_{channel_id}.csv")

    try:
        if channel_id == 1:
            if not os.path.exists(path_total):
                run_total_forecast_for_day_kan(date_str, BASE_DIR)

            df = pd.read_csv(path_total, parse_dates=['timestamp'])
            return jsonify(df[["timestamp", "total_predicted"]].to_dict(orient="records"))
        else:
            if not os.path.exists(path_single):
                df_forecast = run_forecast_for_single_channel_kan(date_str, channel_id, BASE_DIR)
            else:
                df_forecast = pd.read_csv(path_single, parse_dates=['timestamp'])

            return jsonify(df_forecast[["timestamp", "predicted_power"]].to_dict(orient="records"))

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@forecast_bp.route("/metricsKAN/<int:channel_id>/<date_str>", methods=["GET"])
def get_forecast_metrics_kan(channel_id, date_str):
    try:
        metrics = calculate_forecast_metrics_kan(channel_id, date_str, BASE_DIR)
        return jsonify(metrics)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500