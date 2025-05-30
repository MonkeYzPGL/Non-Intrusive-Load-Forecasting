from flask import Flask, send_file, jsonify
import os

from flask_cors import CORS

from Analysis.PlotAnalysis import generate_histogram, generate_correlogram
from Metrics.Metrics import metrics_channels, get_consumption_for_day
import pandas as pd

app = Flask(__name__)
CORS(app)

# Setari directoare
BASE_DIR = r'C:\Users\elecf\Desktop\Licenta\Date\UK-DALE-disaggregated\house_1'
DOWNSAMPLED_DIR = os.path.join(BASE_DIR, 'downsampled', '1H')
DETAILS_DIR = os.path.join(BASE_DIR, 'details')
HISTOGRAM_DIR = os.path.join(BASE_DIR, "analysis", "histogram")
LABELS_FILE = os.path.join(BASE_DIR, "labels.dat")

@app.route("/get_details/<int:channel_id>", methods=["GET"])
def get_channel_details(channel_id):
    channel_name = f"channel_{channel_id}"
    details_file = os.path.join(DETAILS_DIR, f"{channel_name}_details.csv")

    # Daca fisierul exista, il returnam
    if os.path.exists(details_file):
        return send_file(details_file, mimetype="text/csv")

    # Daca NU exista, cautam fisierul CSV de baza
    data_file = os.path.join(DOWNSAMPLED_DIR, f"{channel_name}_downsampled_1H.csv")
    if not os.path.exists(data_file):
        return jsonify({"error": f"Fisierul de date pentru {channel_name} nu exista."}), 404

    try:
        # Cream temporar un folder cu doar acel fisier ca sa rulam functia ta
        temp_input_dir = os.path.join(BASE_DIR, 'temp_input')
        os.makedirs(temp_input_dir, exist_ok=True)
        temp_copy = os.path.join(temp_input_dir, os.path.basename(data_file))
        pd.read_csv(data_file).to_csv(temp_copy, index=False)

        # Rulam functia pe folderul temporar
        metrics_channels(temp_input_dir, DETAILS_DIR)

        # Stergem copia
        os.remove(temp_copy)

        if os.path.exists(details_file):
            return send_file(details_file, mimetype="text/csv")
        else:
            return jsonify({"error": "Fisierul nu a putut fi generat."}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/histogram/<int:channel_id>", methods=["GET"])
def get_histogram(channel_id):
    channel_name = f"channel_{channel_id}"
    histogram_path = os.path.join(HISTOGRAM_DIR, f"{channel_name}_histogram.png")

    #Daca deja exista
    if os.path.exists(histogram_path):
        return send_file(histogram_path, mimetype="image/png")

    #Daca nu exista, o generam
    try:
        generated_path = generate_histogram(
            channel_id=channel_id,
            csv_dir=DOWNSAMPLED_DIR,
            output_dir=HISTOGRAM_DIR,
            labels_file=LABELS_FILE
        )
        return send_file(generated_path, mimetype="image/png")

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/correlogram", methods=["GET"])
def get_correlogram():
    output_file = os.path.join(HISTOGRAM_DIR, "correlogram.png")
    if os.path.exists(output_file):
        return send_file(output_file, mimetype="image/png")

    try:
        result_path = generate_correlogram(
            csv_dir=DOWNSAMPLED_DIR,
            output_path=output_file,
            labels_file=os.path.join(BASE_DIR, "labels.dat")
        )
        return send_file(result_path, mimetype="image/png")

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/consumption/<int:channel_id>/<date_str>", methods=["GET"])
def get_day_consumption(channel_id, date_str):
    try:
        total = get_consumption_for_day(channel_id, DOWNSAMPLED_DIR, date_str)
        return jsonify({
            "channel": f"channel_{channel_id}",
            "date": date_str,
            "total_consumption": total
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
