from flask import Blueprint, send_file, jsonify
import os

from services.DetailsService import generate_histogram, generate_acf_plot, metrics_channels, get_consumption_for_day
import pandas as pd

details_bp = Blueprint("details", __name__)

# Setari directoare
BASE_DIR = r'C:\Users\elecf\Desktop\Licenta\Date\UK-DALE-disaggregated\house_1'
DOWNSAMPLED_DIR = os.path.join(BASE_DIR, 'downsampled', '1H')
DETAILS_DIR = os.path.join(BASE_DIR, 'details')
HISTOGRAM_DIR = os.path.join(BASE_DIR, "analysis", "histogram")
LABELS_FILE = os.path.join(BASE_DIR, "labels.dat")

@details_bp.route("/get_details/<int:channel_id>", methods=["GET"])
def get_channel_details(channel_id):
    channel_name = f"channel_{channel_id}"
    details_file = os.path.join(DETAILS_DIR, f"{channel_name}_details.csv")

    if os.path.exists(details_file):
        return send_file(details_file, mimetype="text/csv")

    data_file = os.path.join(DOWNSAMPLED_DIR, f"{channel_name}_downsampled_1H.csv")
    if not os.path.exists(data_file):
        return jsonify({"error": f"Fisierul de date pentru {channel_name} nu exista."}), 404

    try:
        temp_input_dir = os.path.join(BASE_DIR, 'temp_input')
        os.makedirs(temp_input_dir, exist_ok=True)
        temp_copy = os.path.join(temp_input_dir, os.path.basename(data_file))
        pd.read_csv(data_file).to_csv(temp_copy, index=False)

        metrics_channels(temp_input_dir, DETAILS_DIR)

        os.remove(temp_copy)

        if os.path.exists(details_file):
            return send_file(details_file, mimetype="text/csv")
        else:
            return jsonify({"error": "Fisierul nu a putut fi generat."}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@details_bp.route("/histogram/<int:channel_id>", methods=["GET"])
def get_histogram(channel_id):
    channel_name = f"channel_{channel_id}"
    histogram_path = os.path.join(HISTOGRAM_DIR, f"{channel_name}_histogram.png")

    if os.path.exists(histogram_path):
        return send_file(histogram_path, mimetype="image/png")

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

@details_bp.route("/consumption/<int:channel_id>/<date_str>", methods=["GET"])
def get_day_consumption(channel_id, date_str):
    try:
        result = get_consumption_for_day(
            channel_id=channel_id,
            csv_dir=DOWNSAMPLED_DIR,
            date_str=date_str,
            labels_file=os.path.join(BASE_DIR, "labels.dat")
        )
        return jsonify(result)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@details_bp.route("/csv/<int:channel_id>", methods=["GET"])
def get_csv_for_channel(channel_id):
    channel_name = f"channel_{channel_id}"
    csv_path = os.path.join(DOWNSAMPLED_DIR, f"{channel_name}_downsampled_1H.csv")

    if not os.path.exists(csv_path):
        return jsonify({"error": f"Fisierul CSV pentru {channel_name} nu exista."}), 404

    return send_file(csv_path, mimetype='text/csv')


@details_bp.route("/acf/<int:channel_id>", methods=["GET"])
def get_acf_plot(channel_id):
    try:
        csv_dir = os.path.join(BASE_DIR, "downsampled", "1H")
        acf_output_dir = os.path.join(BASE_DIR, "acf")
        label_path = os.path.join(BASE_DIR, "labels.dat")

        plot_path = generate_acf_plot(
            channel_id=channel_id,
            csv_dir=csv_dir,
            output_dir=acf_output_dir,
            label_path=label_path
        )

        return send_file(plot_path, mimetype="image/png")
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@details_bp.route("/labels", methods=["GET"])
def get_labels_json():
    if not os.path.exists(LABELS_FILE):
        return jsonify({"error": "Fisierul labels.dat nu exista."}), 404

    labels = {}
    with open(LABELS_FILE, 'r') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                channel_id, name = parts
                labels[int(channel_id)] = name

    return jsonify(labels)

@details_bp.route("/downsampled_json/<int:channel_id>", methods=["GET"])
def get_downsampled_json(channel_id):
    channel_name = f"channel_{channel_id}"
    csv_path = os.path.join(DOWNSAMPLED_DIR, f"{channel_name}_downsampled_1H.csv")

    if not os.path.exists(csv_path):
        return jsonify({"error": f"CSV-ul pentru {channel_name} nu exista."}), 404

    try:
        df = pd.read_csv(csv_path, parse_dates=['timestamp'])

        #extragem ultimele 10% date
        count_10_percent = max(1, int(len(df) * 0.1))
        recent_rows = df.tail(count_10_percent)

        return jsonify(recent_rows.to_dict(orient="records"))

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@details_bp.route("/plot/<int:channel_id>", methods=["GET"])
def get_plot_for_channel(channel_id):
    channel_name = f"channel_{channel_id}"
    plot_dir = os.path.join(BASE_DIR, "analysis", "plots")
    plot_path = os.path.join(plot_dir, f"{channel_name}_downsampled_1H.png")

    if os.path.exists(plot_path):
        return send_file(plot_path, mimetype="image/png")
    else:
        return jsonify({"error": f"Plot-ul pentru {channel_name} nu a fost gasit in {plot_dir}"}), 404