import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def load_labels(labels_file):
    labels = {}
    with open(labels_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                channel_id = int(parts[0])
                label = ' '.join(parts[1:])
                labels[channel_id] = label
    return labels


def generate_histogram(channel_id, csv_dir, output_dir, labels_file):
    channel_name = f"channel_{channel_id}"
    csv_path = os.path.join(csv_dir, f"{channel_name}_downsampled_1H.csv")
    output_path = os.path.join(output_dir, f"{channel_name}_histogram.png")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV-ul nu exista: {csv_path}")

    # Incarca numele aparatului
    labels = load_labels(labels_file)
    label_name = labels.get(channel_id, channel_name)

    df = pd.read_csv(csv_path)

    if 'power' not in df.columns:
        raise ValueError("Coloana 'power' lipseste din CSV.")

    plt.figure(figsize=(10, 6))
    plt.hist(df['power'].dropna(), bins=50, alpha=0.7, color='skyblue')
    plt.xlabel("Power (W)")
    plt.ylabel("Frequency")
    plt.title(f"Histogram - {label_name}")
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_path)
    plt.close()

    return output_path

def generate_correlogram(csv_dir, output_path, labels_file):
    labels = load_labels(labels_file)
    combined_data = pd.DataFrame()

    for file in os.listdir(csv_dir):
        if file.endswith("_downsampled_1H.csv"):
            try:
                channel_id = int(file.split("_")[1])  # ex: channel_39
                label = labels.get(channel_id, f"channel_{channel_id}")
                csv_path = os.path.join(csv_dir, file)
                df = pd.read_csv(csv_path, parse_dates=['timestamp'])

                # Includem doar coloana 'power' si redenumim
                combined_data[label] = df['power'].fillna(0).reset_index(drop=True)

            except Exception as e:
                print(f"❌ Eroare la {file}: {str(e)}")

    if combined_data.empty:
        raise ValueError("Nu s-au incarcat date valide.")

    plt.figure(figsize=(14, 12))
    sns.heatmap(combined_data.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    plt.title("Correlogram of Appliance Power Consumption")
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

    print(f"✅ Corelograma salvata in: {output_path}")
    return output_path
