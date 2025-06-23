import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose


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

def decomposition_plot(input_path, output_dir, channel_name):

    df = pd.read_csv(input_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    decomp = seasonal_decompose(df['power'], model='additive', period=24*7)

    fig, axes = plt.subplots(4, 1, figsize=(15, 10), sharex=True)
    decomp.observed.plot(ax=axes[0], title='Observed')
    decomp.trend.plot(ax=axes[1], title='Trend')
    decomp.seasonal.plot(ax=axes[2], title='Seasonality (weekly)')
    decomp.resid.plot(ax=axes[3], title='Residuals')
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{channel_name}_decomposition.png")
    fig.savefig(output_path)
    plt.close(fig)

    print(f" Decomposition plot salvat: {output_path}")
    return output_path


