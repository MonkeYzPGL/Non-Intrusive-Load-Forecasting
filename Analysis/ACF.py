import os
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

def generate_acf_plot(channel_id, csv_dir, output_dir, label_path, lags=168):
    csv_file = os.path.join(csv_dir, f"channel_{channel_id}_downsampled_1H.csv")
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Fisierul {csv_file} nu exista.")

    labels = {}
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                cid, name = parts
                labels[cid] = name

    df = pd.read_csv(csv_file, parse_dates=['timestamp'])
    df = df.set_index('timestamp')

    series = df['power'].dropna()

    plt.figure(figsize=(10, 6))
    plot_acf(series, lags=lags, alpha=0.05)
    plt.title(f"ACF - {labels.get(str(channel_id), f'Channel {channel_id}')}")
    plt.xlabel("Lag (hours)")
    plt.ylabel("Autocorrelation")
    plt.grid(True)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"channel_{channel_id}_acf.png")
    plt.savefig(output_path)
    plt.close()

    return output_path

