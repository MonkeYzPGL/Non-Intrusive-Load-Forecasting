import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
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

    labels = load_labels(labels_file)
    label_name = labels.get(channel_id, channel_name)

    df = pd.read_csv(csv_path)

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

    print(f"decomposition plot salvat: {output_path}")
    return output_path

def generate_correlogram(csv_dir, output_path, labels_file):
    labels = load_labels(labels_file)
    combined_data = pd.DataFrame()

    for file in os.listdir(csv_dir):
        if file.endswith("_downsampled_1H.csv"):
            try:
                channel_id = int(file.split("_")[1])
                label = labels.get(channel_id, f"channel_{channel_id}")
                csv_path = os.path.join(csv_dir, file)
                df = pd.read_csv(csv_path, parse_dates=['timestamp'])

                combined_data[label] = df['power'].fillna(0).reset_index(drop=True)

            except Exception as e:
                print(f"Eroare la {file}: {str(e)}")

    if combined_data.empty:
        raise ValueError("nu s-au incarcat date valide.")

    plt.figure(figsize=(14, 12))
    sns.heatmap(combined_data.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    plt.title("Correlogram of Appliance Power Consumption")
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

    print(f"corelograma salvata in: {output_path}")
    return output_path

def generate_acf_plot(channel_id, csv_dir, output_dir, label_path, lags=168):
    csv_file = os.path.join(csv_dir, f"channel_{channel_id}_downsampled_1H.csv")
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"fisierul {csv_file} nu exista.")

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

def metrics_channels(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        if file.endswith("_downsampled_1H.csv"):
            try:
                path = os.path.join(input_dir, file)
                df = pd.read_csv(path, parse_dates=['timestamp'])

                if 'power' not in df.columns or df.empty:
                    print(f" Fisier invalid: {file}")
                    continue

                df['hour'] = df['timestamp'].dt.hour
                df['weekday'] = df['timestamp'].dt.day_name()

                stats = {
                    'min': df['power'].min(),
                    'max': df['power'].max(),
                    'mean': df['power'].mean(),
                    'sum': df['power'].sum(),
                    'std': df['power'].std(),
                    'median': df['power'].median(),
                    'nr_ore_active': (df['power'] > 0).sum(),
                    'procent_activitate': (df['power'] > 0).mean() * 100,
                    'ora_cu_consum_maxim': df.groupby('hour')['power'].mean().idxmax(),
                    'zi_cu_consum_maxim': df.groupby('weekday')['power'].mean().idxmax(),
                }

                channel_name = file.split("_downsampled")[0]
                output_file = os.path.join(output_dir, f"{channel_name}_details.csv")
                pd.DataFrame([stats]).to_csv(output_file, index=False)

                print(f"detalii salvate: {output_file}")

            except Exception as e:
                print(f"eroare la {file}: {str(e)}")

def get_consumption_for_day(channel_id, csv_dir, date_str, labels_file):
    import pandas as pd
    from datetime import datetime

    channel_name = f"channel_{channel_id}"
    csv_path = os.path.join(csv_dir, f"{channel_name}_downsampled_1H.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV lipsa: {csv_path}")

    df = pd.read_csv(csv_path, parse_dates=['timestamp'])
    df['date'] = df['timestamp'].dt.date

    target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    filtered = df[df['date'] == target_date]
    total = filtered['power'].sum()

    labels = load_labels(labels_file)
    appliance_name = labels.get(channel_id, channel_name)

    return {
        "channel": channel_name,
        "appliance_name": appliance_name,
        "date": date_str,
        "total_consumption": round(total, 2)
    }