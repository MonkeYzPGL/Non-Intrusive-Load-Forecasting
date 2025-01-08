'''
Analiza datelor (Data Analysis)
Citirea datelor din fisiere pentru casa 3 din fisierele .dat.
Vizualizarea datelor sub forma de grafice pentru fiecare din canale.
Analizarea distributiei pentru fiecare aparat.
'''

import os
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

# Functie pentru citirea fisierelor .dat
def load_channel_data(file_path):
    try:
        data = pd.read_csv(file_path, delimiter=' ', names=['timestamp', 'power'], header=None)
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')
        data.set_index('timestamp', inplace=True)
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# Functie pentru citirea etichetelor din labels.dat
def load_labels(file_path):
    try:
        labels = {}
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    channel, label = line.strip().split(' ', 1)
                    labels[f"channel_{channel}.dat"] = label
        return labels
    except Exception as e:
        print(f"Error loading labels: {e}")
        return {}

# Directorul casei 3 si fisierele relevante
house3_dir = r'C:\Users\elecf\Desktop\Licenta\Date\UK-DALE-disaggregated\house_3'
labels_file = os.path.join(house3_dir, 'labels.dat')
channels = ['channel_1.dat', 'channel_2.dat', 'channel_3.dat', 'channel_4.dat', 'channel_5.dat']

# Citirea etichetelor
labels = load_labels(labels_file)

# Incarcare date pentru fiecare channel
data_dict = {}
for channel in channels:
    file_path = os.path.join(house3_dir, channel)
    if os.path.exists(file_path):
        print(f"Loading {channel}...")
        data_dict[channel] = load_channel_data(file_path)
    else:
        print(f"File not found: {file_path}")

# Vizualizare serii temporale pentru fiecare canal
for channel, data in data_dict.items():
    if data is not None:
        plt.figure(figsize=(12, 6))
        plt.plot(data.index, data['power'], label=f"{labels.get(channel, 'Unknown')} (Power)")
        plt.xlabel("Time")
        plt.ylabel("Power (W)")
        plt.title(f"Time Series of {labels.get(channel, 'Unknown')}")
        plt.legend()
        plt.show()

# Calcularea si afisarea metricilor de baza pentru fiecare canal
metrics = []
for channel, data in data_dict.items():
    if data is not None:
        total_power = data['power'].sum()
        mean_power = data['power'].mean()
        max_power = data['power'].max()
        min_power = data['power'].min()
        duration = data.index.max() - data.index.min()
        metrics.append({
            'channel': labels.get(channel, 'Unknown'),
            'total_power': total_power,
            'mean_power': mean_power,
            'max_power': max_power,
            'min_power': min_power,
            'duration': duration
        })

# Transformam metricile intr-un DataFrame
metrics_df = pd.DataFrame(metrics)

print("Metrics for Each Channel:")
print(tabulate(metrics_df, headers='keys', tablefmt='grid'))

output_file = os.path.join(house3_dir, 'metrics.csv')
metrics_df.to_csv(output_file, index=False)
print(f"Metrics saved to {output_file}")
