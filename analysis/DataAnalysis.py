import os
import pandas as pd
import matplotlib.pyplot as plt
import re

class DataAnalyzer:
    def __init__(self, house_dir, labels_file, channels):
        self.house_dir = house_dir
        self.labels_file = labels_file
        self.channels = channels
        self.labels = {}
        self.data_dict = {}
        self.metrics_df = None

    def load_labels(self):
        try:
            with open(self.labels_file, 'r') as f:
                channels_dict = {}

                for line in f:
                    if line.strip():
                        channel, label = line.strip().split(' ', 1)
                        channel_num = int(re.search(r'\d+', channel).group())
                        channels_dict[channel_num] = (channel, label)

                sorted_labels = dict(sorted(channels_dict.items()))

                for channel_num, (channel, label) in sorted_labels.items():
                    self.labels[f"channel_{channel}.dat"] = label

            print("Labels loaded successfully.")
        except Exception as e:
            print(f"Error loading labels: {e}")

    def load_channel_data(self, file_path):
        try:
            data = pd.read_csv(file_path, delimiter=' ', names=['timestamp', 'power'], header=None)
            data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')
            data.set_index('timestamp', inplace=True)
            return data
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None

    def load_data(self):
        self.channels.sort(key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else float('inf'))
        for channel in self.channels:
            file_path = os.path.join(self.house_dir, channel)
            if os.path.exists(file_path):
                print(f"Loading {channel}...")
                self.data_dict[channel] = self.load_channel_data(file_path)
            else:
                print(f"File not found: {file_path}")

    def plot_time_series(self):
        for channel, data in self.data_dict.items():
            if data is not None:
                plt.figure(figsize=(12, 6))
                plt.plot(data.index, data['power'], label=f"{self.labels.get(channel, 'Unknown')} (Power)")
                plt.xlabel("Time")
                plt.ylabel("Power (W)")
                plt.title(f"Time Series of {self.labels.get(channel, 'Unknown')}")
                plt.legend()
                plt.show()

    def plot_downsampled_csvs(self, csv_dir):
        output_dir = os.path.join(self.house_dir, "analysis", "plots")
        os.makedirs(output_dir, exist_ok=True)

        simple_labels = {}
        with open(self.labels_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    channel_id = int(parts[0])
                    label = ' '.join(parts[1:])
                    simple_labels[channel_id] = label

        for file in os.listdir(csv_dir):
            if file.endswith("_downsampled_1H.csv"):
                match = re.search(r'channel_(\d+)', file)
                if not match:
                    continue
                channel_id = int(match.group(1))
                label = simple_labels.get(channel_id, f"Channel {channel_id}")

                try:
                    file_path = os.path.join(csv_dir, file)
                    df = pd.read_csv(file_path, parse_dates=['timestamp'])

                    plt.figure(figsize=(12, 5))
                    plt.plot(df['timestamp'], df['power'], label=label)
                    plt.xlabel("Timp")
                    plt.ylabel("Consum (W)")
                    plt.title(f"{label} - Consum energetic (1H)")
                    plt.grid(True)
                    plt.tight_layout()

                    save_path = os.path.join(output_dir, f"{file.replace('.csv', '.png')}")
                    plt.savefig(save_path)
                    plt.close()
                    print(f"plot salvat: {save_path}")
                except Exception as e:
                    print(f"eroare la {file}: {e}")

