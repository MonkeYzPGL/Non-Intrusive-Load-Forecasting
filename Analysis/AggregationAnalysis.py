import os

import pandas as pd


class AggregationAnalyzer:
    def __init__(self, data_dict, labels):

        self.data_dict = data_dict
        self.labels = labels

    def aggregate_data(self, freq):

        aggregated_data = {}
        for channel, data in self.data_dict.items():
            if data is not None:
                aggregated_data[channel] = data.resample(freq).sum()
        return aggregated_data

    def downsample_data(self, freq):
        downsampled_data = {}
        for channel, data in self.data_dict.items():
            if data is not None:
                downsampled_data[channel] = data.resample(freq).mean().interpolate(method='linear')
        return downsampled_data

    def save_aggregated_data(self, freq, output_dir):

        aggregated_data = self.aggregate_data(freq)
        for channel, data in aggregated_data.items():
            output_path = os.path.join(output_dir, f"{channel.replace('.dat', '')}_aggregated_{freq}.csv")
            data.to_csv(output_path)
            print(f"Aggregated data saved to {output_path}")

    def save_downsampled_data(self, freq, output_dir):

        downsampled_data = self.downsample_data(freq)
        for channel, data in downsampled_data.items():
            data.fillna(0, inplace=True)
            output_path = os.path.join(output_dir, f"{channel.replace('.dat', '')}_downsampled_{freq}.csv")
            data.to_csv(output_path)
            print(f"Downsampled data saved to {output_path}")

    def generate_aggregated(self, input_folder):

        channel_dfs = []

        for filename in sorted(os.listdir(input_folder)):
            if filename.startswith("channel_") and filename.endswith("_downsampled_1H.csv"):
                if filename == "channel_1_downsampled_1H.csv":
                    continue #ignoram channel 1 daca exista deja
                full_path = os.path.join(input_folder, filename)
                df = pd.read_csv(full_path, index_col=0, parse_dates=True)
                df = df.rename(columns={df.columns[0]: filename})
                channel_dfs.append(df)

        if not channel_dfs:
            raise ValueError("Nu s-au gasit fisiere valide in folder")

        combined_df = pd.concat(channel_dfs, axis=1).fillna(0)
        aggregated = combined_df.sum(axis=1).to_frame(name="power")
        aggregated.index.name = "timestamp"

        output_path = os.path.join(input_folder, "channel_1_downsampled_1H.csv")
        aggregated.to_csv(output_path)
        print(f"Fisierul channel 1 nou salvat la: {output_path}")

        return output_path

    def calculate_delta(directory):
        files = [f for f in os.listdir(directory) if f.endswith("1H.csv")]

        # Citirea datelor
        data = {}
        for file in files:
            channel_name = file.replace("_downsampled_1H.csv", "")
            filepath = os.path.join(directory, file)
            df = pd.read_csv(filepath, parse_dates=['timestamp'], index_col='timestamp')
            data[channel_name] = df

        # Verificare daca avem channel_1
        if "channel_1" not in data:
            raise ValueError("Lipseste fisierul pentru channel_1.")

        # Calculul diferentei delta
        other_channels = [ch for ch in data if ch != "channel_1"]
        data["delta"] = data["channel_1"] - sum(data[ch] for ch in other_channels)

        # Salvare rezultat intr-un fisier CSV
        output_file = os.path.join(directory, "delta_values_1T.csv")
        data["delta"].to_csv(output_file)

        print(f"Fisierul cu valorile delta a fost salvat: {output_file}")