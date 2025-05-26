import os

import pandas as pd


class AggregationAnalyzer:
    def __init__(self, data_dict, labels):
        """
        Initializeaza analiza pentru agregare si granularitate.
        :param data_dict: Datele incarcate pentru fiecare canal.
        :param labels: Etichetele pentru fiecare canal.
        """
        self.data_dict = data_dict
        self.labels = labels

    def aggregate_data(self, freq):
        """
        Agrega datele la o anumita frecventa.
        :param freq: Frecventa pentru agregare (e.g., 'D' pentru zi, 'W' pentru saptamana).
        :return: Datele agregate pentru fiecare canal.
        """
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
        """
        Salveaza datele agregate intr-un director.
        :param freq: Frecventa pentru agregare.
        :param output_dir: Directorul unde sa se salveze datele.
        """
        aggregated_data = self.aggregate_data(freq)
        for channel, data in aggregated_data.items():
            output_path = os.path.join(output_dir, f"{channel.replace('.dat', '')}_aggregated_{freq}.csv")
            data.to_csv(output_path)
            print(f"Aggregated data saved to {output_path}")

    def save_downsampled_data(self, freq, output_dir):
        """
        Salveaza datele cu granularitate redusa intr-un director.
        Inlocuieste valorile lipsa (NaN) cu 0 inainte de salvare.
        :param freq: Frecventa dorita pentru reducerea granularitatii.
        :param output_dir: Directorul unde sa se salveze datele.
        """
        downsampled_data = self.downsample_data(freq)
        for channel, data in downsampled_data.items():
            data.fillna(0, inplace=True)
            output_path = os.path.join(output_dir, f"{channel.replace('.dat', '')}_downsampled_{freq}.csv")
            data.to_csv(output_path)
            print(f"Downsampled data saved to {output_path}")

    def generate_aggregated(self, input_folder):
        """
        Creeaza un fisier channel_1_downsampled_1H.csv rezultat prin suma tuturor aparatelor
        ignorand channel1
        :input_folder: Directorul unde sa se salveze csv-ul si sa fie preluat csv-urile aparatelor
        """
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