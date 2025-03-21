import os

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
