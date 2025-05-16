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

    def downsample_data(self, freq='1H'):
        """
        Downsampleaza datele la o anumita frecventa, cu tratarea NaN-urilor.
        :param freq: Frecventa dorita pentru reducerea granularitatii.
        :return: Datele downsampled pentru fiecare canal.
        """
        downsampled_data = {}
        for channel, data in self.data_dict.items():
            if data is not None:
                # Downsamplează cu media orară
                downsampled = data.resample(freq).mean()

                # Tratare NaN cu interpolare și completare inteligentă
                downsampled = downsampled.interpolate(method='linear', limit_direction='both', limit=5)

                # Folosim forward-fill și backward-fill pentru completarea finală
                downsampled.ffill(inplace=True)
                downsampled.bfill(inplace=True)

                downsampled_data[channel] = downsampled

        return downsampled_data

    def check_consistency(self, downsampled_data):
        """
        Verifica daca suma consumului aparatelor nu depaseste consumul total.
        """
        total_channel = downsampled_data.get("channel_1", None)
        if total_channel is not None:
            individual_sum = sum([data for channel, data in downsampled_data.items() if channel != "channel_1"])

            # Compara suma canalelor individuale cu consumul total
            if individual_sum.sum().sum() > total_channel.sum().sum():
                print("⚠️ Inconsistency detected: Individual channels exceed total consumption.")
            else:
                print("✅ Data is consistent.")


    def check_data_consistency(self, dir=None, output_file=None):
        """
        Verifica daca suma consumului aparatelor nu depaseste consumul total.
        Salveaza rezultatele intr-un fisier CSV.
        """
        if dir is None:
            print("⚠️ Downsampled directory not provided.")
            return

        # Incarca canalul 1 (consum total)
        total_file = os.path.join(dir, "channel_1_downsampled_1h.csv")
        total_data = pd.read_csv(total_file, index_col=0, parse_dates=True)

        # Calculam suma tuturor canalelor individuale
        individual_sum = None
        inconsistent_hours = []

        for file_name in os.listdir(dir):
            if file_name.startswith("channel_") and file_name != "channel_1_downsampled_1h.csv":
                file_path = os.path.join(dir, file_name)
                data = pd.read_csv(file_path, index_col=0, parse_dates=True)

                if individual_sum is None:
                    individual_sum = data
                else:
                    individual_sum += data

        # Verificam daca suma canalelor individuale depaseste canalul total
        inconsistency = individual_sum > total_data
        if inconsistency.any().any():
            print("⚠️ Inconsistencies detected:")
            for timestamp in inconsistency.index:
                if inconsistency.loc[timestamp].any():
                    inconsistent_hours.append(timestamp)
                    print(
                        f"Inconsistency at {timestamp}: Total={total_data.loc[timestamp].values[0]}, Individual Sum={individual_sum.loc[timestamp].values[0]}")
        else:
            print("✅ All data is consistent.")

        # Salvam rezultatele
        if inconsistent_hours:
            result_df = individual_sum.loc[inconsistent_hours] - total_data.loc[inconsistent_hours]
            result_df.to_csv(output_file)
            print(f"⚠️ Inconsistent data saved to {output_file}")

    def save_downsampled_data(self, freq='1H', output_dir='./downsampled_data'):
        """
        Salveaza datele cu granularitate redusa intr-un director.
        Inlocuieste valorile lipsa (NaN) cu 0 inainte de salvare.
        :param freq: Frecventa dorita pentru reducerea granularitatii.
        :param output_dir: Directorul unde sa se salveze datele.
        """
        # Creează directorul dacă nu există
        os.makedirs(output_dir, exist_ok=True)

        # Downsamplează datele
        downsampled_data = self.downsample_data(freq)

        # Verifică consistența înainte de a salva
        self.check_consistency(downsampled_data)

        # Salvează fiecare canal
        for channel, data in downsampled_data.items():
            # Curățăm numele fișierelor pentru a elimina extensii nedorite
            channel_name = channel.replace(".dat", "").replace(" ", "_")

            # Salvăm datele într-un fișier CSV
            output_path = os.path.join(output_dir, f"{channel_name}_downsampled_{freq}.csv")
            data.to_csv(output_path)

            print(f"✅ Downsampled data for {channel} saved to {output_path}")

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

    def verify_channel_sums(self, data_directory = None):
        # Load channel 1 data
        channel_1_file = os.path.join(data_directory, 'channel_1_downsampled_1H.csv')
        channel_1_df = pd.read_csv(channel_1_file, parse_dates=['timestamp'], index_col='timestamp')

        # Initialize sum dataframe for channels 2 to 53
        sum_channels_df = pd.DataFrame(index=channel_1_df.index)
        sum_channels_df['sum_power'] = 0.0

        # Add power values from channels 2 to 53
        for channel in range(2, 54):
            channel_file = os.path.join(data_directory, f'channel_{channel}_downsampled_1H.csv')
            if os.path.exists(channel_file):
                channel_df = pd.read_csv(channel_file, parse_dates=['timestamp'], index_col='timestamp')
                sum_channels_df['sum_power'] += channel_df['power'].reindex(sum_channels_df.index, fill_value=0)

        # Combine and compare with channel 1
        comparison_df = channel_1_df.copy()
        comparison_df['sum_channels_2_to_53'] = sum_channels_df['sum_power']
        comparison_df['difference'] = comparison_df['power'] - comparison_df['sum_channels_2_to_53']

        # Check if all values match
        all_match = comparison_df['difference'].abs().sum() == 0

        # Save the detailed comparison as CSV
        output_file = os.path.join(data_directory, 'channel_comparison_report.csv')
        comparison_df.to_csv(output_file)
        print(f"Comparison report saved to: {output_file}")

        # Return the result
        return all_match, comparison_df