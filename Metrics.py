import pandas as pd
from scipy.stats import pearsonr
from tabulate import tabulate


class MetricsAnalyzer:
    def __init__(self, data_dict, labels):
        self.data_dict = data_dict
        self.labels = labels
        self.metrics_df = {}

    def calculate_daily_average(self):
        """Calculăm media zilnică a consumului energetic pentru fiecare canal"""
        daily_averages = []
        for channel, data in self.data_dict.items():
            if data is not None:
                daily_data = data.resample('D').sum()
                average_daily = daily_data['power'].mean()
                daily_averages.append({
                    'channel': self.labels.get(channel, 'Unknown'),
                    'average_daily_power': average_daily
                })
        return pd.DataFrame(daily_averages)

    def identify_peaks(self):
        """Identificăm peak-ul consumului pentru fiecare canal"""
        peaks = []
        for channel, data in self.data_dict.items():
            if data is not None:
                max_power = data['power'].max()
                timestamp = data['power'].idxmax()
                peaks.append({
                    'channel': self.labels.get(channel, 'Unknown'),
                    'peak_power': max_power,
                    'timestamp': timestamp
                })
        return pd.DataFrame(peaks)

    def calculate_correlation(self):
        """Calculăm corelația energetică între consumul total și canalele individuale"""
        if 'channel_1.dat' not in self.data_dict or self.data_dict['channel_1.dat'] is None:
            print("⚠️ Channel 1 data is necesară pentru calculul corelației.")
            return None

        total_power = self.data_dict['channel_1.dat']['power']
        correlations = []

        for channel, data in self.data_dict.items():
            if channel != 'channel_1.dat' and data is not None:
                common_index = total_power.index.intersection(data.index)
                if len(common_index) > 0:
                    corr, _ = pearsonr(total_power.loc[common_index], data['power'].loc[common_index])
                    correlations.append({
                        'channel': self.labels.get(channel, 'Unknown'),
                        'correlation_with_total': corr
                    })
        return pd.DataFrame(correlations)

    def display_metrics(self):
        """Combină și afișează toate metricile"""
        daily_averages = self.calculate_daily_average()
        peaks = self.identify_peaks()
        correlations = self.calculate_correlation()

        self.metrics_df = {
            'Daily Averages': daily_averages,
            'Peaks': peaks,
            'Correlations': correlations
        }

        for key, df in self.metrics_df.items():
            print(f"\n📊 {key}:")
            print(tabulate(df, headers='keys', tablefmt='grid'))

    def save_metrics(self, output_path):
        """Salvează metricile într-un fișier CSV"""
        if not self.metrics_df:
            print("⚠️ Nu există metrici calculate! Asigură-te că ai rulat `display_metrics()` înainte de a salva.")
            return

        # Combinăm toate metricile într-un singur DataFrame
        with open(output_path, 'w') as f:
            for key, df in self.metrics_df.items():
                f.write(f"\n### {key} ###\n")
                df.to_csv(f, index=False)

        print(f"✅ Metricile generale au fost salvate în: {output_path}")
