import pandas as pd
from scipy.stats import pearsonr
from tabulate import tabulate

class MetricsAnalyzer:
    def __init__(self, data_dict, labels):
        self.data_dict = data_dict
        self.labels = labels
        self.metrics_df = None

    def calculate_daily_average(self):
        """Calculam average-ul consumului energetic pentru fiecare canal"""
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
        """Identificam peak-ul consumului pentru fiecare canal"""
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
        """Calculam corelatia energetica intre consumul total si canalele individuale"""
        if 'channel_1.dat' not in self.data_dict or self.data_dict['channel_1.dat'] is None:
            print("Channel 1 data is required for correlation calculation.")
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
        """Combina si da display la toate metricile"""
        daily_averages = self.calculate_daily_average()
        peaks = self.identify_peaks()
        correlations = self.calculate_correlation()

        self.metrics_df = {
            'Daily Averages': daily_averages,
            'Peaks': peaks,
            'Correlations': correlations
        }

        for key, df in self.metrics_df.items():
            print(f"\n{key}:")
            print(tabulate(df, headers='keys', tablefmt='grid'))
