from DataAnalysis import DataAnalyzer
from Metrics import MetricsAnalyzer
import os

if __name__ == "__main__":
    # Define paths and channels
    house3_dir = r'C:\Users\elecf\Desktop\Licenta\Date\UK-DALE-disaggregated\house_3'
    labels_file = os.path.join(house3_dir, 'labels.dat')
    channels = ['channel_1.dat', 'channel_2.dat', 'channel_3.dat', 'channel_4.dat', 'channel_5.dat']

    # Initialize the DataAnalyzer
    analyzer = DataAnalyzer(house_dir=house3_dir, labels_file=labels_file, channels=channels)

    # Load labels and data
    analyzer.load_labels()
    analyzer.load_data()

    # Plot time series for each channel
    analyzer.plot_time_series()

    # Calculate metrics for the data
    analyzer.calculate_metrics()

    # Display metrics and save them to a CSV file
    analyzer.display_metrics()

    # Initialize MetricsAnalyzer with loaded data
    metrics_analyzer = MetricsAnalyzer(data_dict=analyzer.data_dict, labels=analyzer.labels)

    # Calculate and display additional metrics
    metrics_analyzer.display_metrics()
