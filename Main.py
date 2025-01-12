from DataAnalysis import DataAnalyzer
from Metrics import MetricsAnalyzer
from PlotAnalysis import PlotAnalyzer
from AggregationAnalysis import AggregationAnalyzer
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
    metrics = MetricsAnalyzer(data_dict=analyzer.data_dict, labels=analyzer.labels)

    # Calculate and display additional metrics
    metrics.display_metrics()

    # Initialize PlotAnalyzer with loaded data
    plot_analyzer = PlotAnalyzer(data_dict=analyzer.data_dict, labels=analyzer.labels)

    # Plot recommended visualizations
    plot_analyzer.plot_histograms()
    plot_analyzer.plot_correlograms()

    # Initialize AggregationAnalyzer cu datele încărcate
    aggregation_analyzer = AggregationAnalyzer(data_dict=analyzer.data_dict, labels=analyzer.labels)

    # Agregare zilnică (sumarizare zilnică)
    aggregation_analyzer.display_aggregated_data(freq='D')

    # Agregare săptămânală
    aggregation_analyzer.display_aggregated_data(freq='W')

    # Reducere granularitate la 1 minut
    aggregation_analyzer.display_downsampled_data(freq='1T')

    # Salvare date agregate și cu granularitate redusă
    output_dir = r'C:\Users\elecf\Desktop\Licenta\Date\UK-DALE-disaggregated\house_3'
    aggregation_analyzer.save_aggregated_data(freq='D', output_dir=output_dir)
    aggregation_analyzer.save_downsampled_data(freq='1T', output_dir=output_dir)
