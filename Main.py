from DataAnalysis import DataAnalyzer
import os

if __name__ == "__main__":
    # Incarcare si vizualizare date
    house3_dir = r'C:\Users\elecf\Desktop\Licenta\Date\UK-DALE-disaggregated\house_3'
    labels_file = os.path.join(house3_dir, 'labels.dat')
    channels = ['channel_1.dat', 'channel_2.dat', 'channel_3.dat', 'channel_4.dat', 'channel_5.dat']

    analyzer = DataAnalyzer(house_dir=house3_dir, labels_file=labels_file, channels=channels)

    analyzer.load_labels()
    analyzer.load_data()

    analyzer.plot_time_series()

    analyzer.calculate_metrics()

    analyzer.display_metrics()