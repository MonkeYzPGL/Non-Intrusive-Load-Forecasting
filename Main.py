import pandas as pd
from DataAnalysis import DataAnalyzer
from Metrics import MetricsAnalyzer
from PlotAnalysis import PlotAnalyzer
from AggregationAnalysis import AggregationAnalyzer
from LSTMAnalysis import LSTMAnalyzer
import os
import torch

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

    # Initialize AggregationAnalyzer cu datele încărcate
    aggregation_analyzer = AggregationAnalyzer(data_dict=analyzer.data_dict, labels=analyzer.labels)

    # Reducere granularitate la 1 minut
    downsampled_data = aggregation_analyzer.downsample_data(freq='1T')
    channel_1_downsampled = downsampled_data.get('channel_1.dat')

    # Verificăm dacă datele există
    if channel_1_downsampled is not None:
        # Pregătire pentru LSTM
        channel_1_data = channel_1_downsampled['power'].dropna().values

        # Inițializare și rulare LSTM Analyzer
        lstm_analyzer = LSTMAnalyzer(data=channel_1_data, seq_length=10, epochs=20, batch_size=64)
        lstm_analyzer.train()

        # Plotarea pierderii
        lstm_analyzer.plot_loss()

        # Predicții pentru următoarele 24 de minute
        next_minute_predictions = lstm_analyzer.predict_next_day()
        print("Predicted values for the next 24 minutes:")

        # Crearea unui tabel
        minutes = [f"Minute {i + 1}" for i in range(24)]
        prediction_table = pd.DataFrame({
            "Minute": minutes,
            "Predicted Power (W)": next_minute_predictions
        })

        # Salvarea tabelului într-un fișier CSV
        output_path = os.path.join(house3_dir, "next_minute_predictions.csv")
        prediction_table.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")
    else:
        print("Downsampled data for channel_1 is unavailable.")
