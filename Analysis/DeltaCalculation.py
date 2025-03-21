import os
import pandas as pd


def calculate_delta(directory):
    # Identificare fisiere cu terminatia "1T.csv" in director
    files = [f for f in os.listdir(directory) if f.endswith("1T.csv")]

    # Citirea datelor
    data = {}
    for file in files:
        channel_name = file.replace("_downsampled_1T.csv", "")
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

# Exemplu de utilizare
# calculate_delta("/house3/data/")
