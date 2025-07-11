import os

import numpy as np
import pandas as pd

def get_common_period(folder_path, min_months=9):
    csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith("_downsampled_1H.csv")]

    periods = []
    for f in csv_files:
        df = pd.read_csv(f, parse_dates=['timestamp'])
        min_time = df['timestamp'].min()
        max_time = df['timestamp'].max()
        duration_months = (max_time - min_time).days / 30
        if duration_months >= min_months:
            periods.append((os.path.basename(f), min_time, max_time))

    #calculeaza perioada comuna
    start = max(p[1] for p in periods)
    end = min(p[2] for p in periods)
    return start, end

def load_and_filter_channels(folder_path, start, end):
    data_dict = {}

    for file in os.listdir(folder_path):
        if file.endswith("_downsampled_1H.csv"):
            channel_name = file.split("_downsampled")[0]
            path = os.path.join(folder_path, file)
            df = pd.read_csv(path, parse_dates=['timestamp'])
            df = df[(df['timestamp'] >= start) & (df['timestamp'] <= end)]
            df = df.set_index('timestamp')
            df = df.rename(columns={"power": channel_name})
            data_dict[channel_name] = df

    combined_df = pd.concat(data_dict.values(), axis=1).fillna(0)
    return combined_df

def create_nilm_sequences(csv_path, window_size=168):
    df = pd.read_csv(csv_path, parse_dates=['timestamp'])
    df = df.set_index('timestamp')

    appliance_cols = [col for col in df.columns if col != "channel_1"]
    total_col = "channel_1"

    X_list, Y_list = [], []

    for i in range(len(df) - window_size):
        x_window = df.iloc[i:i+window_size][total_col].values
        y_window = df.iloc[i:i+window_size][appliance_cols].values

        X_list.append(x_window.reshape(-1, 1))
        Y_list.append(y_window)

    X = np.array(X_list)
    Y = np.array(Y_list)

    return X, Y, appliance_cols


if __name__ == "__main__":
    BASE_DIR = r'C:\Users\elecf\Desktop\Licenta\Date\UK-DALE-disaggregated\house_1'
    DOWNSAMPLED_DIR = os.path.join(BASE_DIR, 'downsampled', '1H')
    nilm_dir = os.path.join(DOWNSAMPLED_DIR, "NILM")
    os.makedirs(nilm_dir, exist_ok=True)

    common_start, common_end = get_common_period(DOWNSAMPLED_DIR)
    print(f"Perioada comuna: {common_start} - {common_end}")

    #incarcam si filtram datele
    df_combined = load_and_filter_channels(DOWNSAMPLED_DIR, common_start, common_end)

    #salvam csv-ul cu datele sincronizate in folderul NILM
    csv_path = os.path.join(nilm_dir, "nilm_dataset.csv")
    df_combined.to_csv(csv_path)
    print(f"dataset sincronizat salvat la: {csv_path}")

    #cream secventele X/Y
    X, Y, appliance_cols = create_nilm_sequences(csv_path, window_size=168)

    np.save(os.path.join(nilm_dir, "X_total.npy"), X)
    np.save(os.path.join(nilm_dir, "Y_appliances.npy"), Y)

    print(f"shape X: {X.shape} | shape Y: {Y.shape}")
    print(f"canale prezente (Y): {appliance_cols}")

