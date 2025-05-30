import os
import pandas as pd

def metrics_channels(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        if file.endswith("_downsampled_1H.csv"):
            try:
                path = os.path.join(input_dir, file)
                df = pd.read_csv(path, parse_dates=['timestamp'])

                if 'power' not in df.columns or df.empty:
                    print(f" Fisier invalid: {file}")
                    continue

                df['hour'] = df['timestamp'].dt.hour
                df['weekday'] = df['timestamp'].dt.day_name()

                stats = {
                    'min': df['power'].min(),
                    'max': df['power'].max(),
                    'mean': df['power'].mean(),
                    'sum': df['power'].sum(),
                    'std': df['power'].std(),
                    'median': df['power'].median(),
                    'nr_ore_active': (df['power'] > 0).sum(),
                    'procent_activitate': (df['power'] > 0).mean() * 100,
                    'ora_cu_consum_maxim': df.groupby('hour')['power'].mean().idxmax(),
                    'zi_cu_consum_maxim': df.groupby('weekday')['power'].mean().idxmax(),
                }

                channel_name = file.split("_downsampled")[0]
                output_file = os.path.join(output_dir, f"{channel_name}_details.csv")
                pd.DataFrame([stats]).to_csv(output_file, index=False)

                print(f"✅ Detalii salvate: {output_file}")

            except Exception as e:
                print(f" Eroare la {file}: {str(e)}")

def get_consumption_for_day(channel_id, csv_dir, date_str):
    import pandas as pd
    from datetime import datetime

    channel_name = f"channel_{channel_id}"
    csv_path = os.path.join(csv_dir, f"{channel_name}_downsampled_1H.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV lipsă: {csv_path}")

    df = pd.read_csv(csv_path, parse_dates=['timestamp'])
    df['date'] = df['timestamp'].dt.date

    target_date = datetime.strptime(date_str, "%Y-%m-%d").date()

    filtered = df[df['date'] == target_date]
    total = filtered['power'].sum()

    return round(total, 2)
