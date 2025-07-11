import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

def smape(actual, predicted):
    denominator = (np.abs(actual) + np.abs(predicted)) / 2
    mask = denominator != 0
    return np.mean(np.abs(actual[mask] - predicted[mask]) / denominator[mask]) * 100

def compute_metric_by_hour(df, metric_func, metric_name):
    df['hour'] = df['timestamp'].dt.hour
    result = df.groupby('hour').apply(lambda g: metric_func(g['actual'], g['prediction'])).reset_index(name=metric_name)
    return result

def compute_metric_by_weekday(df, metric_func, metric_name):
    df['weekday'] = df['timestamp'].dt.day_name()
    order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    result = df.groupby('weekday').apply(lambda g: metric_func(g['actual'], g['prediction'])).reset_index(name=metric_name)
    result['weekday'] = pd.Categorical(result['weekday'], categories=order, ordered=True)
    return result.sort_values('weekday')

def plot_bar_chart(data, x_col, y_col, title):
    plt.figure(figsize=(10, 5))
    plt.bar(data[x_col], data[y_col], color='skyblue')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

channel_1_path = r"C:\Users\elecf\Desktop\Licenta\Date\UK-DALE-disaggregated\house_1\predictii\LSTM\lstm_predictions_channel_1.csv"
channel_22_path = r"C:\Users\elecf\Desktop\Licenta\Date\UK-DALE-disaggregated\house_1\predictii\LSTM\lstm_predictions_channel_14.csv"

#incarcare fisiere
df1 = pd.read_csv(channel_1_path, parse_dates=['timestamp'])
df22 = pd.read_csv(channel_22_path, parse_dates=['timestamp'])

#channel 1 Agregat
smape_hour_1 = compute_metric_by_hour(df1, smape, 'SMAPE')
smape_weekday_1 = compute_metric_by_weekday(df1, smape, 'SMAPE')

r2_hour_1 = compute_metric_by_hour(df1, r2_score, 'R2')
r2_weekday_1 = compute_metric_by_weekday(df1, r2_score, 'R2')

plot_bar_chart(smape_hour_1, 'hour', 'SMAPE', 'SMAPE pe ora - Channel 1 (Agregat)')
plot_bar_chart(smape_weekday_1, 'weekday', 'SMAPE', 'SMAPE pe zi - Channel 1 (Agregat)')
plot_bar_chart(r2_hour_1, 'hour', 'R2', 'R² pe ora - Channel 1 (Agregat)')
plot_bar_chart(r2_weekday_1, 'weekday', 'R2', 'R² pe zi - Channel 1 (Agregat)')

#channel 14 LCD Office
smape_hour_22 = compute_metric_by_hour(df22, smape, 'SMAPE')
smape_weekday_22 = compute_metric_by_weekday(df22, smape, 'SMAPE')

r2_hour_22 = compute_metric_by_hour(df22, r2_score, 'R2')
r2_weekday_22 = compute_metric_by_weekday(df22, r2_score, 'R2')

plot_bar_chart(smape_hour_22, 'hour', 'SMAPE', 'SMAPE pe ora - Channel 14 (LCD Office)')
plot_bar_chart(smape_weekday_22, 'weekday', 'SMAPE', 'SMAPE pe zi - Channel 14 (LCD Office)')
plot_bar_chart(r2_hour_22, 'hour', 'R2', 'R² pe ora - Channel 14 (LCD Office)')
plot_bar_chart(r2_weekday_22, 'weekday', 'R2', 'R² pe zi - Channel 14 (LCD Office)')
