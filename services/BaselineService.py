import os

import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf


class BaselineGenerator:

    @staticmethod
    def last_week(input_csv, output_csv):
        df = pd.read_csv(input_csv)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        df.set_index('timestamp', inplace=True)

        df['prediction'] = df['power'].shift(24 * 7)
        df = df.dropna().reset_index()
        df.rename(columns={'power': 'actual'}, inplace=True)

        df.to_csv(output_csv, index=False)
        print(f" Baseline Last Week salvat: {output_csv}")

    @staticmethod
    def moving_average(input_csv, output_csv, window=24):
        df = pd.read_csv(input_csv)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')

        df['prediction'] = df['power'].rolling(window=window).mean().shift(1)
        df.dropna(inplace=True)
        df.rename(columns={'power': 'actual'}, inplace=True)

        df.to_csv(output_csv, index=False)
        print(f" Baseline Moving Average salvat: {output_csv}")

    @staticmethod
    def seasonal_hourly(input_csv, output_csv, history_days=5):
        df = pd.read_csv(input_csv)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        df.set_index('timestamp', inplace=True)

        df['dayofweek'] = df.index.dayofweek
        df['hour'] = df.index.hour

        predictions = []
        actuals = []
        timestamps = []

        for current_time in df.index[history_days * 24:]:
            dow = current_time.dayofweek
            hour = current_time.hour
            history = df[:current_time].query('dayofweek == @dow and hour == @hour').tail(history_days)

            if len(history) < 1:
                continue

            prediction = history['power'].mean()
            actual = df.loc[current_time]['power']

            predictions.append(prediction)
            actuals.append(actual)
            timestamps.append(current_time)

        result_df = pd.DataFrame({
            'timestamp': timestamps,
            'prediction': predictions,
            'actual': actuals
        })

        result_df.to_csv(output_csv, index=False)
        print(f" Baseline Seasonal Hourly salvat: {output_csv}")
