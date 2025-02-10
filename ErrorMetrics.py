import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class ErrorMetricsAnalyzer:
    def __init__(self, predictions, actuals, output_path):
        self.predictions = np.array(predictions)
        self.actuals = np.array(actuals)
        self.output_path = output_path

    def mse(self):
        return mean_squared_error(self.actuals, self.predictions)

    def rmse(self):
        return np.sqrt(self.mse())

    def mae(self):
        return mean_absolute_error(self.actuals, self.predictions)

    def r2(self):
        return r2_score(self.actuals, self.predictions)

    def mape(self):
        mask = self.actuals != 0  # Evită impartirea la 0
        return np.mean(np.abs((self.actuals[mask] - self.predictions[mask]) / self.actuals[mask])) * 100

    def smape(self):
        denominator = (np.abs(self.actuals) + np.abs(self.predictions)) / 2
        mask = denominator != 0  # Evită impartirea la 0
        return np.mean(np.abs(self.actuals[mask] - self.predictions[mask]) / denominator[mask]) * 100

    def compute_all_metrics(self):
        metrics = {
            "MSE": self.mse(),
            "RMSE": self.rmse(),
            "MAE": self.mae(),
            "R2 Score": self.r2(),
            "MAPE": self.mape(),
            "SMAPE": self.smape()
        }
        return metrics

    def save_metrics(self):
        metrics = self.compute_all_metrics()
        df = pd.DataFrame(metrics.items(), columns=["Metric", "Value"])
        df.to_csv(self.output_path, index=False)
