import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class PlotAnalyzer:
    def __init__(self, data_dict, labels):
        self.data_dict = data_dict
        self.labels = labels

    def plot_histograms(self):
        """Plot histograme pentru consumul de energie al fiecarui appliance."""
        for channel, data in self.data_dict.items():
            if data is not None:
                plt.figure(figsize=(10, 6))
                plt.hist(data['power'], bins=50, alpha=0.7, label=f"{self.labels.get(channel, 'Unknown')}")
                plt.xlabel("Power (W)")
                plt.ylabel("Frequency")
                plt.title(f"Histogram of {self.labels.get(channel, 'Unknown')} Power Consumption")
                plt.legend()
                plt.show()

    def plot_correlograms(self):
        """Plot correlograme to understand relationships between appliances."""
        combined_data = pd.DataFrame()
        for channel, data in self.data_dict.items():
            if data is not None:
                combined_data[self.labels.get(channel, f"{channel}")] = data['power'].reindex(combined_data.index, fill_value=0)

        if not combined_data.empty:
            plt.figure(figsize=(12, 10))
            sns.heatmap(combined_data.corr(), annot=True, fmt=".2f", cmap="coolwarm")
            plt.title("Correlogram of Appliance Power Consumption")
            plt.show()

    def plot_future_predictions(self, future_file, future_steps):
        """
        Afiseaza predictiile generate pentru viitor.
        """
        data = pd.read_csv(future_file)

        plt.figure(figsize=(12, 6))
        plt.plot(data['Future Predictions'], label=f'Predicted Power ({future_steps} steps)', color='red')
        plt.xlabel("Time Steps")
        plt.ylabel("Power Consumption (W)")
        plt.title(f"Future Power Consumption Predictions for {future_steps} Steps")
        plt.legend()
        plt.show()
