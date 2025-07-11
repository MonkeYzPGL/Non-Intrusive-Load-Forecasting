import os
from analysis.DataAnalysis import DataAnalyzer

house_dir = r'C:\Users\elecf\Desktop\Licenta\Date\UK-DALE-disaggregated\house_1'
labels_file = os.path.join(house_dir, "labels.dat")
csv_dir = os.path.join(house_dir, "downsampled", "1H")

analyzer = DataAnalyzer(house_dir, labels_file, [])
analyzer.plot_downsampled_csvs(csv_dir)
