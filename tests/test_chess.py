import matplotlib.pyplot as plt
from nxs_analysis_tools import TempDependence

sample = TempDependence()

sample.load_datasets(folder='../docs/source/examples/example_data/sample_name')

print(sample.datasets)

print(sample.datasets['15'])

sample.datasets['15'][:,:,0.0].plot()

sample.cut_data(center=(0,0,0), window=(0.1,1,0.1))

sample.plot_linecuts()