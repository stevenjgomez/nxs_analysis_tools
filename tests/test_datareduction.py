import sys
sys.path.append('../src/nxs_analysis_tools/')
from datareduction import load_data, Scissors, rotate_data, plot_slice

data = load_data('../docs/source/examples/example_data/sample_name/15/example_hkli.nxs')
# scissors = Scissors(data, center=(0,0,0), window=(0.1,2,0.3))
# scissors.cut_data()
# print(scissors.integration_window)
# scissors.plot_integration_window()
# scissors.linecut.plot()
# scissors.highlight_integration_window()

import matplotlib.pyplot as plt

plot_slice(data[:, :, 0.0])
plt.show()
rotated_data = rotate_data(data=data, lattice_angle=90, rotation_angle=45, rotation_axis=2, printout=True)
plot_slice(rotated_data[:, :, 0.0])
plt.show()
