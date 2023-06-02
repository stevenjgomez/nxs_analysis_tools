import sys
sys.path.append('../src/nxs_analysis_tools/')
from datareduction import load_data, Scissors
data = load_data('../docs/source/examples/example_data/sample_name/15/example_hkli.nxs')
scissors = Scissors(data, center=(0,0,0), window=(0.1,2,0.3))
scissors.cut_data()
print(scissors.integration_window)
scissors.plot_integration_window()
scissors.linecut.plot()
scissors.highlight_integration_window()