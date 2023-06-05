import sys
sys.path.append('../src/nxs_analysis_tools/')
from datareduction import load_data, plot_slice
from pairdistribution import Symmetrizer3D
import matplotlib.pyplot as plt

data = load_data('../docs/source/examples/example_data/pairdistribution_data/test_hkli.nxs')
s = Symmetrizer3D(data)
s.set_symmetry_operation_plane1(theta_min=0, theta_max=90, mirror=False)
s.test_symmetry_operation_plane1(data[:,:,len(data.L)//2])