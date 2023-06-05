import sys
sys.path.append('../src/nxs_analysis_tools/')
from datareduction import load_data
from pairdistribution import Symmetrizer3D
import matplotlib.pyplot as plt

data = load_data('../docs/source/examples/example_data/pairdistribution_data/test_hkli.nxs')
s = Symmetrizer3D(data)

s.set_plane1symmetrizer(theta_min=0, theta_max=90, mirror=True)
s.test_plane1(data[:,:,len(data.L)//2])

s.set_plane2symmetrizer(theta_min=45, theta_max=90, mirror=True)
s.test_plane2(data[:,len(data.K)//2,:])

s.set_plane3symmetrizer(theta_min=0, theta_max=90, mirror=False)
s.test_plane3(data[len(data.H)//2,:,:])

s.symmetrize()