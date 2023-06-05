import sys
sys.path.append('../src/nxs_analysis_tools/')
from datareduction import load_data
from pairdistribution import Symmetrizer3D
import matplotlib.pyplot as plt

data = load_data('../docs/source/examples/example_data/pairdistribution_data/test_hkli.nxs')
s = Symmetrizer3D(data)

s.plane1symmetrizer.set_parameters(theta_min=0, theta_max=90, mirror=True)
s.plane1symmetrizer.test(data[:,:,len(data.L)//2])

s.plane2symmetrizer.set_parameters(theta_min=45, theta_max=90, mirror=True)
s.plane2symmetrizer.test(data[:,len(data.K)//2,:])

s.plane3symmetrizer.set_parameters(theta_min=0, theta_max=90, mirror=False)
s.plane3symmetrizer.test(data[len(data.H)//2,:,:])
#
s.symmetrize()