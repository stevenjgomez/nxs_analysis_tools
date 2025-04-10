from nxs_analysis_tools.datareduction import load_data,plot_slice
from nxs_analysis_tools.datareduction import rotate_data
import matplotlib.pyplot as plt
import numpy as np
from nexusformat.nexus import *

h = np.linspace(-4,4,400)
k = np.linspace(-1,1, 100)
l = np.linspace(-2, 2, 200)
hh,kk,ll = np.meshgrid(h,k,l)
data = np.zeros(hh.shape)
data[np.logical_and(np.logical_and(np.abs(hh)<0.5,np.abs(kk)<0.5),np.abs(ll)<0.5)] = 1
data = data.transpose(1,0,2)
h = NXfield(h, name='H')
k = NXfield(k, name='K')
l = NXfield(l, name='L')
data = NXdata(NXfield(data, name='counts'), (h,k,l))
print(data.tree)
plot_slice(data[:,:,0.0])
plt.show()
rot = rotate_data(data[:,:,-0.2:0.2], rotation_angle=90, lattice_angle=90, rotation_axis=2, printout=True)
plot_slice(rot[:,:,0.0], data.H*1/4, data.K*4/1)
plt.show()