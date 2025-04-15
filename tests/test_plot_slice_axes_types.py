from src.nxs_analysis_tools import *
import matplotlib.pyplot as plt
import numpy as np

data = load_data('../docs/source/examples/example_data/plot_slice_data/cubic_hkli.nxs')

# NXdata with inherent axes
_,ax = plt.subplots()
plot_slice(data[:,:,0.0], ax=ax)
# plt.show()

# NXdata with NXfield axes
_,ax = plt.subplots()
plot_slice(data[:,:,0.0],X=data.nxaxes[0]*2,Y=data.nxaxes[1]*2, ax=ax)
# plt.show()

# NXdata with ndarray axes
_,ax = plt.subplots()
plot_slice(data[:,:,0.0],X=np.linspace(-0.1,0.1,len(data.nxaxes[0])),Y=np.linspace(-0.3,0.3,len(data.nxaxes[1])), ax=ax)
# plt.show()

# ndarray with inherent axes
_,ax = plt.subplots()
plot_slice(data[:,:,0.0].counts.nxdata, ax=ax)
# plt.show()

# ndarray with NXfield axes
_,ax = plt.subplots()
plot_slice(data[:,:,0.0].counts.nxdata,X=data.nxaxes[0]*2,Y=data.nxaxes[1]*2, ax=ax)
# plt.show()

# ndarray with ndarray axes
_,ax = plt.subplots()
plot_slice(data[:,:,0.0].counts.nxdata,X=np.linspace(-0.1,0.1,len(data.nxaxes[0])),Y=np.linspace(-0.3,0.3,len(data.nxaxes[1])), ax=ax)
# plt.show()

# ndarray with invalid axes
# _,ax = plt.subplots()
# plot_slice(data[:,:,0.0].counts.nxdata,X='test',Y='test', ax=ax)
# plt.show()