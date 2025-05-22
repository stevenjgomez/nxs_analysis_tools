import numpy as np
import matplotlib.pyplot as plt
from nxs_analysis_tools import *

from nxs_analysis_tools.pairdistribution import Interpolator

data = load_data('docs/source/examples/example_data/plot_slice_data/cubic_hkli.nxs')
h = Interpolator()
h.set_data(data)
h.set_ellipsoidal_tukey_window(tukey_alpha=0.0)
h.apply_window()
fig,axs = plt.subplots(2,1, figsize=(4,8), dpi=100)
plot_slice(h.tapered[:,:,0.0], vmin=0, vmax=0.01, ax=axs[0])
plot_slice(h.tapered[:,0.0,:], vmin=0, vmax=0.01, ax=axs[1])
plt.show()