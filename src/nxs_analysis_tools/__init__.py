'''
Reduce and transform nexus format (.nxs) scattering data.
'''

import numpy as np
from _meta import __author__, __copyright__, __license__, __version__
from .datareduction import *
from .chess import TempDependence

# What to import when running "from nxs_analysis_tools import *"
__all__ = ['load_data', 'load_transform', 'plot_slice', 'Scissors',
           'reciprocal_lattice_params', 'rotate_data', 'rotate_data_2D',
           'convert_to_inverse_angstroms', 'array_to_nxdata', 'Padder',
           'rebin_nxdata', 'rebin_3d', 'rebin_1d'] + ['TempDependence']
