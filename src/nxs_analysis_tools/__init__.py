'''
Reduce and transform nexus format (.nxs) scattering data.
'''

from _meta import __author__, __copyright__, __license__, __version__
from .datareduction import plot_slice, cut_data

# What to import when running "from nxs_analysis_tools import *"
__all__ = ['plot_slice','cut_data']
