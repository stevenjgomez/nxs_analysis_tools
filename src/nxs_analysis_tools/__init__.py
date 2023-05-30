'''
Reduce and transform nexus format (.nxs) scattering data.
'''

from _meta import __author__, __copyright__, __license__, __version__
from .datareduction import load_data, plot_slice, Scissors
from .chess import TempDependence

# What to import when running "from nxs_analysis_tools import *"
__all__ = ['load_data','plot_slice','Scissors', 'TempDependence']
