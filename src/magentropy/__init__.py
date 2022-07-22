'''
Perform magnetoentropic mapping of magnetic materials based on DC
magnetization data.
'''

from _meta import __author__, __copyright__, __license__, __version__
from .magentro import MagentroData
from . import errors
from . import typedefs

__all__ = ['MagentroData', 'errors', 'typedefs']
