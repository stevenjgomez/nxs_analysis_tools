'''Exception classes.'''

from __future__ import annotations

__all__ = ['MagentroError', 'UnitError', 'MissingDataError']

class MagentroError(Exception):
    '''Base exception class for |MagentroData|.'''

class UnitError(ValueError, MagentroError):
    '''Exception class for invalid units or conversions.'''

class MissingDataError(MagentroError):
    '''Exception class for attempting to plot or operate on empty data.'''
