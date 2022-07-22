'''Type definitions.'''

from __future__ import annotations

from typing import Any, Union, Optional, TypedDict, Literal
from numpy.random import Generator, BitGenerator, SeedSequence
from numpy.typing import ArrayLike, _ArrayLikeInt_co
from pandas._typing import FilePath, ReadCsvBuffer

__all__ = [
    'ReadCsvFile', 'RngSeed', 'GriddataMethod', 'Kwargs', 'MatchErrStr', 'ToMinimizeStr',
    'PlotType', 'DataProp', 'PlotDataVersion', 'MapDataVersion', 'GroupingProp',
    'ColumnDataDict', 'Presets', 'SetterPresets'
]

ReadCsvFile = Union[FilePath, ReadCsvBuffer[bytes], ReadCsvBuffer[str]]

RngSeed = Optional[Union[_ArrayLikeInt_co, SeedSequence, BitGenerator, Generator]]

GriddataMethod = Literal['linear', 'nearest', 'cubic']

Kwargs = dict[str, Any]

MatchErrStr = Literal['min', 'mean', 'max']

ToMinimizeStr = Literal['gcv', 'std', 'err']

PlotType = Literal['lines', 'map']

DataProp = Literal['M_per_mass', 'dM_dT', 'Delta_SM']

PlotDataVersion = Literal['raw', 'converted', 'processed', 'compare']

MapDataVersion = Literal['raw', 'converted', 'processed']

GroupingProp = Literal['H', 'T']

class ColumnDataDict(TypedDict):
    '''Type for ``dict`` describing column data, such as temperature and magnetic moment.'''

    T: str
    H: str
    M: str
    M_err: str
    M_per_mass: str
    M_per_mass_err: str
    dM_dT: str
    Delta_SM: str

class Presets(TypedDict):
    '''
    Type for :meth:`process_data() <magentropy.MagentroData.process_data>` presets.

    .. admonition:: Implementation note
        :class: note

        These should all have defaults in
        :attr:`_DEFAULT_PRESETS <magentropy.MagentroData._DEFAULT_PRESETS>`,
        be parameters in
        :meth:`process_data() <magentropy.MagentroData.process_data>`,
        and be verified and returned in
        :func:`_validation.check_presets() <magentropy._validation.check_presets>`.
    '''

    npoints: int
    temp_range: ArrayLike
    fields: ArrayLike
    decimals: int
    max_diff: float
    min_sweep_len: int
    d_order: int
    lmbds: ArrayLike
    lmbd_guess: float
    weight_err: bool
    match_err: Union[bool, ArrayLike, MatchErrStr]
    min_kwargs: Kwargs
    add_zeros: bool

class SetterPresets(TypedDict):
    '''
    Same as :class:`Presets`, except all are optional.

    For :attr:`presets <magentropy.MagentroData.presets>` setter typing.
    '''

    npoints: Optional[int]
    temp_range: Optional[ArrayLike]
    fields: Optional[ArrayLike]
    decimals: Optional[int]
    max_diff: Optional[float]
    min_sweep_len: Optional[int]
    d_order: Optional[int]
    lmbds: Optional[ArrayLike]
    lmbd_guess: Optional[float]
    weight_err: Optional[bool]
    match_err: Optional[Union[bool, ArrayLike, MatchErrStr]]
    min_kwargs: Optional[Kwargs]
    add_zeros: Optional[bool]
