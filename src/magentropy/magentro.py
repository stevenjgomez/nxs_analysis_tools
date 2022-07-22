'''
Perform magnetoentropic mapping of magnetic materials based on DC
magnetization data.
'''

from __future__ import annotations

from typing import ClassVar, Any, Optional, Union, Callable, Sequence, Hashable, get_args
import warnings
import logging
import re
from io import StringIO
from copy import deepcopy

import numpy as np
from numpy.typing import ArrayLike, NDArray

import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy

import pint
import pint_pandas

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Colormap

from .errors import UnitError, MissingDataError
from .typedefs import (
    ReadCsvFile, RngSeed, GriddataMethod,
    Kwargs, MatchErrStr, PlotType, DataProp, PlotDataVersion, MapDataVersion,
    ColumnDataDict, Presets, SetterPresets
)
from . import _classvars
from . import _validation as vd
from . import _pint_utils as pu
from . import _grouping as gr
from . import _calculations as calc
from . import _plotting as pt

__all__ = ['MagentroData']

# configure simple logging
logger = logging.getLogger('magentropy')
logger.setLevel(logging.INFO)
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(message)s')
console.setFormatter(formatter)
logger.addHandler(console)

class MagentroData:
    '''
    Representation of DC magnetization data.

    Magnetization data is collected for a sample by varying the temperature
    monotonically for each of many magnetic field strengths.
    This class provides methods for reading, processing, and plotting the data.

    Uses the :mod:`pint` package internally for unit convertsions.

    Notes
    -----
    All |DataFrame| attributes (:attr:`raw_df`, :attr:`converted_df`, etc.)
    are immutable and return copies of the internal instance attributes.
    If repeated access is required, for example to a |DataFrame|'s columns,
    it is best to first save the |DataFrame| as a local variable to avoid
    repeatedly copying large amounts of data.
    '''

    _QD_DAT_SPLITTING_STR: ClassVar[str] = '\n[Data]\n'
    _SAMPLE_MASS_DEFAULT_VALUE: ClassVar[float] = 1.0
    _SAMPLE_MASS_DEFAULT_UNITS: ClassVar[str] = 'mg'
    _RAW_DF_COLUMNS: ClassVar[ColumnDataDict] = _classvars.RAW_DF_COLUMNS
    _RAW_DF_DEFAULT_UNITS: ClassVar[ColumnDataDict] = _classvars.RAW_DF_DEFAULT_UNITS
    _INITIAL_UNITS: ClassVar[ColumnDataDict] = _classvars.INITIAL_UNITS
    _CONVERTED_DF_COLUMNS: ClassVar[ColumnDataDict] = _classvars.CONVERTED_DF_COLUMNS
    _CONVERTED_DF_UNITS: ClassVar[ColumnDataDict] = _classvars.CONVERTED_DF_UNITS
    _PROCESSED_DF_COLUMNS: ClassVar[ColumnDataDict] = _classvars.PROCESSED_DF_COLUMNS
    _PROCESSED_DF_UNITS: ClassVar[ColumnDataDict] = _classvars.PROCESSED_DF_UNITS
    _CONVERSIONS: ClassVar[dict[str, dict[str, Callable[[float], float]]]] = _classvars.CONVERSIONS
    _DEFAULT_PRESETS: ClassVar[Presets] = _classvars.DEFAULT_PRESETS

    def __init__(
        self,
        file_or_df: Union[ReadCsvFile, pd.DataFrame],
        qd_dat: bool = True,
        comment_col: Optional[Hashable] = 'Comment',
        T: Hashable = 'Temperature (K)',
        H: Hashable = 'Magnetic Field (Oe)',
        M: Hashable = 'Moment (emu)',
        M_err: Optional[Hashable] = 'M. Std. Err. (emu)',
        sample_mass: Optional[float] = None,
        units_level: Optional[Union[int, str]] = None,
        raw_data_units: Optional[dict[str, str]] = None,
        presets: Optional[SetterPresets] = None,
        **read_csv_kwargs: Any
        ) -> None:
        '''
        Initialize data with a source file or |DataFrame|.

        Parameters
        ----------
        file_or_df : str, path object, file-like object, or DataFrame
            An input file or |DataFrame|. A |DataFrame| should have the
            specified columns (parameters `comment_col` through `M_err`).
            Files will be read by :func:`pandas.read_csv` with additional
            arguments given in `**read_csv_kwargs`, and the resultant
            |DataFrame| should have the specified columns.
        qd_dat : bool, default True
            If ``True`` and `file_or_df` is not a |DataFrame|, the input file is
            assumed to be a Quantum Design ``.dat`` file with the sample mass
            given in the header as "``INFO,<sample_mass>,SAMPLE_MASS``" and the
            delimited data separated from the header by "``\\n[Data]\\n``".
            The delimited data will then be read by :func:`pandas.read_csv`
            with additional arguments given in `**read_csv_kwargs`.
        comment_col : label, optional, default 'Comment'
            The name of the input |DataFrame|'s comment column. If a row has a
            non-``NaN`` value in the comment column, it will be omitted.
            Set to ``None`` to ignore (do not omit any rows based on comments).
        T : label, default 'Temperature (K)'
            The name of the input temperature column.
        H : label, default 'Magnetic Field (Oe)'
            The name of the input magnetic field strength column.
        M : label, default 'Moment (emu)'
            The name of the input magnetic moment column.
            (Moment only, not per mass unit.)
        M_err : label, optional, default 'M. Std. Err. (emu)'
            The name of the input moment standard error column.
        sample_mass : float, optional
            The mass of the sample that was measured. If supplied, this will
            override any value determined from an input file when `qd_dat`
            is ``True``. Defaults to 1.0. Keep default of 1.0 if magnetic
            moment was already measured per mass unit, and set the units of
            moment and sample mass so that the dimensionality is correct.
        units_level : int or str, optional
            If supplied, data is expected to have units specified in this
            level of the column index.
            The column name parameters should still account for this level
            so they each refer to a single |Series|
            (i.e., include all levels in the column names).
        raw_data_units : dict, optional
            Keyword arguments specifying the units of the raw data that
            will be passed to `set_raw_data_units`. If supplied, this will
            override any units determined from column levels when `units_level`
            is supplied.
        presets : dict, optional
            Keyword arguments to pass to :meth:`set_presets`.
            See :meth:`set_presets` and :meth:`process_data` for more info.
        **read_csv_kwargs
            Passed to :func:`pandas.read_csv` for reading delimited data.
        '''

        self._init_ureg()

        self._init_dfs()
        self._init_sample_mass_and_presets()

        if raw_data_units is not None:
            raw_data_units = {key: unit for key, unit in raw_data_units.items() if unit is not None}
        else:
            raw_data_units = {}

        self._prep_data(
            file_or_df, qd_dat,
            comment_col, T, H, M, M_err,
            units_level, raw_data_units, sample_mass, presets,
            **read_csv_kwargs
        )

        with self._ureg.context('M_per_mass', 'dM/dT', 'Delta_SM'):
            self._convert_data()

    def _init_ureg(self) -> None:
        '''Initialize :mod:`pint` unit registry instance with correct settings.'''

        # initialize pint unit registry with pint-pandas
        ## see bad practice comment below
        ##pint_pandas.PintType.ureg = pint.UnitRegistry(on_redefinition='ignore')
        self._ureg: pint.UnitRegistry = pint_pandas.PintType.ureg

        # ureg settings
        self._reset_ureg()

    def _reset_ureg(self) -> None:
        '''Ensure :attr:`_ureg` has correct settings.'''

        self._ureg.disable_contexts()

        self._ureg.enable_contexts('Gaussian')

        ## bad practice, but more certain than setting only once as part of initialization (above)
        self._ureg._on_redefinition = 'ignore'

        # define emu, default unit in QD .dat files
        self._ureg.define('emu = 1 * erg / G')

        # short pretty string formatting
        self._ureg.default_format = '~P'

        # context for emu / g -> A * m**2 / kg conversion (M_per_mass)
        self._add_conversion_context(
            'M_per_mass',
            {'emu': 1, 'g': -1},
            {'A': 1, 'm': 2, 'kg': -1},
            lambda ureg, m_mass: (
                (m_mass * ureg('emu').to('A * m**2') / ureg('emu')
                ).to('A * m**2 / kg')
            )
        )

        # context for cal/K/g/Oe -> J/K/kg/T conversion (dM/dT)
        self._add_conversion_context(
            'dM/dT',
            {'cal': 1, 'K': -1, 'g': -1, 'Oe': -1},
            {'J': 1, 'K': -1, 'kg': -1, 'T': -1},
            lambda ureg, dmdt: (
                (dmdt * ureg('cal').to('J') / ureg('cal') * ureg('Oe') / ureg('Oe').to('T')
                ).to('J/K/kg/T')
            )
        )

        # context for cal/K/g -> J/K/kg conversion (Delta_SM)
        self._add_conversion_context(
            'Delta_SM',
            {'cal': 1, 'K': -1, 'g': -1},
            {'J': 1, 'K': -1, 'kg': -1},
            lambda ureg, delsm: (
                (delsm * ureg('cal').to('J') / ureg('cal')
                ).to('J/K/kg/T')
            )
        )

    def _add_conversion_context(
        self,
        name: str,
        units_from: dict[str, int],
        units_to: dict[str, int],
        conv_func: Callable[[pint.UnitRegistry, pint.Quantity], pint.Quantity]
        ) -> None:
        '''Add (but do not enable) a conversion context to :attr:`_ureg`.'''

        ctx = pint.Context(name)
        ctx.add_transformation(
            pint.util.UnitsContainer(units_from),
            pint.util.UnitsContainer(units_to),
            conv_func
        )
        self._ureg.add_context(ctx)

    def _init_dfs(self) -> None:
        '''Initialize raw, converted, and processed |DataFrame| instance variables.'''

        self._raw_df: pd.DataFrame = pu.df_from_units(
            self._RAW_DF_COLUMNS.values(),
            self._RAW_DF_DEFAULT_UNITS.values()
        )

        self._converted_df: pd.DataFrame = pu.df_from_units(
            self._CONVERTED_DF_COLUMNS.values(),
            self._CONVERTED_DF_UNITS.values()
        )

        self._processed_df: pd.DataFrame = pu.df_from_units(
            self._PROCESSED_DF_COLUMNS.values(),
            self._PROCESSED_DF_UNITS.values()
        )

    def _init_sample_mass_and_presets(self):
        '''Initialize `sample_mass` and `presets` instance variables.'''

        self._sample_mass: self._ureg.Quantity = (
            self._SAMPLE_MASS_DEFAULT_VALUE * self._ureg(self._SAMPLE_MASS_DEFAULT_UNITS)
        )
        self._presets: Presets = deepcopy(self._DEFAULT_PRESETS)
        self._last_presets: Optional[Presets] = None

    def _prep_data(
        self,
        file_or_df: Union[ReadCsvFile, pd.DataFrame],
        qd_dat: bool,
        comment_col: Optional[Hashable],
        T: Hashable,
        H: Hashable,
        M: Hashable,
        M_err: Optional[Hashable],
        units_level: Optional[Union[int, str]],
        raw_data_units: dict[str, str],
        sample_mass: Optional[float] = None,
        presets: Optional[SetterPresets] = None,
        **read_csv_kwargs: Any
        ) -> None:
        '''
        Clean, verify, and set :attr:`raw_df`. Sets units,
        :attr:`sample_mass`, and :attr:`presets`.

        Determines :attr:`sample_mass` from file and units from
        data if applicable.

        See :meth:`__init__` for parameter info.
        '''

        if isinstance(file_or_df, pd.DataFrame):
            prepped_df = file_or_df.copy()
        else:
            prepped_df, sample_mass = self._prep_file(
                file_or_df,
                qd_dat,
                sample_mass,
                **read_csv_kwargs
            )

        prepped_df.reset_index(drop=True, inplace=True)

        # remove abnormal measurement points specified by comment_col, then comment_col itself
        if comment_col is not None:
            drop_index = prepped_df.index[prepped_df.loc[:, comment_col].notna()]
            prepped_df.drop(index=drop_index, inplace=True)
            prepped_df.drop(columns=comment_col, inplace=True)

        # handle missing err column
        if M_err is None:
            expected_cols = pd.Series([T, H, M], dtype=object)
            col_keys = ['T', 'H', 'M']
        else:
            expected_cols = pd.Series([T, H, M, M_err], dtype=object)
            col_keys = ['T', 'H', 'M', 'M_err']

        # verify and reorder columns
        prepped_df = vd.use_expected_columns(prepped_df, expected_cols)

        # drop rows with missing values
        prepped_df.dropna(axis=0, subset=expected_cols, inplace=True)

        # only use units from df if not already given in raw_data_units
        if units_level is not None:
            units_list = list(prepped_df.columns.get_level_values(units_level))
            units_dict = dict(zip(['T', 'H', 'M'], units_list))
            raw_data_units = {**units_dict, **raw_data_units}

        # set units, sample mass, and presets before adding to raw df
        self._set_initial_settings(raw_data_units, sample_mass, presets)

        # get relevant raw df columns and units
        raw_col_names = [self._RAW_DF_COLUMNS[key] for key in col_keys]
        raw_units = [self._get_all_raw_data_units()[key] for key in col_keys]

        # fill raw df with prepped_df values
        self._raw_df.loc[:, raw_col_names] = pu.df_from_units(
            raw_col_names,
            raw_units,
            prepped_df.loc[:, expected_cols].values
        )

    @classmethod
    def _prep_file(
        cls,
        file: str,
        qd_dat: bool,
        sample_mass: Optional[float],
        **read_csv_kwargs: Any
        ) -> tuple[pd.DataFrame, float]:
        '''Process an input data file and return a |DataFrame| and sample mass ``float``.

        If `sample_mass` is ``None``, the returned `sample_mass` will either
        be the mass determined from the file, if `qd_dat` is ``True`` and mass
        is successfully found, or the default value for :attr:`sample_mass`
        otherwise.

        See :meth:`_prep_data` for parameter info.
        '''

        text_metadata = ''

        with open(file, 'r', encoding='utf-8') as f:

            if qd_dat:
                # try to split QD .dat file
                split_file = f.read().split(cls._QD_DAT_SPLITTING_STR)
                split_str_print = cls._QD_DAT_SPLITTING_STR.strip('\n')
                if len(split_file) == 2:
                    logger.info('"%s" tag detected, assuming QD .dat file.', split_str_print)
                    text_metadata = split_file[0]
                    text_data = split_file[1]
                else:
                    logger.warning(
                        '"%s" tag does not occur in %s exactly once. Now assuming delimited file.',
                        split_str_print, file
                    )
                    text_data = ''.join(split_file)
            else:
                text_data = f.read()

        if text_metadata != '' and sample_mass is None:

            try:
                # parse sample mass
                match = re.search(r'INFO,(\d+\.\d+),SAMPLE_MASS', text_metadata)
                sample_mass = float(match.group(1))
                logger.info('The sample mass was determined from the QD .dat file: %s', sample_mass)

            except (IndexError, AttributeError, TypeError, ValueError):
                sample_mass = cls._SAMPLE_MASS_DEFAULT_VALUE
                logger.warning(
                    'Using default sample mass of %s '
                    'because the sample mass could not be determined from %s.',
                    sample_mass, file
                )

        # write delimited data to file buffer and read into DataFrame
        delimited_file = StringIO()
        delimited_file.write(text_data)
        delimited_file.seek(0)
        df = pd.read_csv(delimited_file, **read_csv_kwargs)

        return df, sample_mass

    def _set_initial_settings(
        self,
        raw_data_units: dict[str, str],
        sample_mass: Optional[float],
        presets: Optional[SetterPresets]
        ):
        '''Set raw data units, sample mass, and presets.'''

        self.set_raw_data_units(**raw_data_units)

        if sample_mass is not None:
            self.sample_mass = sample_mass

        if presets is not None:
            self.presets = presets

    def _convert_data(self):
        '''Add per-mass columns to :attr:`raw_df` and convert all to :attr:`converted_df`.'''

        pu.create_per_mass_cols(
            self._raw_df,
            self._RAW_DF_COLUMNS,
            self._RAW_DF_DEFAULT_UNITS,
            self._sample_mass
        )

        if len(self._raw_df) > 0:
            pu.convert_df(
                self._raw_df,
                self._converted_df,
                self._RAW_DF_COLUMNS.values(),
                self._CONVERTED_DF_COLUMNS.values(),
                self._CONVERTED_DF_UNITS.values()
            )

    @property
    def raw_df(self) -> pd.DataFrame:
        '''A copy of the raw data.'''
        df = self.raw_df_with_units
        df.columns = self._RAW_DF_COLUMNS.values()
        return df

    @property
    def raw_df_with_units(self) -> pd.DataFrame:
        '''A copy of the raw data with a second header level indicating units.'''
        return self._raw_df.pint.dequantify()

    @property
    def converted_df(self) -> pd.DataFrame:
        '''A copy of the converted data (SI units).'''
        df = self.converted_df_with_units
        df.columns = self._CONVERTED_DF_COLUMNS.values()
        return df

    @property
    def converted_df_with_units(self) -> pd.DataFrame:
        '''A copy of the converted data (SI units) with a second header level indicating units.'''
        return self._converted_df.pint.dequantify()

    @property
    def processed_df(self) -> pd.DataFrame:
        '''A copy of the processed data.'''
        df = self.processed_df_with_units
        df.columns = self._PROCESSED_DF_COLUMNS.values()
        return df

    @property
    def processed_df_with_units(self) -> pd.DataFrame:
        '''A copy of the processed data with a second header level indicating units.'''
        return self._processed_df.pint.dequantify()

    @property
    def sample_mass(self) -> float:
        '''
        The magnitude of the sample mass.

        Can be set with a ``float``.
        '''

        return self._sample_mass.magnitude

    @sample_mass.setter
    def sample_mass(self, sample_mass: float) -> None:
        '''Set `sample_mass` magnitude with a ``float``.'''

        self._reset_ureg()

        sample_mass = float(sample_mass)

        if sample_mass <= 0:
            raise ValueError('sample_mass must be a positive number.')

        if len(self._raw_df) > 0:
            self._update_data(new_sample_mass=sample_mass)

        self._sample_mass = sample_mass * self._sample_mass.units

    @property
    def sample_mass_with_units(self) -> tuple[float, str]:
        '''
        The magnitude and units of the sample mass.

        Can be set with a ``tuple`` containing a ``float``
        (magnitude) and a ``str`` (units).
        '''

        return self._sample_mass.magnitude, f'{self._sample_mass.units:~P}'

    @sample_mass_with_units.setter
    def sample_mass_with_units(self, sample_mass_with_units: tuple[float, str]) -> None:
        '''
        Set `sample_mass` with a ``tuple`` containing a ``float``
        (magnitude) and a ``str`` (units).
        '''

        sample_mass_with_units = tuple(sample_mass_with_units)

        if len(sample_mass_with_units) != 2:
            raise ValueError('sample_mass_with_units must be a tuple of length 2')

        sample_mass_with_units = (float(sample_mass_with_units[0]), str(sample_mass_with_units[1]))

        prev_mass = self.sample_mass_with_units

        try:
            self.set_raw_data_units(sample_mass=sample_mass_with_units[1])
            self.sample_mass = sample_mass_with_units[0]
        except Exception as e:
            self.set_raw_data_units(sample_mass=prev_mass[1])
            self.sample_mass = prev_mass[0]
            raise e

    def get_raw_data_units(self) -> dict[str, str]:
        '''
        The raw data units for T, H, M, and sample mass.

        Returns
        -------
        dict
            Raw data units.
        '''

        units = self._get_all_raw_data_units()

        units.pop('M_err')
        units.pop('M_per_mass')
        units.pop('M_per_mass_err')
        units.pop('dM_dT')
        units.pop('Delta_SM')

        return units

    def _get_all_raw_data_units(self) -> dict[str, str]:
        '''Return a dict of all raw data units, including sample_mass.'''

        return {
            **dict(zip(
                self._RAW_DF_DEFAULT_UNITS.keys(),
                [pu.punits(self._raw_df.loc[:, col]) for col in self._RAW_DF_COLUMNS.values()]
            )),
            'sample_mass': f'{self._sample_mass.units:~P}'
        }

    def set_raw_data_units(
        self,
        T: Optional[str] = None,
        H: Optional[str] = None,
        M: Optional[str] = None,
        sample_mass: Optional[str] = None
        ) -> None:
        '''
        Set the units of the raw data.

        After the units are set, all other data is converted accordingly,
        so there is no need to re-process data if units are changed
        retroactively.

        Parameters
        ----------
        T, H, M, sample_mass : str, optional
            New units for temperature, magnetic field strength, measured
            moment, and sample mass, respectively. Moment is not per mass.
            Parameters left as ``None`` will not change the corresponding units.
        '''

        self._reset_ureg()

        new_units = {'T': T, 'H': H, 'M': M, 'sample_mass': sample_mass}
        new_units = {key: unit for key, unit in new_units.items() if unit is not None}

        self._verify_dimensionalities(new_units)

        if len(self._raw_df) > 0:
            self._update_data(new_raw_data_units=new_units)

        all_new_units = {**self._get_all_raw_data_units(), **new_units}
        all_new_units['M_err'] = all_new_units['M']

        new_columns = [
            (self._RAW_DF_COLUMNS[key], unit)
            for key, unit in all_new_units.items()
            if key != 'sample_mass'
        ]

        # dequantify raw df, change columns to reflect new units, re-quantify
        new_df = self.raw_df_with_units
        new_df.columns = pd.MultiIndex.from_tuples(new_columns)
        self._raw_df = new_df.pint.quantify(level=-1)

        # change sample mass units
        self._sample_mass = self._sample_mass.magnitude * self._ureg(all_new_units['sample_mass'])

    def _verify_dimensionalities(self, new_units: dict[str, str]) -> None:
        '''
        Verify dimensionalities of new units.

        Raises
        ------
        UnitError
            If a unit is invalid.
        '''

        for key, unit in new_units.items():
            if unit in self._ureg:
                if self._dimensionality_is_valid(key, unit):
                    logger.info(
                        '%s will have units of %s',
                        self._RAW_DF_COLUMNS[key] if key != "sample_mass" else key,
                        self._ureg.Unit(unit)
                    )
                else:
                    raise UnitError(
                        f'Mismatching dimensionality for {key}. '
                        f'("{unit}" has dimensionality '
                        f'{self._ureg(unit).dimensionality}.)'
                    )
            else:
                raise UnitError(f'Invalid unit "{unit}".')

    def _dimensionality_is_valid(self, key: str, unit: str) -> bool:
        '''Return ``True`` if unit is compatible with the default units, accessed by key.'''

        if key == 'sample_mass':
            valid_for_cgs = valid_for_si = self._ureg(unit).check('[mass]')
        else:
            valid_for_cgs = self._ureg(unit).check(self._RAW_DF_DEFAULT_UNITS[key])
            valid_for_si = self._ureg(unit).check(self._CONVERTED_DF_UNITS[key])

        return valid_for_cgs or valid_for_si

    def _update_data(
        self,
        new_raw_data_units: Optional[dict[str, str]] = None,
        new_sample_mass: Optional[float] = None
        ) -> None:
        '''
        Update all columns calculated from raw data upon a unit or sample mass change.

        Either new units or new sample mass should be supplied, not both.
        '''

        r_cols = self._RAW_DF_COLUMNS
        c_cols = self._CONVERTED_DF_COLUMNS
        p_cols = self._PROCESSED_DF_COLUMNS

        conv = self._CONVERSIONS

        raw_conv = {
            'M_per_mass': conv['M_per_mass'],
            'M_per_mass_err': conv['M_per_mass_err'],
            'dM_dT': conv['dM_dT'],
            'Delta_SM': conv['Delta_SM']
        }

        if new_raw_data_units is not None:
            update = new_raw_data_units
            if 'T' in update:
                self._update_temperature_cols(update['T'])

        elif new_sample_mass is not None:
            update = new_sample_mass

        # update contains either a dict (unit change) or a float (sample mass change)
        self._update_df(self._raw_df, r_cols, raw_conv, update)
        self._update_df(self._converted_df, c_cols, conv, update)
        self._update_df(self._processed_df, p_cols, conv, update)

    def _update_temperature_cols(self, new_T: str):
        '''
        Change :attr:`converted_df` and :attr:`processed_df` temperature columns
        due to :attr:`raw_df` unit change.
        '''

        c_cols = self._CONVERTED_DF_COLUMNS
        p_cols = self._PROCESSED_DF_COLUMNS

        # convert back to raw data units, but take only the values
        orig_T = self.get_raw_data_units()['T']
        c_orig_T_values = pu.pmag(self._converted_df.loc[:, c_cols['T']].pint.to(orig_T))
        p_orig_T_values = pu.pmag(self._processed_df.loc[:, p_cols['T']].pint.to(orig_T))

        # change the values to the new raw units and convert back to converted/processed units
        pu.convert_series(
            pd.Series(c_orig_T_values, dtype=f'pint[{new_T}]'),
            self._converted_df,
            c_cols['T'],
            self._CONVERTED_DF_UNITS['T']
        )
        pu.convert_series(
            pd.Series(p_orig_T_values, dtype=f'pint[{new_T}]'),
            self._processed_df,
            p_cols['T'],
            self._PROCESSED_DF_UNITS['T']
        )

    def _update_df(
        self,
        df: pd.DataFrame,
        cols: dict[str, str],
        conv: dict[str, dict[str, Callable[[float], float]]],
        update: Union[dict[str, str], float]
        ) -> None:
        ''' Scale a |DataFrame|'s columns according to the conversion functions in `conv`.'''

        if len(df) == 0:
            return

        if isinstance(update, dict):
            self._update_df_units(df, cols, conv, update)
        else:
            self._update_df_mass(df, cols, conv, update)

    def _update_df_units(
        self,
        df: pd.DataFrame,
        cols: dict[str, str],
        conv: dict[str, dict[str, Callable[[float], float]]],
        update: dict[str, str]
        ) -> None:
        '''Handle column scaling for unit changes.'''

        new_units_dict = update.items()
        old_units_dict = self.get_raw_data_units()

        # iterate over the provided conversion definitions for each column
        for col_key, conv_function_dict in conv.items():
            # dict of relevant updates for the column
            relevant_units_dict = {
                update_key: new_units for update_key, new_units in new_units_dict
                if update_key in conv_function_dict
            }
            if relevant_units_dict:
                self._scale_column_units(
                    df,
                    cols,
                    col_key,
                    relevant_units_dict,
                    old_units_dict,
                    conv_function_dict
                )

    def _scale_column_units(
        self,
        df: pd.DataFrame,
        cols: dict[str, str],
        col_key: str,
        relevant_units_dict: dict[str, str],
        old_units_dict: dict[str, str],
        conv_function_dict: dict[str, Callable[[float], float]]
        ) -> None:
        '''Scale a column based on relevant unit changes.'''

        col_to_update = df.loc[:, cols[col_key]]

        multiplier = np.prod([
            conv_function_dict[update_key](
                self._unit_ratio(new_units, old_units_dict[update_key], update_key=='T')
            )
            for update_key, new_units in relevant_units_dict.items()
        ])

        # more efficient to scale the ndarray and construct a new Series
        # (rather than directly scaling the PintArray, which is slow)
        df.loc[:, cols[col_key]] = pd.Series(
            pu.pmag(col_to_update) * multiplier,
            dtype=f'pint[{pu.punits(col_to_update)}]'
        )

    def _unit_ratio(self, new_units: str, old_units: str, temperature: bool) -> float:
        '''
        Return the ratio of `new_units` / `old_units` relative to `old_units`.

        If `temperature` is ``True``, the ratio will be calculated by comparing
        the unit temperature spacing.
        '''

        if temperature:
            # see pint documentation for information on delta temperature units
            new_diff = self._ureg.Quantity(2, new_units) - self._ureg(new_units)
            old_diff = self._ureg.Quantity(2, old_units) - self._ureg(old_units)
            ratio = (new_diff.to(old_diff.units) / old_diff).magnitude
        else:
            ratio = (self._ureg(new_units).to(old_units) / self._ureg(old_units)).magnitude

        return ratio

    def _update_df_mass(
        self,
        df: pd.DataFrame,
        cols: dict[str, str],
        conv: dict[str, dict[str, Callable[[float], float]]],
        update: float
        ) -> None:
        '''Handle column scaling for sample mass changes.'''

        # iterate over the provided conversion definitions for each column
        for col_key, conv_function_dict in conv.items():

            if 'sample_mass' in conv_function_dict:

                col_to_update = df.loc[:, cols[col_key]]

                multiplier = conv_function_dict['sample_mass'](update / self._sample_mass.magnitude)

                # more efficient to scale the ndarray and construct a new Series
                # (rather than directly scaling the PintArray, which is slow)
                df.loc[:, cols[col_key]] = pd.Series(
                    pu.pmag(col_to_update) * multiplier,
                    dtype=f'pint[{pu.punits(col_to_update)}]'
                )

    @property
    def presets(self) -> Presets:
        '''
        The current :meth:`process_data` presets.

        Can be set with a ``dict``.
        '''

        return deepcopy(self._presets)

    @presets.setter
    def presets(self, presets: SetterPresets) -> None:
        '''Set :attr:`presets` with a ``dict``.'''
        self.set_presets(**presets)

    def get_presets(self) -> Presets:
        '''Alias for attribute :attr:`presets`.'''

        return self.presets

    def set_presets(self, **kwargs: Any) -> None:
        '''
        Set :attr:`presets` for :meth:`process_data`.

        Parameters left as ``None`` will not change the corresponding preset.

        Parameters
        ----------
        **kwargs
            See :meth:`process_data` for valid parameters and parameter info.
        '''

        presets = vd.check_presets(self._presets, **kwargs)

        self._presets = deepcopy(presets)

    @property
    def last_presets(self) -> Optional[Presets]:
        '''
        The most recently used :meth:`process_data` presets,
        or ``None`` if :meth:`process_data` has not been run.
        '''

        return deepcopy(self._last_presets)

    def _set_last_presets(self, last_presets: Presets) -> None:
        '''Set last used presets.'''

        last_presets = vd.check_presets(self._last_presets, **last_presets)

        self._last_presets = deepcopy(last_presets)

    def to_string(self, **kwargs: Any) -> str:
        '''
        Render as console-friendly output.

        Parameters
        ----------
        **kwargs : Any
            Passed to :meth:`DataFrame.to_string() <pandas.DataFrame.to_string>`
            for rendering :attr:`raw_df_with_units`,
            :attr:`converted_df_with_units`, and
            :attr:`processed_df_with_units`. Excludes ``buf`` parameter.

        Returns
        -------
        str
            Console-friendly output.
        '''

        kwargs['buf'] = None
        return self._to_string(*self._dfs_to_string(**kwargs), html=False)

    def to_html(self, **kwargs: Any) -> str:
        '''
        Render as an HTML table.

        Parameters
        ----------
        **kwargs : Any
            Passed to :meth:`DataFrame.to_html() <pandas.DataFrame.to_html>`
            for rendering :attr:`raw_df_with_units`,
            :attr:`converted_df_with_units`, and
            :attr:`processed_df_with_units`. Excludes ``buf`` parameter.

        Returns
        -------
        str
            HTML table.
        '''

        kwargs['buf'] = None
        return self._to_string(*self._dfs_to_html(**kwargs), html=True)

    def _dfs_to_string(self, **kwargs) -> tuple[str, str, str]:
        '''Return ``*_df.to_string`` as ``tuple``, passing along ``**kwargs``.'''

        return (
            self.raw_df_with_units.to_string(**kwargs),
            self.converted_df_with_units.to_string(**kwargs),
            self.processed_df_with_units.to_string(**kwargs)
        )

    def _dfs_to_html(self, **kwargs) -> tuple[str, str, str]:
        '''Return ``*_df.to_html`` as ``tuple``, passing along ``**kwargs``.'''

        return (
            self.raw_df_with_units.to_html(**kwargs),
            self.converted_df_with_units.to_html(**kwargs),
            self.processed_df_with_units.to_html(**kwargs)
        )

    def _repr_html_(self) -> str:
        return self._to_string(
            self.raw_df_with_units._repr_html_(),
            self.converted_df_with_units._repr_html_(),
            self.processed_df_with_units._repr_html_(),
            html=True
        )

    def _to_string(self, raw_str: str, converted_str: str, processed_str: str, html: bool) -> str:
        '''Return a nicely-formatted str representing the class.'''

        beg, end, nl = ('<p>', '</p>', '<br>') if html else ('', '\n', '\n')
        presets_str = self._presets_str(self._presets, html = html)

        return (
            f'{beg}Sample mass: {self._sample_mass}{end}'
            f'{beg}Raw data:{end}{raw_str}{nl}'
            f'{beg}Converted data:{end}{converted_str}{nl}'
            f'{beg}Processed data:{end}{processed_str}{nl}'
            f'{beg}Presets:{nl}'
            f'{presets_str}{end}'
        )

    def __str__(self):
        return self.to_string()

    @staticmethod
    def _presets_str(presets: Presets, html: bool = False) -> str:
        '''Return a nicely-formatted str of the presets dict.'''

        nl, tab = ('<br>', '&emsp;') if html else ('\n', '\t')
        dict_str = f'{nl}{tab}'.join([f'{key}: {val},' for key, val in presets.items()])[:-1]

        return f'{{{nl}{tab}{dict_str}{nl}}}'

    @classmethod
    def sim_data(
        cls,
        temps: ArrayLike,
        fields: ArrayLike,
        sigma_t: float = 1e-6,
        sigma_h: float = 1e-6,
        sigma_m: float = 1e-6,
        random_seed: RngSeed = None,
        m_max: float = 0.01,
        slope: float = 1.5,
        bump_height: float = 0.1
        ) -> pd.DataFrame:
        '''
        Simulate data for testing and example purposes.

        The simulated data model function is a decreasing logistic function
        with maximum `m_max` plus a tiny Gaussian bump, the center of which
        varies linearly with field strength.

        The moment error column will the filled with `sigma_m`.

        Parameters
        ----------
        temps, fields : array_like
            Temperatures and fields at which to generate data.
        sigma_t, sigma_h, sigma_m : float, default 1e-6
            Standard deviation of random normally-distributed errors added
            to the temperatures, fields, and moments, respectively.
        random_seed : None, int, array_like[ints], SeedSequence, BitGenerator, or Generator
            Passed to :func:`numpy.random.default_rng`.
        m_max : float, default 0.01
            The limit as temperature goes to -inf of the moment for the
            highest field strength.
        slope : float, default 1.5
            Controls the steepness of the moment curves. Higher slope results
            in a faster decrease with temperature.
        bump_height : float, default 0.1
            Amplitude of the Gaussian bump as a proportion of `m_max`.

        Returns
        -------
        df : DataFrame
            Simulated data for temperature, field, moment, and moment error.
        '''

        temps = np.sort(np.array(temps, dtype=np.float64, ndmin=1).ravel())
        fields = np.sort(np.array(fields, dtype=np.float64, ndmin=1).ravel())

        if np.any(temps <= 0):
            raise ValueError('All temps must be positive.')

        if np.any(fields <= 0):
            raise ValueError('All fields must be positive.')

        rng = np.random.default_rng(random_seed)

        temps_err = rng.normal(0.0, sigma_t, (len(fields), len(temps)))
        fields_err = rng.normal(0.0, sigma_h, (len(fields), len(temps)))
        moment_err = rng.normal(0.0, sigma_m, (len(fields), len(temps)))

        m_max = abs(np.float64(m_max))
        slope = abs(np.float64(slope))
        bump_height = np.float64(bump_height)

        t_grid, h_grid = np.meshgrid(temps, fields)
        t_grid += temps_err
        h_grid += fields_err
        m_grid = calc.sim_func(temps, fields, m_max, slope, bump_height)
        m_grid += moment_err

        r_cols = cls._RAW_DF_COLUMNS

        return pd.DataFrame(
            np.column_stack((
                t_grid.ravel(),
                h_grid.ravel(),
                m_grid.ravel(),
                np.repeat(sigma_m, repeats=len(temps)*len(fields))
            )),
            columns = [r_cols['T'], r_cols['H'], r_cols['M'], r_cols['M_err']]
        )

    def test_grouping(
        self,
        fields: Optional[ArrayLike] = None,
        decimals: Optional[int] = None,
        max_diff: Optional[float] = None
        ) -> tuple[dict[str, Union[NDArray[np.float64], np.int64, np.float64]], DataFrameGroupBy]:
        '''
        Test grouping parameters before processing data, if desired.

        See :meth:`process_data` for parameter info. Default parameters are
        those in :attr:`presets`.

        Returns
        -------
        grouping_presets : dict
            The field grouping parameters `fields`, `decimals`, and `max_diff`
            after being checked and any defaults are used.
        grouped_by : DataFrameGroupBy
            Object on which one may test the results of the grouping.

        Notes
        -----
        The :meth:`pandas.core.groupby.DataFrameGroupBy.count` method is
        useful for viewing the number of observations in each field group.
        For example, ``test_grouping(...)['T'].count()`` returns a
        |DataFrame| of the group counts.
        Groups with less than `min_sweep_len` observations are
        ignored in :meth:`process_data`.
        '''

        presets = vd.check_presets(
            self._presets,
            fields=fields,
            decimals=decimals,
            max_diff=max_diff
        )

        # raw_df instead of _raw_df is intentional, want without units
        return gr.test_grouping(self.raw_df, presets)

    @classmethod
    def test_grouping_(
        cls,
        raw_df: pd.DataFrame,
        fields: Optional[ArrayLike] = None,
        decimals: Optional[int] = None,
        max_diff: Optional[float] = None
        ) -> tuple[dict[str, Union[NDArray[np.float64], np.int64, np.float64]], DataFrameGroupBy]:
        '''
        Class method corresponding to :meth:`test_grouping`.

        See :meth:`test_grouping` and :meth:`process_data` for parameters
        following `raw_df` and return values.
        Default parameters are the class defaults.

        A copy of `raw_df` is grouped, so the returned
        :class:`DataFrameGroupBy` cannot modify `raw_df`.

        Parameters
        ----------
        raw_df : DataFrame
            |DataFrame| on which to test grouping.
        '''

        vd.verify_df(raw_df, cls._RAW_DF_COLUMNS, 'raw_df')

        presets = vd.check_presets(
            cls._DEFAULT_PRESETS,
            fields=fields,
            decimals=decimals,
            max_diff=max_diff
        )

        return gr.test_grouping(raw_df.copy(), presets)

    def process_data(
        self,
        npoints: Optional[int] = None,
        temp_range: Optional[ArrayLike] = None,
        fields: Optional[ArrayLike] = None,
        decimals: Optional[int] = None,
        max_diff: Optional[float] = None,
        min_sweep_len: Optional[int] = None,
        d_order: Optional[int] = None,
        lmbds: Optional[ArrayLike] = None,
        lmbd_guess: Optional[float] = None,
        weight_err: Optional[bool] = None,
        match_err: Optional[Union[bool, ArrayLike, MatchErrStr]] = None,
        min_kwargs: Optional[Kwargs] = None,
        add_zeros: Optional[bool] = None
        ) -> None:
        '''
        Smooth magnetic moment and calculate raw, converted, and processed derivative and entropy.

        Groups raw data, smooths magnetic moment using Tikhonov regularization,
        and fills :attr:`processed_df`. Calculates derivative ``'dM_dT'`` and
        entropy ``'Delta_SM'`` for raw and converted data without smoothing, and
        for processed data using smoothed moment.

        Requires that all sweeps are taken on cooling, or all sweeps are taken
        on warming (monotonic). Warming and cooling sweeps should not both be
        included in the data.

        .. note::

            Rows of zero field and zero moment are prepended to the data
            before integration, so it is not necessary to include measurements
            at zero field in the input data. Whether or not the zeros are added
            to :attr:`processed_df` after processing can be controlled with
            `add_zeros`.

        Parameters left as the default ``None`` will use the corresponding values
        in :attr:`presets`. All parameters should be given in raw data units if
        applicable (`temp_range`, `max_diff`, etc.).

        Parameters
        ----------
        npoints : int, optional
            Number of temperature points in `temp_range` to use to output
            smoothed ``'M_per_mass'``, ``'dM_dT'``, and ``'Delta_SM'`` for each
            field strength.
        temp_range : (2,) array_like, optional
            Temperature range (inclusive) in raw data units over which to
            analyze the data. Bounds less than or greater than the lowest or
            highest given temperatures, respectively, will be adjusted to the
            data range when creating output temperatures.
        fields : array_like, optional
            Expected field strengths for grouping data. If `fields` has
            length zero, the groups are determined automatically based
            on `decimals` and/or `max_diff`.
        decimals : int, optional
            The decimal place to which to round the automatically determined
            field groups. (A negative integer specifies the number of positions
            to the left of the decimal point. See :func:`numpy.around`.)
            Ignored if `fields` has length greater than zero.
            If `fields` has length zero and `max_diff` is :data:`numpy.inf`,
            groups will be determined solely by rounding to this decimal place.
        max_diff : float, optional
            If `fields` has length greater than zero, `max_diff` is the maximum
            difference allowed between each raw field strength and the closest
            field in `fields`. Raw fields too far away from any field group will
            be omitted, unless `max_diff` is :data:`numpy.inf`.
            If `fields` has length zero, `max_diff` is the maximum difference
            allowed between any two items in each field group, which is used to
            determine groups automatically.
            Generally, `decimals` is enough to determine groups, but `max_diff`
            can be used for finer control, e.g. to get exact means.
        min_sweep_len : int, optional
            Minimum number of observations required for a field to be included
            in the smoothed output. Field groups with less than this number
            will be skipped.
        d_order : int, optional
            Order of derivative used to calculate roughness during
            regularization. For example, if `d_order` is 2, the second
            temperature derivative of magnetic moment is used to calculate the
            roughness. Generally 2 or 3 work well. Choice of `d_order` will
            change optimal regularization parameter :math:`\\lambda`.
        lmbds : array_like, optional
            Specifies regularization parameter :math:`\\lambda`.
            If `lmbds` is a single number (or :term:`array_like` of length 1),
            it will be applied to all magnetic field strengths.
            If `lmbds` is :data:`numpy.nan` or length 0, each :math:`\\lambda`
            will be determined automatically.
            If `lmbds` is the same length as the number of field strengths,
            each element (numerical or :data:`numpy.nan`) will be applied to
            the corresponding field, in order of increasing field strength.
        lmbd_guess : float, optional
            Initial guess for regularization parameter :math:`\\lambda` when
            determining automatically.
        weight_err : bool, optional
            If ``True``, weight measurements by the normalized inverse squares
            of the errors.
        match_err : bool, array_like, or one of {'min', 'mean', 'max'}, optional
            Ignored if :math:`\\lambda` is given in `lmbds`.
            If `match_err` is ``False``, use generalized cross validation (GCV)
            to find optimal :math:`\\lambda`.
            If `match_err` is ``True``, find optimal :math:`\\lambda` by
            matching absolute differences between the measured and smoothed
            values with the errors.
            If `match_err` is a single number (or :term:`array_like` of
            length 1), match the standard deviation of the absolute
            differences with this number.
            If `match_err` is the same length as the number of field strengths,
            each element (numeric) will be applied to the corresponding field,
            in order of increasing field strength.
            If `match_err` is one of ``'min'``, ``'mean'``, or ``'max'``,
            match the standard deviation of the absolute differences with the
            minimum, mean, or maximum error for each field.
        min_kwargs : dict, optional
            Keyword arguments to pass to :func:`scipy.optimize.minimize` when
            determining optimal :math:`\\lambda`. The parameters ``fun``,
            ``x0``, and ``args`` will be ignored if included. Note that
            :math:`\\log_{10}\\lambda` is passed to
            :func:`scipy.optimize.minimize`, so arguments such as
            ``bounds`` should be adjusted accordingly.
            (The same is not true, however, for `lmbd_guess`.)
        add_zeros : bool, optional
            If ``True``, rows of zeros corresponding to zero field and zero
            moment will be prepended to `processed_df` after processing.
        '''

        self._reset_ureg()

        presets = vd.check_presets(
            self._presets,
            npoints=npoints, temp_range=temp_range,
            fields=fields, decimals=decimals, max_diff=max_diff, min_sweep_len=min_sweep_len,
            d_order=d_order, lmbds=lmbds, lmbd_guess=lmbd_guess,
            weight_err=weight_err, match_err=match_err, min_kwargs=min_kwargs,
            add_zeros=add_zeros
        )

        presets_str = self._presets_str(presets)

        self._adjust_temp_range(presets)

        # short names for convenience
        n, tr, fds, dms, md, msl, d, lms, lm_g, we, me, mk, az = presets.values()
        r_cols = self._RAW_DF_COLUMNS

        # raw data without units
        raw_df = self.raw_df

        # group raw data (self.raw_data, no units) by M(T) sweeps
        by_field = gr.group_by(
            raw_df,
            group_col=r_cols['H'],
            groups=fds,
            decimals=dms,
            max_diff=md
        )

        # extract fields and sorted numpy data for groups with length greater than min_sweep_len
        used_fields, df_arrays = zip(*(
            (field, df.sort_values(by=r_cols['T']).values) for field, df in by_field
            if len(df) >= msl
        ))

        used_fields = np.array(used_fields, dtype=np.float64, ndmin=1)

        # limit to temp range
        df_arrays = tuple(
            data[(data[:, 0] >= tr[0]) & (data[:, 0] <= tr[1]), :] for data in df_arrays
        )

        # verify length of lmbds
        if len(lms) == 1:
            lms = lms.repeat(len(used_fields))
        elif len(lms) != len(used_fields):
            raise ValueError(
                'lmbds must be a single value or have one element for each '
                f'unique field. Expected length {len(used_fields)}, got length {len(lms)}.'
            )

        # verify length of match_err
        if not isinstance(me, bool) and len(me) not in (1, len(used_fields)):
            raise ValueError(
                'match_err must be a bool, a single value, or have one element for each '
                f'unique field. Expected length {len(used_fields)}, got length {len(me)}.'
            )

        logger.info(
            'The data contains the following %d '
            'magnetic field strengths and observations per field%s',
            by_field.ngroups,
            (':' if by_field.ngroups == len(used_fields) else '.')
        )

        if by_field.ngroups != len(used_fields):
            logger.warning(
                'Only %d of these will be used. '
                'Fields with less than %d observations will be skipped.',
                len(used_fields), msl
            )

        logger.info('%s\n', by_field[r_cols["T"]].count())
        logger.info('Processing data using the following settings:\n%s\n', presets_str)

        # processed data smoothing, differentiation, integration
        processed_df = calc.smooth_diff_int_processed(
            used_fields, df_arrays,
            n, tr, d, lms, lm_g, we, me, mk, az,
            self._initial_units(), self._sample_mass
        )

        # raw data differentiation and integration, fill raw df
        self._raw_df = self._interp_diff_int_raw(
            raw_df,
            n,
            tr[1] - tr[0],
            np.fromiter((grp for grp in by_field.groups.keys()), dtype=np.float64)
        )

        # last conversions, fill converted and processed dfs
        self._final_conversion(processed_df)

        # update last presets
        presets['fields'] = used_fields
        presets['lmbds'] = lms
        self._set_last_presets(presets)
        logger.info('last_presets set to:\n%s\n', self._presets_str(self.last_presets))

        logger.info('Finished.')

    def _adjust_temp_range(self, presets: Presets) -> None:
        '''
        Check that temperature range contains raw data temperatures and
        adjusts bounds that are lower or higher than necessary.
        '''

        r_cols = self._RAW_DF_COLUMNS

        if (presets['temp_range'][0] > pu.pmag(self._raw_df[r_cols['T']]).max() or
            presets['temp_range'][1] < pu.pmag(self._raw_df[r_cols['T']]).min()):
            raise ValueError(
                'No temperatures in range '
                f'{presets["temp_range"][0]} to {presets["temp_range"][1]}.'
            )

        presets['temp_range'][0] = max(
            presets['temp_range'][0],
            pu.pmag(self._raw_df[r_cols['T']]).min()
        )

        presets['temp_range'][1] = min(
            presets['temp_range'][1],
            pu.pmag(self._raw_df[r_cols['T']]).max()
        )

    def _initial_units(self) -> dict[str, str]:
        '''
        Return a ``dict`` containing the starting units for conversions on
        processed data.

        ``'T'``, ``'H'``, ``'M'``, and ``'M_err'`` get the
        current raw data units.
        ``'M_per_mass'`` and ``'M_per_mass_err'`` get the
        default raw data units.
        ``'dM_dT'`` and ``'Delta_SM'`` get the units that they
        would if they were to be constructed directly from the
        default raw data units (instead of the standard cgs energy units).
        '''

        r_units = self._RAW_DF_DEFAULT_UNITS
        initial_units = self._get_all_raw_data_units()
        initial_units.pop('sample_mass')
        initial_units['M_per_mass'] = r_units['M_per_mass']
        initial_units['M_per_mass_err'] = r_units['M_per_mass_err']
        initial_units['dM_dT'] = f'{initial_units["M_per_mass"]}/{initial_units["T"]}'
        initial_units['Delta_SM'] = f'{initial_units["dM_dT"]}*{initial_units["H"]}'

        return initial_units

    def _interp_diff_int_raw(
        self,
        raw_df: pd.DataFrame,
        n: np.int64,
        temp_diff: np.float64,
        fields: NDArray[np.float64]
        ) -> None:
        '''
        Perform the differentiation and integration on the raw data.

        Computes raw ``'dM_dT'`` and ``'Delta_SM'`` by interpolating raw values onto a
        regularly-spaced grid with (npoints * total temp range / temp range)
        temps and zero field plus the determined fields, taking the derivative
        and integral, and interpolating back to the original points.
        '''

        r_cols, r_units = self._RAW_DF_COLUMNS, self._RAW_DF_DEFAULT_UNITS

        t_h_raw_conv_values = pu.get_raw_conv_t_h(self._raw_df)
        calc.interp_diff_int_raw(raw_df, n, temp_diff, fields, t_h_raw_conv_values)

        # fill raw data with the data after operations and convert
        raw_df_with_units = pu.df_from_units(
            r_cols.values(),
            self._initial_units().values(),
            raw_df.values
        )
        conv_cols = [r_cols['dM_dT'], r_cols['Delta_SM']]
        conv_units = [r_units['dM_dT'], r_units['Delta_SM']]
        pu.convert_df(raw_df_with_units, raw_df_with_units, conv_cols, conv_cols, conv_units)

        logger.info('Calculated raw derivative and entropy.\n')

        return raw_df_with_units

    def _final_conversion(self, processed_df: pd.DataFrame) -> None:
        '''
        Convert :attr:`raw_df` to :attr:`converted_df`, and `processed_df` in
        raw data units to final :attr:`processed_df`.
        '''

        r_cols = self._RAW_DF_COLUMNS
        c_cols, c_units = self._CONVERTED_DF_COLUMNS, self._CONVERTED_DF_UNITS
        p_cols, p_units = self._PROCESSED_DF_COLUMNS, self._PROCESSED_DF_UNITS

        with self._ureg.context('M_per_mass', 'dM/dT', 'Delta_SM'):
            pu.convert_df(
                self._raw_df, self._converted_df,
                r_cols.values(), c_cols.values(),
                c_units.values()
            )
            pu.convert_df(
                processed_df, self._processed_df,
                p_cols.values(), p_cols.values(),
                p_units.values()
            )

    def bootstrap(self, n_bootstrap: int = 100, random_seed: RngSeed = None) -> None:
        '''
        Calculate bootstrap estimates of the errors in the smoothed
        magnetic moment output and fill :attr:`processed_df`'s ``'M_err'`` and
        ``'M_per_mass_err'`` columns.

        Parameters
        ----------
        n_bootstrap : int, default 100
            The number of times to sample from the data and fit a model.

        random_seed : None, int, array_like[ints], SeedSequence, BitGenerator, or Generator
            Passed to :func:`numpy.random.default_rng`.

        Notes
        -----
        Bootstrap procedures involve repeatedly sampling N points from data of
        length N *with replacement*, fitting a model to each data sample,
        and computing the parameter of interest from the `n_bootstrap` fitted
        models. In this case, the standard deviation of each smoothed
        magnetic moment point is computed from the values of the `n_bootstrap`
        models at each point.

        .. attention::
            The bootstrap method presented here is purely experimental and is
            not detailed in either of the sources listed on the
            :doc:`homepage <index>`.

        .. caution::
            This method is computationally expensive and can take upwards of
            ten minutes to run on typical magnetization data.

        .. important::

            Bootstrap estimates in the context of regularization are dependent
            on the chosen regularization parameter :math:`\\lambda`. These error
            estimates should not be viewed as "true" estimates but rather as
            the estimates for a given :math:`\\lambda`. This should only be used
            once the user is confident their :math:`\\lambda`'s are appropriate.

        TL;DR: Use wisely, and take the results with a grain of salt!
        '''

        if len(self._processed_df) == 0:
            raise MissingDataError(
                'processed_df is empty. '
                'Please run process_data() before calculating bootstrap estimates.'
            )

        # short names for convenience
        n, tr, fds, dms, md, msl, d, lms, _, we, _, _, az = self.last_presets.values()
        r_cols = self._RAW_DF_COLUMNS
        p_cols = self._PROCESSED_DF_COLUMNS
        p_units = self._PROCESSED_DF_UNITS

        # group raw data (self.raw_data, no units) by M(T) sweeps
        by_field = gr.group_by(
            self.raw_df,
            group_col=r_cols['H'],
            groups=fds,
            decimals=dms,
            max_diff=md
        )

        # extract fields and sorted numpy data for groups with length greater than min_sweep_len
        used_fields, df_arrays = zip(*(
            (field, df.sort_values(by=r_cols['T']).values) for field, df in by_field
            if len(df) >= msl
        ))

        used_fields = np.array(used_fields, dtype=np.float64, ndmin=1)

        # limit to temp range
        df_arrays = tuple(
            data[(data[:, 0] >= tr[0]) & (data[:, 0] <= tr[1]), :] for data in df_arrays
        )

        logger.info('Performing bootstrap calculations...')

        rng = np.random.default_rng(random_seed)

        processed_df = calc.bootstrap(
            used_fields, df_arrays,
            n, tr, d, lms, we, az,
            self._initial_units(), self._sample_mass,
            n_bootstrap, rng
        )

        with self._ureg.context('M_per_mass'):
            pu.convert_df(
                processed_df, self._processed_df,
                (p_cols['M_err'], p_cols['M_per_mass_err']),
                (p_cols['M_err'], p_cols['M_per_mass_err']),
                (p_units['M_err'], p_units['M_per_mass_err'])
            )

        logger.info('Finished.')

    @classmethod
    def plot_processed_lines(
        cls,
        processed_df: pd.DataFrame,
        compare_df: Optional[pd.DataFrame] = None,
        data_prop: DataProp = 'M_per_mass',
        ax: Optional[Axes] = None,
        T_range: ArrayLike = np.array([-np.inf, np.inf]),
        H_range: ArrayLike = np.array([-np.inf, np.inf]),
        offset: float = 0,
        at_temps: Optional[ArrayLike] = None,
        fields: Optional[ArrayLike] = None,
        decimals: Optional[int] = None,
        max_diff: Optional[float] = None,
        colormap: Optional[Union[Colormap, str]] = None,
        legend: bool = False,
        colorbar: bool = False,
        plot_kwargs: Optional[Union[Kwargs, Sequence[Kwargs]]] = None,
        compare_kwargs: Optional[Union[Kwargs, Sequence[Kwargs]]] = None,
        colorbar_kwargs: Optional[Kwargs] = None
        ) -> Union[Axes, tuple[Axes, Colorbar]]:
        '''
        Plot processed data from a |DataFrame| as lines.

        This class method allows already-processed data to be easily plotted so
        that raw data needn't be re-processed.

        See :meth:`plot_lines` for parameters following `compare_df` and
        return values.

        Parameters
        ----------
        processed_df : DataFrame
            Processed data. Expected to have column names matching those of the
            :attr:`processed_df`
            attribute of an instance of |MagentroData|.
        compare_df : DataFrame, optional
            Converted data. Expected to have column names matching those of the
            :attr:`converted_df`
            attrubute of an instance of |MagentroData|.
            Expected to use the same units as `processed_df`.
        '''

        c_cols = cls._CONVERTED_DF_COLUMNS
        p_cols = cls._PROCESSED_DF_COLUMNS

        # check arguments

        data_prop = vd.check_processed_args(processed_df, data_prop)

        if compare_df is not None:
            vd.verify_df(compare_df, c_cols.values(), 'compare_df')

        if compare_df is not None and len(compare_df) == 0:
            raise MissingDataError('compare_df is empty.')

        # get the settings and do the plotting
        presets = vd.check_presets(
            cls._DEFAULT_PRESETS,
            fields=fields,
            decimals=decimals,
            max_diff=max_diff
        )

        ax, cbar = pt.plot_processed_lines(
            processed_df, 'processed', data_prop, ax, p_cols,
            T_range, H_range, offset, at_temps,
            colormap, legend, colorbar, plot_kwargs, colorbar_kwargs, presets
        )

        # handle compare case
        if compare_df is not None:
            pt.plot_processed_lines(
                compare_df, 'compare', data_prop, ax, c_cols,
                T_range, H_range, offset, at_temps,
                colormap, False, False, compare_kwargs, None, presets
            )

        if cbar is None:
            return ax

        return ax, cbar

    def plot_lines(
        self,
        data_prop: DataProp = 'M_per_mass',
        data_version: PlotDataVersion = 'raw',
        ax: Optional[Axes] = None,
        T_range: ArrayLike = np.array([-np.inf, np.inf]),
        H_range: ArrayLike = np.array([-np.inf, np.inf]),
        offset: float = 0,
        at_temps: Optional[ArrayLike] = None,
        colormap: Optional[Union[Colormap, str]] = None,
        legend: bool = False,
        colorbar: bool = False,
        plot_kwargs: Optional[Union[Kwargs, Sequence[Kwargs]]] = None,
        compare_kwargs: Optional[Union[Kwargs, Sequence[Kwargs]]] = None,
        colorbar_kwargs: Optional[Kwargs] = None
        ) -> Union[Axes, tuple[Axes, Colorbar]]:
        '''
        Plot the moment per mass, derivative with respect to temperature,
        or entropy as lines.

        All parameter units should correspond to those of the data specified by
        `data_version`.

        Parameters
        ----------
        data_prop : {'M_per_mass', 'dM_dT', 'Delta_SM'}, default 'M_per_mass'
            The property to plot.
        data_version : {'raw', 'converted', 'processed', 'compare'}, default 'raw'
            The version of the data to plot. If ``'compare'``, converted and
            processed data will be plotted together.
        ax : Axes, optional
            |Axes| on which to plot. If ``None``, new |Axes| will be
            created from the current |Figure|.
        T_range, H_range : (2,) array_like
            Temperature and magnetic field strength ranges to display.
        offset : float, default 0
            If a nonzero `offset` is supplied, successive lines (fields or
            temperatures) will have an offset added to them.
            Good for seeing curve shapes at different fields or temperatures.
        at_temps : array_like, optional
            Temperatures to group data. If supplied, data will be plotted
            versus magnetic field strength instead of temperature, at the
            temperatures in the data that are closest to the supplied
            `at_temps`.
        colormap : Colormap or str, optional
            Color map to cycle through when plotting lines.
        legend : bool, default False
            If ``True``, add a legend to the |Axes| with |Axes.legend|.
        colorbar : bool, default False
            If ``True``, add a discrete color bar to the |Figure| containing
            `ax` with |Figure.colorbar|.
        plot_kwargs : dict or array_like of dicts, optional
            Keyword arguments for |Axes.plot|. A single ``dict`` will be
            applied to each line. Multiple ``dict``s will be applied to
            successive lines. Not checked for length; a ``dict`` is applied
            to each line until either the end is reached or there are no
            more lines to plot.
        compare_kwargs : dict or array_like of dicts, optional
            `plot_kwargs` for converted data, used if `data_version` is
            ``'compare'``.
        colorbar_kwargs : dict, optional
            Keyword arguments for |Figure.colorbar|, excluding ``mappable``.

        Returns
        -------
        ax : Axes
            If `colorbar` is ``False``, the |Axes| on which the data is plotted.
        ax, cbar : Axes, Colorbar
            If `colorbar` is ``True``, the |Axes| on which the data is plotted
            and the |Colorbar|.
        '''

        data_prop, data_version = self._check_prop_version(data_prop, data_version)

        last_presets = self.last_presets
        if last_presets is None:
            last_presets = self.get_presets()

        if at_temps is not None and len(last_presets['fields']) == 0:
            raise MissingDataError(
                'Could not determine field groups. In order to plot at temperatures, '
                'please supply fields in presets or run process_data() before plotting.'
            )

        case_params = self._grouping_plotting_cases(
            (data_version if data_version != 'compare' else 'processed'),
            data_prop,
            at_temps,
            last_presets
        )

        grouped_data = gr.group_by(
            case_params['group_df'],
            case_params['group_col'],
            groups=case_params['groups'],
            decimals=case_params['decimals'],
            max_diff=case_params['max_diff'],
            match_col=case_params['match_col'],
            match_values=case_params['match_values']
        )

        if data_version == 'converted':
            grouped_data = self._convert_grouped(grouped_data, at_temps)

        ax, cbar = pt.plot_grouped_data(
            data_prop, data_version, ax, grouped_data,
            case_params['group_col'], case_params['T_col'],
            case_params['H_col'], case_params['prop_col'],
            case_params['T_units'], case_params['H_units'], case_params['prop_units'],
            T_range, H_range, offset, at_temps, colormap,
            case_params['marker'], case_params['markersize'],
            case_params['linestyle'], case_params['legend_label'], case_params['colorbar_label'],
            legend, colorbar, plot_kwargs, colorbar_kwargs
        )

        # handle compare case
        if data_version == 'compare':
            self._handle_compare_lines(
                data_prop, ax,
                T_range, H_range, offset, at_temps,
                colormap, compare_kwargs, last_presets
            )

        if cbar is None:
            return ax

        return ax, cbar

    def _check_prop_version(
        self,
        data_prop: DataProp,
        data_version: PlotDataVersion
        ) -> tuple[DataProp, PlotDataVersion]:
        '''Check validity of `data_prop` and `data_version`.'''

        data_prop = str(data_prop)
        available_props = get_args(DataProp)

        if data_prop not in available_props:
            raise ValueError(f'data_prop must be one of {available_props}')

        data_version = str(data_version)
        available_versions = get_args(PlotDataVersion)

        if data_version not in available_versions:
            raise ValueError(f'data_version must be one of {available_versions}')

        if data_version in ['raw', 'converted'] and len(self._raw_df) == 0:
            raise MissingDataError(
                f'raw_df is empty. Please re-initialize {self.__class__.__name__}() '
                'with a file or DataFrame containing magnetization data.'
            )

        if data_version in ['processed', 'compare'] and len(self._processed_df) == 0:
            raise MissingDataError(
                f'argument {data_version} is invalid because processed_df is empty. '
                'Please run process_data() before plotting.'
            )

        if (data_version in ['raw', 'converted', 'compare'] and
            data_prop in ['dM_dT', 'Delta_SM'] and
            len(self._processed_df) == 0):
            raise MissingDataError(
                f'{data_prop} {data_version} is not computed until process_data() is run.'
                'Please run process_data() first.'
            )

        return data_prop, data_version

    def _grouping_plotting_cases(
        self,
        data_version: PlotDataVersion,
        data_prop: str,
        at_temps: Optional[ArrayLike],
        presets: Presets
        ) -> dict[str, Union[str, np.int64, np.float64, NDArray[np.float64], pd.DataFrame]]:
        '''
        Return the grouping and plotting parameters depending on different cases.

        Parameters for `data_version` ``'compare'`` are used for grouping the
        raw data, which will be converted to converted units, then for plotting
        this converted data using compare-specific options.
        '''

        df_col_unit_params = self._df_col_unit_params(data_version, data_prop)

        grouping_plotting_params = pt.grouping_plotting_params(
            data_version,
            at_temps,
            presets,
            df_col_unit_params['group_df']
        )

        return {**df_col_unit_params, **grouping_plotting_params}

    def _df_col_unit_params(
        self,
        data_version: PlotDataVersion,
        data_prop: DataProp
        ) -> dict[str, Union[str, pd.DataFrame]]:
        '''
        Return grouping df, grouping column, and unit labels to use based
        on `data_version` and `data_prop`.
        '''

        r_cols = self._RAW_DF_COLUMNS
        c_units = self._CONVERTED_DF_UNITS
        p_cols = self._PROCESSED_DF_COLUMNS
        p_units = self._PROCESSED_DF_UNITS

        # determine df, columns, and unit labels to use
        if data_version in ['raw', 'converted', 'compare']:
            group_df = self.raw_df
            T_col = r_cols['T']
            H_col = r_cols['H']
            prop_col = r_cols[data_prop]
            if data_version == 'raw':
                units = self._get_all_raw_data_units()
            elif data_version == 'converted':
                units = c_units
            else:
                units = p_units # compare should still label with processed units
        else:
            group_df = self.processed_df
            T_col = p_cols['T']
            H_col = p_cols['H']
            prop_col = p_cols[data_prop]
            units = p_units

        T_units, H_units, prop_units = pu.get_latex_units(self._ureg, data_prop, units)

        return {
            'group_df': group_df,
            'T_col': T_col,
            'H_col': H_col,
            'prop_col': prop_col,
            'T_units': T_units,
            'H_units': H_units,
            'prop_units': prop_units
        }

    def _convert_grouped(
        self,
        grouped_data: DataFrameGroupBy,
        at_temps: Optional[ArrayLike]
        ) -> DataFrameGroupBy:
        '''Given a :class:`DataFrameGroupBy` of raw data, convert to converted units.'''

        r_units = self._RAW_DF_DEFAULT_UNITS
        c_units = self._CONVERTED_DF_UNITS

        conv_name = 'H' if at_temps is None else 'T'

        group_dict = grouped_data.indices
        df = grouped_data.apply(self._apply_convert_grouped)
        converted_groups = np.zeros(len(df))

        for key in group_dict:
            converted_key = (key * self._ureg(r_units[conv_name])).to(c_units[conv_name]).magnitude
            for i in group_dict[key]:
                converted_groups[i] = converted_key

        return df.groupby(converted_groups)

    def _apply_convert_grouped(self, da: pd.DataFrame) -> pd.DataFrame:
        '''For use in :meth:`DataFrameGroupBy.apply`.'''

        r_cols = self._RAW_DF_COLUMNS
        r_units = self._RAW_DF_DEFAULT_UNITS
        c_units = self._CONVERTED_DF_UNITS

        conv_df = pu.df_from_units(r_cols.values(), r_units.values(), da.values)

        with self._ureg.context('M_per_mass', 'dM/dT', 'Delta_SM'):
            pu.convert_df(
                conv_df, conv_df,
                r_cols.values(), r_cols.values(),
                c_units.values()
            )

        da.iloc[:, :] = conv_df.pint.dequantify().values

        return da

    def _handle_compare_lines(
        self,
        data_prop: DataProp,
        ax: Optional[Axes],
        T_range: ArrayLike,
        H_range: ArrayLike,
        offset: float,
        at_temps: Optional[ArrayLike],
        colormap: Optional[Union[Colormap, str]],
        compare_kwargs: Optional[Union[Kwargs, Sequence[Kwargs]]],
        last_presets: Presets
        ) -> None:
        '''
        Handle the case in which `data_version` is ``'compare'``, so the
        converted data is to be plotted after the processed data.
        '''

        c_case_params = self._grouping_plotting_cases(
            'compare',
            data_prop,
            at_temps,
            last_presets
        )

        c_grouped_data = gr.group_by(
            c_case_params['group_df'],
            c_case_params['group_col'],
            groups=c_case_params['groups'],
            decimals=c_case_params['decimals'],
            max_diff=c_case_params['max_diff'],
            match_col=c_case_params['match_col'],
            match_values=c_case_params['match_values']
        )

        c_grouped_data = self._convert_grouped(c_grouped_data, at_temps)

        pt.plot_grouped_data(
            data_prop, 'compare', ax, c_grouped_data,
            c_case_params['group_col'], c_case_params['T_col'],
            c_case_params['H_col'], c_case_params['prop_col'],
            c_case_params['T_units'], c_case_params['H_units'], c_case_params['prop_units'],
            T_range, H_range, offset, at_temps, colormap,
            c_case_params['marker'], c_case_params['markersize'],
            c_case_params['linestyle'], c_case_params['legend_label'],
            c_case_params['colorbar_label'],
            False, False, compare_kwargs, None
        )

    @classmethod
    def plot_processed_map(
        cls,
        processed_df: pd.DataFrame,
        data_prop: DataProp = 'M_per_mass',
        ax: Optional[Axes] = None,
        T_range: ArrayLike = np.array([-np.inf, np.inf]),
        H_range: ArrayLike = np.array([-np.inf, np.inf]),
        T_npoints: int = 1000,
        H_npoints: int = 1000,
        interp_method: str = 'linear',
        center: Optional[bool] = None,
        contour: bool = False,
        colorbar: bool = True,
        imshow_kwargs: Optional[Kwargs] = None,
        contour_kwargs: Optional[Kwargs] = None,
        colorbar_kwargs: Optional[Kwargs] = None
        ) -> Union[Axes, tuple[Axes, Colorbar]]:
        '''
        Plot processed data from a |DataFrame| as a map.

        This class method allows already-processed data to be easily plotted so
        that raw data needn't be re-processed.

        See :meth:`plot_map` for parameters following `processed_df` and return
        values.

        Parameters
        ----------
        processed_df : DataFrame
            Processed data. Expected to have column names matching those of the
            `processed_df` attribute of an instance of |MagentroData|.
        '''

        p_cols = cls._PROCESSED_DF_COLUMNS
        p_units = cls._PROCESSED_DF_UNITS

        # check arguments
        data_prop = vd.check_processed_args(processed_df, data_prop)

        # determine units to use
        T_units, H_units, prop_units = pu.get_latex_units(
            pint_pandas.PintType.ureg,
            data_prop,
            p_units
        )

        ax, cbar = pt.plot_map(
            data_prop, 'processed',
            ax, T_range, H_range, T_npoints, H_npoints, interp_method,
            center, contour, colorbar, imshow_kwargs, contour_kwargs, colorbar_kwargs,
            processed_df, p_cols, T_units, H_units, prop_units
        )

        if cbar is None:
            return ax

        return ax, cbar

    def get_map_grid(
        self,
        data_prop: DataProp = 'M_per_mass',
        data_version: MapDataVersion = 'raw',
        T_range: ArrayLike = np.array([-np.inf, np.inf]),
        H_range: ArrayLike = np.array([-np.inf, np.inf]),
        T_npoints: int = 1000,
        H_npoints: int = 1000,
        interp_method: str = 'linear'
        ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        '''
        Return the temperature, field, and property grids used to construct maps.

        See :meth:`plot_map` for parameters.

        Returns
        -------
        T_grid, H_grid, grid : ndarray
            Grids corresponding to temperature, field, and property, respectively.
        '''

        # check arguments

        data_prop, data_version = self._check_prop_version(data_prop, data_version)

        if data_version == 'compare':
            raise ValueError('compare is not available for maps.')

        # determine df to use
        df, cols, _ = self._df_for_map(data_version)

        _, _, _, _, _, _, T_grid, H_grid, prop_grid = pt.bounds_and_grid(
            data_prop, T_range, H_range, T_npoints, H_npoints, interp_method, df, cols
        )

        return T_grid, H_grid, prop_grid

    def plot_map(
        self,
        data_prop: DataProp = 'M_per_mass',
        data_version: MapDataVersion = 'raw',
        ax: Optional[Axes] = None,
        T_range: ArrayLike = np.array([-np.inf, np.inf]),
        H_range: ArrayLike = np.array([-np.inf, np.inf]),
        T_npoints: int = 1000,
        H_npoints: int = 1000,
        interp_method: GriddataMethod = 'linear',
        center: Optional[bool] = None,
        contour: bool = False,
        colorbar: bool = True,
        imshow_kwargs: Optional[Kwargs] = None,
        contour_kwargs: Optional[Kwargs] = None,
        colorbar_kwargs: Optional[Kwargs] = None
        ) -> Union[Axes, tuple[Axes, Colorbar]]:
        '''
        Plot the moment per mass, derivative with respect to temperature, or
        entropy as a map.

        All parameter units should correspond to those of the data specified by
        `data_version`.

        .. note::

            Different default colormaps are used depending on `center`.
            The colormap can be specified manually in `imshow_kwargs`.
            For example, ``imshow_kwargs = {'cmap': 'RdBu_r'}``.

        Parameters
        ----------
        data_prop : {'M_per_mass', 'dM_dT', 'Delta_SM'}, default 'M_per_mass'
            The property to plot.
        data_version : {'raw', 'converted', 'processed'}, default 'raw'
            The version of the data to plot.
            (``'compare'`` is not available for maps.)
        ax : Axes, optional
            |Axes| on which to plot. If ``None``, new |Axes| will be
            created from the current |Figure|.
        T_range, H_range : (2,) array_like
            Temperature and magnetic field strength ranges to display.
        T_npoints, H_npoints : int, default 1000
            Number of points to use for grid interpolation
            in the horizontal (T) and vertical (H) directions.
        interp_method : {'linear', 'nearest', 'cubic'}, default 'linear'
            Map grid interpolation method. See
            :func:`scipy.interpolate.griddata`'s ``method`` parameter.
            The ``'cubic'`` method may give a smoother result,
            but it is recommended to start with ``'linear'`` interpolation,
            as artifacts can occasionally occur in the output
            when using higher-order interpolation.
        center : bool, optional
            If ``True``, center the pixel values around zero, setting values
            beyond the central range to the values at the boundaries of the
            range. This is helpful for ignoring extreme values. ``None``
            defaults to ``False`` when `data_prop` is ``'M_per_mass'`` and
            ``True`` otherwise.
        contour : bool, default False
            If ``True``, add contours to the plot with |Axes.contour|.
        colorbar : bool, default True
            If ``True``, add a continuous color bar to the |Figure| containing
            `ax` with |Figure.colorbar|.
        imshow_kwargs : dict, optional
            Keyword arguments for |Axes.imshow|.
        contour_kwargs : dict, optional
            Keyword arguments for |Axes.contour|.
        colorbar_kwargs : dict, optional
            Keyword arguments for |Figure.colorbar|, excluding ``mappable``.

        Returns
        -------
        ax : Axes
            If `colorbar` is ``False``, the |Axes| on which the data is plotted.
        ax, cbar : Axes, Colorbar
            If `colorbar` is ``True``, the |Axes| on which the data is plotted
            and the |Colorbar|.
        '''

        # check arguments

        data_prop, data_version = self._check_prop_version(data_prop, data_version)

        if data_version == 'compare':
            raise ValueError('compare is not available for maps.')

        # determine df and units to use
        df, cols, units = self._df_for_map(data_version)
        T_units, H_units, prop_units = pu.get_latex_units(self._ureg, data_prop, units)

        ax, cbar = pt.plot_map(
            data_prop, data_version,
            ax, T_range, H_range, T_npoints, H_npoints, interp_method,
            center, contour, colorbar, imshow_kwargs, contour_kwargs, colorbar_kwargs,
            df, cols, T_units, H_units, prop_units
        )

        if cbar is None:
            return ax

        return ax, cbar

    def _df_for_map(
        self,
        data_version: MapDataVersion
        ) -> tuple[pd.DataFrame, ColumnDataDict, ColumnDataDict]:
        '''Return the |DataFrame|, columns, and units to use for a map.'''

        if data_version == 'raw':
            df = self.raw_df
            cols = self._RAW_DF_COLUMNS
            units = self._get_all_raw_data_units()
        elif data_version in ['converted', 'compare']:
            df = self.converted_df
            cols = self._CONVERTED_DF_COLUMNS
            units = self._CONVERTED_DF_UNITS
        else:
            df = self.processed_df
            cols = self._PROCESSED_DF_COLUMNS
            units = self._PROCESSED_DF_UNITS

        return df, cols, units

    @classmethod
    def plot_processed(
        cls,
        plot_type: PlotType,
        **kwargs: Any
        ) -> Union[Axes, tuple[Axes, Colorbar]]:
        '''
        Plot processed property as lines or as a map.

        See :meth:`plot_processed_lines` or :meth:`plot_processed_map`
        for parameters and return values.

        Parameters
        ----------
        plot_type : {'lines', 'map'}
            Plot lines or map.
        **kwargs : dict, optional
            Passed to :meth:`plot_processed_lines` or
            :meth:`plot_processed_map`, depending on `plot_type`.
        '''

        plot_type = str(plot_type)
        available_plot_types = get_args(PlotType)

        if plot_type not in available_plot_types:
            raise ValueError(f'plot_type must be one of {available_plot_types}')

        if plot_type == 'lines':
            return cls.plot_processed_lines(**kwargs)

        return cls.plot_processed_map(**kwargs)

    def plot(self, plot_type: PlotType, **kwargs: Any) -> Union[Axes, tuple[Axes, Colorbar]]:
        '''
        Plot property as lines or as a map.

        See :meth:`plot_lines` or :meth:`plot_map` for parameters and return values.

        Parameters
        ----------
        plot_type : {'lines', 'map'}
            Plot lines or map.
        **kwargs : dict, optional
            Passed to :meth:`plot_lines` or :meth:`plot_map`,
            depending on `plot_type`.
        '''

        plot_type = str(plot_type)
        available_plot_types = get_args(PlotType)

        if plot_type not in available_plot_types:
            raise ValueError(f'plot_type must be one of {available_plot_types}')

        if plot_type == 'lines':
            return self.plot_lines(**kwargs)

        return self.plot_map(**kwargs)

    def plot_all(self) -> tuple[Axes, ...]:
        '''
        Plot all combinations of `data_prop` and `data_version` for both
        line plots and maps with default settings.

        Line plots grouped by temperature are also plotted, with five
        evenly-spaced temperature groups.

        Each returned |Axes| gets its own |Figure|. All |Figure|\s are plotted
        immediately if run in a notebook, so this is a "quick-and-dirty" way
        to view every plot after initial processing.

        .. tip::

            If using a notebook, be sure to put a semicolon (``;``) after this
            method to suppress nasty-looking text output!

        Returns
        -------
        tuple[Axes, ...]
            The |Axes|, one for each plot.
        '''

        if len(self._raw_df) == 0:
            raise MissingDataError(
                f'raw_df is empty. Please re-initialize {self.__class__.__name__} '
                'with a file or DataFrame containing magnetization data.'
            )

        if len(self._processed_df) == 0:
            raise MissingDataError(
                'processed_df is empty. Please run process_data() before plotting.'
            )

        available_props = get_args(DataProp)

        axes_list = []

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='More than 20 figures have been opened.')
            self._plot_all_lines(available_props, axes_list)
            self._plot_all_maps(available_props, axes_list)

        return tuple(axes_list)

    def _plot_all_lines(
        self,
        available_props: tuple[str, ...],
        axes_list: list[Axes]
        ) -> None:
        '''Plot all line plots.'''

        available_versions = get_args(PlotDataVersion)

        # grouped by field
        for prop in available_props:
            for version in available_versions:
                ax = plt.figure().subplots()
                self.plot_lines(prop, version, ax)
                axes_list.append(ax)

        at_temps_dict = {
            'raw': self._five_temps(self.raw_df, self._RAW_DF_COLUMNS),
            'converted': self._five_temps(self.converted_df, self._CONVERTED_DF_COLUMNS),
            'processed': self._five_temps(self.processed_df, self._PROCESSED_DF_COLUMNS)
        }

        at_temps_dict['compare'] = at_temps_dict['processed']

        # grouped by temperature
        for prop in available_props:
            for version in available_versions:
                ax = plt.figure().subplots()
                self.plot_lines(prop, version, ax, at_temps=at_temps_dict[version], colorbar=True)
                axes_list.append(ax)

    @staticmethod
    def _five_temps(df: pd.DataFrame, cols: ColumnDataDict) -> NDArray[np.float64]:
        '''
        Return :class:`ndarray <numpy.ndarray>` of five evenly-spaced
        temperatures in `df` ranging from the minimum plus one tenth of the
        total range to the maximum minus one tenth of the total range.
        '''

        tenth = (df[cols['T']].max() - df[cols['T']].min()) / 10.
        return np.linspace(df[cols['T']].min() + tenth, df[cols['T']].max() - tenth, 5)

    def _plot_all_maps(
        self,
        available_props: tuple[str, ...],
        axes_list: list[Axes]
        ) -> None:
        '''Plot all maps.'''

        available_versions = get_args(MapDataVersion)

        for prop in available_props:
            for version in available_versions:
                ax = plt.figure().subplots()
                self.plot_map(prop, version, ax)
                axes_list.append(ax)
