'''Contains convenience functions for working with Pint.'''

from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from pint import UnitRegistry, Quantity

from .typedefs import ColumnDataDict, DataProp
from ._classvars import RAW_DF_COLUMNS, RAW_DF_DEFAULT_UNITS

def pmag(pint_series: pd.Series) -> NDArray[np.float64]:
    '''Return the underlying float ``ndarray`` of a |Series| containing a ``PintArray``.'''
    return pint_series.values.quantity.magnitude

def punits(pint_series: pd.Series) -> str:
    '''Return the short, pretty-printed units ``str`` of a |Series| containing a ``PintArray``.'''
    return f'{pint_series.values.quantity.units:~P}'

def df_from_units(
    col_names: Sequence[str],
    units: Sequence[str],
    data: Optional[NDArray[np.float64]] = None
    ) -> pd.DataFrame:
    '''Construct a |DataFrame| given column names and units, optionally with data.'''

    if data is not None:
        return pd.DataFrame(
            data={
                col: pd.Series(data[:, i], dtype=f'pint[{unit}]')
                for col, unit, i in zip(col_names, units, range(len(col_names)))
            }
        )

    return pd.DataFrame(
        data={col: pd.Series([], dtype=f'pint[{unit}]') for col, unit in zip(col_names, units)}
    )

def create_per_mass_cols(
    df: pd.DataFrame,
    cols: dict[str, str],
    units: dict[str, str],
    sample_mass: Quantity
    ) -> None:
    '''Create per-mass columns in `df`.'''

    convert_series(
        df.loc[:, cols['M']] / sample_mass,
        df,
        cols['M_per_mass'],
        units['M_per_mass']
    )

    convert_series(
        df.loc[:, cols['M_err']] / sample_mass,
        df,
        cols['M_per_mass_err'],
        units['M_per_mass_err']
    )

def convert_df(
    initial_df: pd.DataFrame,
    final_df: pd.DataFrame,
    i_names: Sequence[str],
    f_names: Sequence[str],
    f_unit_strs: Sequence[str]
    ) -> None:
    '''
    Fill columns in `final_df` (`f_names`) with data converted from
    `initial_df` columns (`i_names`).

    Conversion is from units contained in `initial_df`
    to units given by `f_unit_strs`.
    '''

    for i, f, units in zip(i_names, f_names, f_unit_strs):
        convert_series(initial_df.loc[:, i], final_df, f, units)

def convert_series(a_series: pd.Series, b_df: pd.DataFrame, b_col: str, units: str) -> None:
    '''Fill `b_col` in `b_df` with `a_series` converted to `units`.'''
    b_df.loc[:, b_col] = a_series.pint.to(units)

def convert_processed_raw(
    processed_df: pd.DataFrame,
    cols_to_convert: Sequence[str],
    raw_units: Sequence[str]
    ) -> None:
    '''Convert `cols_to_convert` to `raw_units` in `processed_df` in-place.'''

    convert_df(
        processed_df, processed_df,
        cols_to_convert, cols_to_convert,
        raw_units
    )

def get_raw_conv_t_h(df) -> NDArray[np.float64]:
    '''Convert T and H to default raw data units and return values as 2D ``ndarray``.'''

    r_cols, r_units = RAW_DF_COLUMNS, RAW_DF_DEFAULT_UNITS

    t_h_cols = [r_cols['T'], r_cols['H']]
    t_h_units = [r_units['T'], r_units['H']]

    t_h_raw_converted_df = df_from_units(t_h_cols, t_h_units)
    convert_df(df, t_h_raw_converted_df, t_h_cols, t_h_cols, t_h_units)

    return t_h_raw_converted_df.pint.dequantify().values

def get_latex_units(
    ureg: UnitRegistry,
    data_prop: DataProp,
    units: ColumnDataDict) -> tuple[str, str, str]:
    '''Return LaTeX strings for `units`.'''

    T_units = f'{ureg(units["T"]).units:~L}'
    H_units = f'{ureg(units["H"]).units:~L}'
    prop_units = f'{ureg(units[data_prop]).units:~L}'

    return T_units, H_units, prop_units
