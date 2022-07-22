'''Contains functions used for argument validation.'''

from typing import Union, Optional, Sequence, Any, get_args
from copy import deepcopy

import numpy as np
from numpy.typing import ArrayLike, NDArray
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.pyplot import Axes

from .errors import MissingDataError
from .typedefs import Kwargs, MatchErrStr, DataProp, Presets
from ._classvars import PROCESSED_DF_COLUMNS

def use_expected_columns(prepped_df: pd.DataFrame, expected_cols: pd.Series) -> pd.DataFrame:
    '''
    Confirm that `expected_cols` exist in `prepped_df`,
    drop unneeded columns in-place, reorder columns.
    '''

    missing_cols = list(expected_cols[~expected_cols.isin(prepped_df.columns)])

    if len(missing_cols) > 0:
        raise ValueError(f'Data is missing required columns: {missing_cols}')

    prepped_df.drop(
        columns=[col for col in prepped_df.columns if col not in list(expected_cols)],
        inplace=True
    )

    prepped_df = prepped_df.reindex(columns=expected_cols)

    used_series = [(col, isinstance(prepped_df.loc[:,col], pd.Series)) for col in expected_cols]
    invalid_names = [s[0] for s in used_series if not s[1]]

    if len(invalid_names) > 0:
        raise ValueError(
            'Supplied column names do not access Series (might be accessing DataFrames '
            f'via hierarchical indexing): {invalid_names}'
        )

    return prepped_df

def check_pos_int(to_check: Any, name: str) -> np.int64:
    '''
    Take input and attempt to return a positive integer.

    Raises an error if this fails.
    '''

    to_check = np.int64(to_check)

    if to_check <= 0:
        raise ValueError(f'{name} must be a positive integer.')

    return to_check

def check_range(to_check: Any, name: str) -> NDArray[np.float64]:
    '''
    Take input and attempt to return a sorted ``ndarray`` of length 2.

    Raises an error if this fails.
    '''

    message = f'{name} must be have length 2.'

    try:
        to_check = np.sort(np.array(to_check, dtype=np.float64, ndmin=1).ravel())
    except TypeError as e:
        raise TypeError(message) from e
    except ValueError as e:
        raise ValueError(message) from e
    if len(to_check) != 2:
        raise ValueError(message)

    return to_check

def check_presets(current_presets: Presets, **kwargs: Any) -> Presets:
    '''
    Check validity of preset parameters and return ``dict`` of `presets`.

    Raises an error if any preset is invalid.
    '''

    n = kwargs.get('npoints')
    tr = kwargs.get('temp_range')
    fds = kwargs.get('fields')
    dms = kwargs.get('decimals')
    md = kwargs.get('max_diff')
    msl = kwargs.get('min_sweep_len')
    d = kwargs.get('d_order')
    lms = kwargs.get('lmbds')
    lm_g = kwargs.get('lmbd_guess')
    we = kwargs.get('weight_err')
    me = kwargs.get('match_err')
    mk = kwargs.get('min_kwargs')
    az = kwargs.get('add_zeros')

    n = check_pos_int(n, 'npoints') if n is not None else current_presets['npoints']
    if n < 2:
        raise ValueError('npoints must be at least 2.')

    tr = (
        check_range(tr, 'temp_range') if tr is not None
        else current_presets['temp_range']
    )

    if fds is not None:
        fds_message = 'fields must be array_like and numeric.'
        try:
            fds = np.array(fds, dtype=np.float64, ndmin=1).ravel()
        except TypeError as e:
            raise TypeError(fds_message) from e
        except ValueError as e:
            raise ValueError(fds_message) from e
    else:
        fds = current_presets['fields']

    dms = np.int64(dms) if dms is not None else current_presets['decimals']

    md = abs(np.float64(md)) if md is not None else current_presets['max_diff']

    msl = (
        check_pos_int(msl, 'min_sweep_len') if msl is not None
        else current_presets['min_sweep_len']
    )

    we = bool(we) if we is not None else current_presets['weight_err']

    d = check_pos_int(d, 'npoints') if d is not None else current_presets['d_order']

    if lms is not None:
        lms_message = 'lmbds must be array_like and numeric.'
        try:
            lms = abs(np.array(lms, dtype=np.float64, ndmin=1).ravel())
        except TypeError as e:
            raise TypeError(lms_message) from e
        except ValueError as e:
            raise ValueError(lms_message) from e
        if len(lms) == 0:
            lms = np.array([np.nan])
    else:
        lms = current_presets['lmbds']

    lm_g = abs(np.float64(lm_g)) if lm_g is not None else current_presets['lmbd_guess']

    if me is not None:
        if not isinstance(me, bool) and me not in get_args(MatchErrStr):
            me_message = 'match_err must be array_like and numeric.'
            try:
                me = abs(np.array(me, dtype=np.float64, ndmin=1).ravel())
            except TypeError as e:
                raise TypeError(me_message) from e
            except ValueError as e:
                raise ValueError(me_message) from e
    else:
        me = current_presets['match_err']

    mk = dict(mk) if mk is not None else current_presets['min_kwargs']

    az = bool(az) if az is not None else current_presets['add_zeros']

    return deepcopy({
        'npoints': n,
        'temp_range': tr,
        'fields': fds,
        'decimals': dms,
        'max_diff': md,
        'min_sweep_len': msl,
        'd_order': d,
        'lmbds': lms,
        'lmbd_guess': lm_g,
        'weight_err': we,
        'match_err': me,
        'min_kwargs': mk,
        'add_zeros': az
    })

def verify_df(df: pd.DataFrame, cols: Sequence[str], name: str = 'df') -> None:
    '''
    Verify that `df` is a |DataFrame| and has the required `cols`.

    Parameters
    ----------
    df : DataFrame
        |DataFrame| to verify.
    cols : iterable
        Column names to check for.
    name : str
        Name of |DataFrame| for error messages.

    Raises
    ------
    TypeError
        If `df` is not an instance of |DataFrame|.
    ValueError
        If `df` does not contain every column in `cols`.
    '''

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f'{name} must be a DataFrame.')

    cols = pd.Series(cols)

    missing_cols = list(cols[~cols.isin(df.columns)])

    if len(missing_cols) > 0:
        raise ValueError(f'{name} is missing required columns: {missing_cols}')

def check_axes(ax: Axes) -> Axes:
    '''Verify and return `ax`.'''

    if ax is None:
        ax = plt.axes()
    elif not isinstance(ax, Axes):
        raise TypeError('ax must be an instance of Axes.')

    return ax

def cast_strs(*args: Any, optional: bool = False) -> tuple[Optional[str], ...]:
    '''Return ``str``s of `args`.'''
    return tuple(((arg if optional and arg is None else str(arg)) for arg in args))

def check_processed_args(
    processed_df: pd.DataFrame,
    data_prop: DataProp
    ) -> DataProp:
    '''
    Check arguments to ``magentropy.MagentroData.plot_processed_lines`` and
    ``magentropy.MagentroData.plot_processed_map``.
    '''

    verify_df(processed_df, PROCESSED_DF_COLUMNS.values(), 'processed_df')

    data_prop = str(data_prop)
    available_props = get_args(DataProp)

    if data_prop not in available_props:
        raise ValueError(f'data_prop must be one of {available_props}')

    if len(processed_df) == 0:
        raise MissingDataError('processed_df is empty.')

    return data_prop

def check_plot_kwargs(
    plot_kwargs: Optional[Union[Kwargs, Sequence[Kwargs]]],
    length: int
    ) -> list[Kwargs]:
    '''
    Make `plot_kwargs` into a ``list`` of ``dict``s with length equal to
    the number of lines to be plotted.

    If necessary, `plot_kwargs` is duplicated, empty ``dicts``s are added,
    or the sequence is truncated.
    '''

    if plot_kwargs is None:
        plot_kwargs = [{} for i in range(length)]
    else:
        if isinstance(plot_kwargs, dict):
            plot_kwargs = [dict(plot_kwargs) for i in range(length)]
        else:
            plot_kwargs = [dict(kwds) for kwds in plot_kwargs]
            if len(plot_kwargs) >= length:
                plot_kwargs = plot_kwargs[:length]
            else:
                plot_kwargs.extend([
                    {} for i in range(length - len(plot_kwargs))
                ])

    return plot_kwargs

def check_kwargs(kwargs: Optional[Kwargs]) -> Kwargs:
    '''Check `kwargs` ``dict``.'''
    return {} if kwargs is None else dict(kwargs)

def check_grid_args(
    T_range: ArrayLike,
    H_range: ArrayLike,
    T_npoints: int,
    H_npoints: int,
    T_values: NDArray[np.float64],
    H_values: NDArray[np.float64]
    ) -> tuple[
        NDArray[np.float64], NDArray[np.float64],
        np.int64, np.int64,
        np.float64, np.float64, np.float64, np.float64
    ]:
    '''Check arguments for map grid construction.'''

    T_range = check_range(T_range, 'T_range')
    H_range = check_range(H_range, 'H_range')

    T_npoints = check_pos_int(T_npoints, 'T_npoints')
    H_npoints = check_pos_int(H_npoints, 'H_npoints')

    T_min = max(T_range[0], T_values.min())
    T_max = min(T_range[1], T_values.max())
    H_min = max(H_range[0], H_values.min())
    H_max = min(H_range[1], H_values.max())

    return T_range, H_range, T_npoints, H_npoints, T_min, T_max, H_min, H_max

def check_map_args(
    data_prop: DataProp,
    ax: Axes,
    center: Optional[bool],
    contour: bool,
    imshow_kwargs: Kwargs,
    contour_kwargs: Kwargs,
    colorbar_kwargs: Kwargs
    ) -> tuple[Axes, bool, bool, Kwargs, Kwargs, Kwargs]:
    '''Check arguments for map plotting.'''

    ax = check_axes(ax)

    if center is None:
        center = (data_prop != 'M_per_mass')
    else:
        center = bool(center)

    contour = bool(contour)

    imshow_kwargs = check_kwargs(imshow_kwargs)
    contour_kwargs = check_kwargs(contour_kwargs)
    colorbar_kwargs = check_kwargs(colorbar_kwargs)

    return ax, center, contour, imshow_kwargs, contour_kwargs, colorbar_kwargs
