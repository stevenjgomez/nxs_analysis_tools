'''
Contains functions used for simulation, smoothing, interpolation, differentiation, and integration.
'''

from typing import Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
import pandas as pd
from pint import Quantity
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import griddata, RegularGridInterpolator
from numba import njit

from .typedefs import Kwargs, MatchErrStr
from ._classvars import RAW_DF_COLUMNS, RAW_DF_DEFAULT_UNITS, PROCESSED_DF_COLUMNS
from . import _pint_utils as pu
from . import _smoothing

@njit(cache=True)
def sim_func(
    t: NDArray[np.float64],
    h: NDArray[np.float64],
    m_max: np.float64,
    slope: np.float64,
    bump_height: np.float64
    ) -> NDArray[np.float64]:
    '''
    Compute simulated data.

    Model function is a decreasing logistic function with a tiny Gaussian bump.

    Decreasing logistic function: L / (1 + e^[-k(t-t0)])

    Gaussian: A e^(-(t-mu)^2 / a^2)
    '''

    # choose minimum moment to be 1/len(h) of m_max
    m_maxes = np.linspace(m_max / len(h), m_max, len(h))

    # Choose x0 to be two-thirds of the temperature range
    t_range = t.max() - t.min()
    t0 = t.min() + (2./3.) * t_range

    # derivative at t0 is log_max * k / 4, so scale k inversely with t_range
    k = -4. / t_range * slope

    gaussian_amp = m_max * bump_height

    # choose the Gaussian spread to be about one-fifth of the temperature range
    gaussian_spr = t_range / 5.

    # make the Gaussian center vary with field
    mus = t.min() + np.linspace(t_range * (2./3.), t_range / 3., len(h))

    m_grid = np.zeros((len(h), len(t)), dtype=np.float64)

    for i in range(len(h)):
        m_grid[i, :] = (
            m_maxes[i] / (1. + np.exp(-k * (t-t0)))
            + gaussian_amp * np.exp(-((t-mus[i]) / gaussian_spr)**2)
        )

    return m_grid

def smooth_diff_int_processed(
    used_fields: NDArray[np.float64],
    df_arrays: tuple[NDArray[np.float64], ...],
    n: np.int64,
    tr: NDArray[np.float64],
    d: np.int64,
    lms: NDArray[np.float64],
    lm_g: np.float64,
    we: bool,
    me: Union[bool, ArrayLike, MatchErrStr],
    mk: Kwargs,
    az: bool,
    current_p_units: dict[str, str],
    sample_mass: Quantity
    ) -> pd.DataFrame:
    '''Perform the smoothing, differentiation, and integration on the processed data.'''

    r_units = RAW_DF_DEFAULT_UNITS
    p_cols = PROCESSED_DF_COLUMNS

    # smooth in the supplied units for T and M, respectively
    smooth_output = create_output_arr(used_fields, n, np.linspace(tr[0], tr[1], n))
    _smoothing.smooth_processed(used_fields, df_arrays, smooth_output, d, lms, lm_g, we, me, mk)

    # fill processed df with the smoothed data in the supplied units
    smooth_output = smooth_output.reshape((len(used_fields)+1) * n, 8)
    processed_df = pu.df_from_units(p_cols.values(), current_p_units.values(), smooth_output)

    # convert T, H, M, and M_err to the default raw data units
    pu.convert_processed_raw(
        processed_df,
        (p_cols['T'], p_cols['H'], p_cols['M'], p_cols['M_err']),
        (r_units['T'], r_units['H'], r_units['M'], r_units['M_err'])
    )

    current_p_units['T'] = r_units['T']
    current_p_units['H'] = r_units['H']
    current_p_units['M'] = r_units['M']
    current_p_units['M_err'] = r_units['M_err']

    # compute per-mass columns
    pu.create_per_mass_cols(processed_df, p_cols, r_units, sample_mass)

    # perform the differentiation and integration in the default raw data units
    diff_int_output = processed_df.pint.dequantify().values.reshape(len(used_fields)+1, n, 8)
    diff_int_processed(diff_int_output)
    diff_int_output = diff_int_output.reshape((len(used_fields)+1) * n, 8)

    # remove zeros at beginning if add_zeros is False
    if not az:
        diff_int_output = diff_int_output[n:, :]

    processed_df = pu.df_from_units(p_cols.values(), current_p_units.values(), diff_int_output)

    # convert dM_dT and Delta_SM, which are in units directly constructed from the
    # defaults for T, H, and M, to the default (standard cgs) raw data units.
    pu.convert_processed_raw(
        processed_df,
        (p_cols['dM_dT'], p_cols['Delta_SM']),
        (r_units['dM_dT'], r_units['Delta_SM'])
    )

    return processed_df

@njit(parallel=True, cache=True)
def create_output_arr(
    used_fields: NDArray[np.float64],
    n: np.int64,
    t_grid: NDArray[np.float64]
    ) -> NDArray[np.float64]:
    '''
    Return an output array to hold the processed data.

    Like a stack of df values: (0 + fields) x npoints x columns

    The prepended field row holds data for zero field, where moment is zero
    (for integration).

    The first depth layer, output[:, :, 0], holds the temperature grid for
    each field.

    The second depth layer excluding zero field, output[1:, :, 1], holds the
    field values.

    M_err and M_per_mass_err are NaN.
    '''

    nfields = len(used_fields)

    output = np.zeros((nfields + 1, n, 8))
    output[:, :, 0] = t_grid.repeat(nfields + 1).reshape((n, nfields + 1)).T
    output[1:, :, 1] = used_fields.repeat(n).reshape((nfields, n))
    output[:, :, 3] = np.nan
    output[:, :, 5] = np.nan

    return output

def diff_int_processed(output: NDArray[np.float64]) -> None:
    '''
    Perform the differentiation and integration for the processed data.

    Data indices corresponding to columns are
    T: 0, H: 1, M_per_mass: 4, dM_dT: 6, Delta_SM: 7

    output[0,:,0] is a 1D array of the temperature grid

    output[:,0,1] is a 1D array of the field values
    '''

    # gradient along the npoints temperatures (axis=1)
    output[1:, :, 6] = np.gradient(output[1:,:,4], output[0,:,0], axis=1, edge_order=2)

    # integrate derivatives along the fields (axis=0)
    output[:, :, 7] = cumulative_trapezoid(output[:,:,6], x=output[:,0,1], axis=0, initial=0.0)

def interp_diff_int_raw(
    raw_df: pd.DataFrame,
    n: np.int64,
    temp_diff: np.float64,
    fields: NDArray[np.float64],
    t_h_raw_conv_values: NDArray[np.float64]
    ) -> None:
    '''Perform the interpolation, differentiation, and integration on the raw data.'''

    t_lin, h_lin, m_per_mass_grid = interp_setup(raw_df, n, temp_diff, fields, t_h_raw_conv_values)

    dm_dt_call, delta_sm_call = interp_callables(t_lin, h_lin, m_per_mass_grid)

    raw_df[RAW_DF_COLUMNS['dM_dT']] = dm_dt_call(t_h_raw_conv_values)
    raw_df[RAW_DF_COLUMNS['Delta_SM']] = delta_sm_call(t_h_raw_conv_values)

def interp_setup(
    raw_df: pd.DataFrame,
    n: np.int64,
    temp_diff: np.float64,
    fields: NDArray[np.float64],
    t_h_raw_conv_values: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    '''Return T and H linspaces, and M_per_mass grid.'''

    t_min, t_max = t_h_raw_conv_values[:, 0].min(), t_h_raw_conv_values[:, 0].max()
    num_temps = np.int64(np.around(n * (t_max - t_min) / temp_diff))

    t_lin = np.linspace(t_min, t_max, num_temps)
    h_lin = np.concatenate(([0.0], fields, [t_h_raw_conv_values[:, 1].max()]))

    t_grid, h_grid = np.meshgrid(t_lin, h_lin, indexing='ij')

    # interpolated moment grid
    m_per_mass_grid = griddata(
        (t_h_raw_conv_values[:, 0], t_h_raw_conv_values[:, 1]),
        raw_df[RAW_DF_COLUMNS['M_per_mass']],
        (t_grid, h_grid),
        method='linear'
    )

    # same, but with nearest method instead of linear
    m_per_mass_nearest = griddata(
        (t_h_raw_conv_values[:, 0], t_h_raw_conv_values[:, 1]),
        raw_df[RAW_DF_COLUMNS['M_per_mass']],
        (t_grid, h_grid),
        method='nearest'
    )

    # fill any missing values in the "linear" grid with those from the "nearest" grid
    m_per_mass_nan_idx = np.isnan(m_per_mass_grid)
    m_per_mass_grid[m_per_mass_nan_idx] = m_per_mass_nearest[m_per_mass_nan_idx]

    return t_lin, h_lin, m_per_mass_grid

def interp_callables(
    t_lin: NDArray[np.float64],
    h_lin: NDArray[np.float64],
    m_per_mass_grid: NDArray[np.float64]
    ) -> tuple[RegularGridInterpolator, RegularGridInterpolator]:
    '''
    Compute dM_dT and Delta_SM grids and return callable
    ``RegularGridInterpolator``s for interpolating dM_dT and Delta_SM,
    respectively, back to original T and H points.
    '''

    # compute dM_dT and Delta_SM from the grid
    dm_dt_grid = np.gradient(m_per_mass_grid, t_lin, axis=0, edge_order=2)
    delta_sm_grid = cumulative_trapezoid(dm_dt_grid, x = h_lin, axis=1, initial=0.0)

    # interpolate dM_dT back to the original T, H points
    dm_dt_call = RegularGridInterpolator(
        (t_lin, h_lin),
        dm_dt_grid,
        method='nearest',
        bounds_error=False
    )

    # interpolate Delta_SM back to the original T, H points
    delta_sm_call = RegularGridInterpolator(
        (t_lin, h_lin),
        delta_sm_grid,
        method='nearest',
        bounds_error=False
    )

    return dm_dt_call, delta_sm_call

def bootstrap(
    used_fields: NDArray[np.float64],
    df_arrays: tuple[NDArray[np.float64], ...],
    n: np.int64,
    tr: NDArray[np.float64],
    d: np.int64,
    lms: NDArray[np.float64],
    we: bool,
    az: bool,
    current_p_units: dict[str, str],
    sample_mass: Quantity,
    n_bootstrap: np.int64,
    rng: np.random.Generator
    ) -> pd.DataFrame:
    '''Perform the bootstrap calculation on the processed data.'''

    r_units = RAW_DF_DEFAULT_UNITS
    p_cols = PROCESSED_DF_COLUMNS

    # bootstrap in the supplied units for T and M, respectively
    bootstrap_output = create_output_arr(used_fields, n, np.linspace(tr[0], tr[1], n))
    _smoothing.bootstrap(
        used_fields, df_arrays, bootstrap_output,
        d, lms, we, n_bootstrap, rng
    )

    # fill processed df with the smoothed data in the supplied units
    bootstrap_output = bootstrap_output.reshape((len(used_fields)+1) * n, 8)

    bootstrap_df = pu.df_from_units(p_cols.values(), current_p_units.values(), bootstrap_output)

    # convert T, H, M, and M_err to the default raw data units
    pu.convert_processed_raw(
        bootstrap_df,
        (p_cols['T'], p_cols['H'], p_cols['M'], p_cols['M_err']),
        (r_units['T'], r_units['H'], r_units['M'], r_units['M_err'])
    )

    current_p_units['T'] = r_units['T']
    current_p_units['H'] = r_units['H']
    current_p_units['M'] = r_units['M']
    current_p_units['M_err'] = r_units['M_err']

    # compute per-mass columns
    pu.create_per_mass_cols(bootstrap_df, p_cols, r_units, sample_mass)

    bootstrap_output = bootstrap_df.pint.dequantify().values

    # remove zeros at beginning if add_zeros is False
    if not az:
        bootstrap_output = bootstrap_output[n:, :]

    bootstrap_df = pu.df_from_units(p_cols.values(), current_p_units.values(), bootstrap_output)

    return bootstrap_df
