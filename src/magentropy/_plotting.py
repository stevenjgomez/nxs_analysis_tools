'''Contains functions used for plotting data.'''

from typing import Union, Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
import pandas as pd
import pint_pandas
from pandas.core.groupby.generic import DataFrameGroupBy
from scipy.interpolate import griddata

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Colormap
from cycler import cycler

from .typedefs import (
    Kwargs, DataProp, PlotDataVersion, MapDataVersion, GroupingProp,
    ColumnDataDict, Presets
)
from ._classvars import RAW_DF_COLUMNS, PROCESSED_DF_COLUMNS, PROCESSED_DF_UNITS
from . import _validation as vd
from . import _pint_utils as pu
from . import _grouping as gr

MAX_DECIMALS: np.int64 = np.int64(15)

OFFSET_FMT_STR: str = ', offset by {offset}'

UNITS_FMT_STR = '$\\left({units_str}\\right)$'

PLOT_DEFAULTS = {
    'dot_marker': '.',
    'compare_marker': '.',
    'dot_linestyle': '',
    'line_marker': '',
    'line_linestyle': '-',
    'T_marker': '.',
    'T_linestyle': '-',
    'T_compare_linestyle': '--',

    'markersize': 1.0,
    'T_markersize': 2.0,
    'linewidth': 1.0,

    'H_lines_cmap': 'viridis',
    'T_lines_cmap': 'plasma',
    'sequential_map_cmap': 'magma',
    'diverging_map_cmap': 'RdBu_r',
    'map_interp': 'bicubic',
    'map_aspect': 'auto',
    'map_origin': 'lower',
    'contour_colors': 'k',
    'contour_origin': 'lower',
    'contour_linestyles': 'solid',

    'T_title': f'{{version_str}}, Grouped by Temperature $T${{offset_str}} {UNITS_FMT_STR}',
    'T_xlabel': f'$T$ {UNITS_FMT_STR}',
    'T_legend_label': '$T = {mag_str}$ ${units_str}$',

    'H_title': f'{{version_str}}, Grouped by Field Strength $H${{offset_str}} {UNITS_FMT_STR}',
    'H_xlabel': f'$H$ {UNITS_FMT_STR}',
    'H_legend_label': '$H = {mag_str}$ ${units_str}$',

    'M_per_mass_ylabel': f'$M$ {UNITS_FMT_STR}',
    'M_per_mass_raw_title': f'Raw Moment $M$ {UNITS_FMT_STR}',
    'M_per_mass_converted_title': f'Converted Moment $M$ {UNITS_FMT_STR}',
    'M_per_mass_processed_title': f'Processed Moment $M$ {UNITS_FMT_STR}',

    'dM_dT_ylabel': f'd$M$/d$T$ {UNITS_FMT_STR}',
    'dM_dT_raw_title': f'Raw Derivative d$M$/d$T$ {UNITS_FMT_STR}',
    'dM_dT_converted_title': f'Converted Derivative d$M$/d$T$ {UNITS_FMT_STR}',
    'dM_dT_processed_title': f'Processed Derivative d$M$/d$T$ {UNITS_FMT_STR}',

    'Delta_SM_ylabel': f'$\\Delta S_M$ {UNITS_FMT_STR}',
    'Delta_SM_raw_title': f'Raw Entropy $\\Delta S_M$ {UNITS_FMT_STR}',
    'Delta_SM_converted_title': f'Converted Entropy $\\Delta S_M$ {UNITS_FMT_STR}',
    'Delta_SM_processed_title': f'Processed Entropy $\\Delta S_M$ {UNITS_FMT_STR}',
}

def plot_processed_lines(
    df: pd.DataFrame,
    data_version: PlotDataVersion,
    data_prop: DataProp,
    ax: Axes,
    cols: ColumnDataDict,
    T_range: ArrayLike,
    H_range: ArrayLike,
    offset: float,
    at_temps: ArrayLike,
    colormap: Optional[Union[str, Colormap]],
    legend: bool,
    colorbar: bool,
    plot_kwargs: Optional[Union[Kwargs, Sequence[Kwargs]]],
    colorbar_kwargs: Optional[Kwargs],
    presets: Presets
    ) -> tuple[Axes, Optional[Colorbar]]:
    '''Plot already-processed data.'''

    p_units = PROCESSED_DF_UNITS

    T_units, H_units, prop_units = pu.get_latex_units(
        pint_pandas.PintType.ureg,
        data_prop,
        p_units
    )

    case_params = {
        'group_df': df,
        'T_col': cols['T'],
        'H_col': cols['H'],
        'prop_col': cols[data_prop],
        'T_units': T_units,
        'H_units': H_units,
        'prop_units': prop_units
    }

    grp_plot_params = grouping_plotting_params(
        data_version,
        at_temps,
        presets,
        df
    )

    case_params.update(grp_plot_params)

    grouped_data = gr.group_by(
        case_params['group_df'],
        case_params['group_col'],
        groups=case_params['groups'],
        decimals=case_params['decimals'],
        max_diff=case_params['max_diff'],
        match_col=case_params['match_col'],
        match_values=case_params['match_values']
    )

    ax, cbar = plot_grouped_data(
        data_prop, data_version, ax, grouped_data,
        case_params['group_col'], case_params['T_col'],
        case_params['H_col'], case_params['prop_col'],
        case_params['T_units'], case_params['H_units'], case_params['prop_units'],
        T_range, H_range, offset, at_temps, colormap,
        case_params['marker'], case_params['markersize'],
        case_params['linestyle'], case_params['legend_label'], case_params['colorbar_label'],
        legend, colorbar, plot_kwargs, colorbar_kwargs
    )

    return ax, cbar

def grouping_plotting_params(
    data_version: PlotDataVersion,
    at_temps: Optional[ArrayLike],
    presets: Presets,
    group_df: pd.DataFrame
    ) -> dict[str, Union[str, np.int64, np.float64, NDArray[np.float64]]]:
    '''Return grouping and plotting parameters based on whether grouping is by field or by temp.'''

    r_cols = RAW_DF_COLUMNS
    p_cols = PROCESSED_DF_COLUMNS

    if at_temps is None:
        match_col = None
        match_values = None
        markersize = PLOT_DEFAULTS['markersize']
        legend_label = PLOT_DEFAULTS['H_legend_label']
        colorbar_label = PLOT_DEFAULTS['H_xlabel']
        if data_version in ['raw', 'converted', 'compare']:
            group_col = r_cols['H']
            groups = presets['fields']
            decimals = presets['decimals']
            max_diff = presets['max_diff']
            marker = PLOT_DEFAULTS['dot_marker']
            linestyle = PLOT_DEFAULTS['dot_linestyle']
        elif data_version == 'processed':
            group_col = p_cols['H']
            groups = group_df.loc[:, p_cols['H']]
            decimals = MAX_DECIMALS
            max_diff = np.inf
            marker = PLOT_DEFAULTS['line_marker']
            linestyle = PLOT_DEFAULTS['line_linestyle']
    else:
        groups = np.array(at_temps, dtype=np.float64, ndmin=1)
        decimals = MAX_DECIMALS
        max_diff = np.diff(at_temps).min() / 2.
        marker = PLOT_DEFAULTS['T_marker']
        markersize = PLOT_DEFAULTS['T_markersize']
        legend_label = PLOT_DEFAULTS['T_legend_label']
        colorbar_label = PLOT_DEFAULTS['T_xlabel']

        if data_version == 'compare':
            linestyle = PLOT_DEFAULTS['T_compare_linestyle']
        else:
            linestyle = PLOT_DEFAULTS['T_linestyle']

        if data_version in ['raw', 'converted', 'compare']:
            group_col = r_cols['T']
            match_col = r_cols['H']
            match_values = presets['fields']
        else:
            group_col = p_cols['T']
            match_col = p_cols['H']
            match_values = group_df.loc[:, p_cols['H']]

    if data_version == 'compare':
        marker = PLOT_DEFAULTS['compare_marker']

    return {
        'group_col': group_col,
        'groups': groups,
        'decimals': decimals,
        'max_diff': max_diff,
        'match_col': match_col,
        'match_values': match_values,
        'marker': marker,
        'markersize': markersize,
        'linestyle': linestyle,
        'legend_label': legend_label,
        'colorbar_label': colorbar_label
    }

def plot_grouped_data(
    data_prop: DataProp,
    data_version: PlotDataVersion,
    ax: Optional[Axes],
    grouped_data: DataFrameGroupBy,
    group_col: str,
    T_col: str,
    H_col: str,
    prop_col: str,
    T_units: str,
    H_units: str,
    prop_units: str,
    T_range: ArrayLike,
    H_range: ArrayLike,
    offset: float,
    at_temps: Optional[ArrayLike],
    colormap: Optional[Union[Colormap, str]],
    default_marker: str,
    default_markersize: str,
    default_linestyle: str,
    default_legend_label: str,
    default_colorbar_label: str,
    legend: bool,
    colorbar: bool,
    plot_kwargs: Optional[Union[Kwargs, Sequence[Kwargs]]],
    colorbar_kwargs: Optional[Kwargs]
    ) -> tuple[Axes, Optional[Colorbar]]:
    '''Plot each group in grouped_data as a line.'''

    # check arguments

    ax = vd.check_axes(ax)

    group_col, T_col, H_col, prop_col = vd.cast_strs(group_col, T_col, H_col, prop_col)

    if group_col not in (T_col, H_col):
        raise ValueError('group_col must be equal to either T_col or H_col.')

    T_units, H_units = vd.cast_strs(T_units, H_units)
    T_range = vd.check_range(T_range, 'T_range')
    H_range = vd.check_range(H_range, 'H_range')

    offset = np.float64(offset)

    if colormap is None:
        colormap = PLOT_DEFAULTS[f'{"H" if at_temps is None else "T"}_lines_cmap']
    elif not isinstance(colormap, Colormap):
        colormap = str(colormap)

    plot_kwargs = vd.check_plot_kwargs(plot_kwargs, grouped_data.ngroups)

    # cycle colors
    cmap = plt.get_cmap(colormap)
    colors = [cmap(x) for x in np.linspace(0, 1, grouped_data.ngroups)]
    ax.set_prop_cycle(cycler('color', colors))

    # variables that depend on grouping
    if group_col == T_col:
        x_col = H_col
        x_min, x_max = H_range
        grp_min, grp_max = T_range
        units_str = T_units
    else:
        x_col = T_col
        x_min, x_max = T_range
        grp_min, grp_max = H_range
        units_str = H_units

    _plot_each_group(
        ax, grouped_data, prop_col, offset,
        default_marker, default_markersize, default_linestyle, default_legend_label,
        plot_kwargs, x_col, x_min, x_max, grp_min, grp_max, units_str
    )

    _label_lines(
        ax, data_prop, data_version, ('H' if at_temps is None else 'T'), offset,
        T_units, H_units, prop_units
    )

    if legend:
        ax.legend()

    if colorbar:
        cbar = _colorbar_lines(
            ax, grouped_data,
            default_colorbar_label, colorbar_kwargs,
            cmap, units_str
        )
    else:
        cbar = None

    return ax, cbar

def _plot_each_group(
    ax: Axes,
    grouped_data: DataFrameGroupBy,
    prop_col: str,
    offset: np.float64,
    default_marker: str,
    default_markersize: str,
    default_linestyle: str,
    default_legend_label: str,
    plot_kwargs: list[Kwargs],
    x_col: str,
    x_min: np.float64,
    x_max: np.float64,
    grp_min: np.float64,
    grp_max: np.float64,
    units_str: str
    ) -> None:
    '''Plot each group as a line, taking supplied limits and default formatting into account.'''

    for i, (grp, da) in enumerate(grouped_data):

        if (grp_min <= grp <= grp_max) and (len(da) > 0):

            da_subset = da.loc[(da.loc[:, x_col] >= x_min) & (da.loc[:, x_col] <= x_max), :]
            x_vals = da_subset.loc[:, x_col].values
            y_vals = da_subset.loc[:, prop_col].values

            plot_kwargs[i]['marker'] = plot_kwargs[i].get('marker', default_marker)
            plot_kwargs[i]['markersize'] = plot_kwargs[i].get('markersize', default_markersize)
            plot_kwargs[i]['linestyle'] = plot_kwargs[i].get('linestyle', default_linestyle)
            plot_kwargs[i]['linewidth'] = plot_kwargs[i].get(
                'linewidth',
                PLOT_DEFAULTS['linewidth']
            )

            if default_legend_label is not None:
                plot_kwargs[i]['label'] = plot_kwargs[i].get(
                    'label',
                    default_legend_label.format(
                        mag_str=np.around(grp, decimals=6),
                        units_str=units_str
                    )
                )

            ax.plot(x_vals, y_vals + i*offset, **(plot_kwargs[i]))

def _label_lines(
    ax: Axes,
    data_prop: DataProp,
    data_version: PlotDataVersion,
    grouping: GroupingProp,
    offset: np.float64,
    T_units: str,
    H_units: str,
    prop_units: str
    ) -> None:
    '''Label line plot with appropriate xlabel, ylabel, and title.'''

    offset_str = '' if offset == 0 else OFFSET_FMT_STR.format(offset=offset)

    if grouping == 'H':
        x_var, x_unit, grp_unit = 'T', T_units, H_units
    else:
        x_var, x_unit, grp_unit = 'H', H_units, T_units

    ax.set(
        title=PLOT_DEFAULTS[f'{grouping}_title'].format(
            version_str=data_version.capitalize(),
            offset_str=offset_str,
            units_str=grp_unit
        ),
        xlabel=PLOT_DEFAULTS[f'{x_var}_xlabel'].format(units_str=x_unit),
        ylabel=PLOT_DEFAULTS[f'{data_prop}_ylabel'].format(units_str=prop_units)
    )

def _colorbar_lines(
    ax: Axes,
    grouped_data: DataFrameGroupBy,
    default_colorbar_label: str,
    colorbar_kwargs: Optional[Kwargs],
    cmap: Colormap,
    units_str: str
    ) -> Colorbar:
    '''Add a color bar to the parent |Figure| of `ax`.'''

    colorbar_kwargs = vd.check_kwargs(colorbar_kwargs)

    if default_colorbar_label is not None:
        colorbar_kwargs['label'] = colorbar_kwargs.get(
            'label',
            default_colorbar_label.format(units_str=units_str)
        )

    # construct tick labels and equally-spaced colorbar bounds
    tick_labels = list(grouped_data.groups.keys())
    colorbar_bounds = np.linspace(0, 1, len(tick_labels) + 1)
    colorbar_norm = mpl.colors.BoundaryNorm(colorbar_bounds, cmap.N)

    # construct tick locations
    colorbar_kwargs['ticks'] = np.linspace(
        1/len(tick_labels)/2,
        1-1/len(tick_labels)/2,
        len(tick_labels)
    )

    cbar = ax.get_figure().colorbar(
        mpl.cm.ScalarMappable(norm=colorbar_norm, cmap=cmap),
        **colorbar_kwargs
    )
    cbar.ax.tick_params(size=0)
    cbar.set_ticklabels(np.around(tick_labels, decimals=6))

    return cbar

def plot_map(
    data_prop: DataProp,
    data_version: MapDataVersion,
    ax: Axes,
    T_range: ArrayLike,
    H_range: ArrayLike,
    T_npoints: int,
    H_npoints: int,
    interp_method: str,
    center: Optional[bool],
    contour: bool,
    colorbar: bool,
    imshow_kwargs: Kwargs,
    contour_kwargs: Kwargs,
    colorbar_kwargs: Kwargs,
    df: pd.DataFrame,
    cols: ColumnDataDict,
    T_units: str,
    H_units: str,
    prop_units: str
    ) -> tuple[Axes, Optional[Colorbar]]:
    '''Plot data as a map.'''

    ax, center, contour, imshow_kwargs, contour_kwargs, colorbar_kwargs  = vd.check_map_args(
        data_prop, ax, center, contour, imshow_kwargs, contour_kwargs, colorbar_kwargs
    )

    T_min, T_max, H_min, H_max, prop_min, prop_max, grid_T, grid_H, grid = bounds_and_grid(
        data_prop, T_range, H_range, T_npoints, H_npoints, interp_method, df, cols
    )

    image = _imshow_map(
        ax, center, imshow_kwargs,
        T_min, T_max, H_min, H_max, prop_min, prop_max, grid
    )

    if contour:
        _contour_map(ax, contour_kwargs, grid_T, grid_H, grid)

    _label_map(ax, data_prop, data_version, T_units, H_units, prop_units)

    if colorbar:
        prop_cbar_label = PLOT_DEFAULTS[f'{data_prop}_ylabel'].format(units_str=prop_units)
        cbar = _colorbar_map(
            ax, imshow_kwargs, colorbar_kwargs,
            prop_cbar_label, prop_min, prop_max, image
        )
    else:
        cbar = None

    return ax, cbar

def bounds_and_grid(
    data_prop: DataProp,
    T_range: ArrayLike,
    H_range: ArrayLike,
    T_npoints: int,
    H_npoints: int,
    interp_method: str,
    df: pd.DataFrame,
    cols: ColumnDataDict,
    ) -> tuple[
        np.float64, np.float64, np.float64, np.float64, np.float64,
        NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
    ]:
    '''
    Return the T, H, and prop bounds, and the corresponding grids.

    See :func:`_get_complete_grid`.
    '''

    T_values = df[cols['T']].values
    H_values = df[cols['H']].values
    prop_values = df[cols[data_prop]].values

    T_range, H_range, T_npoints, H_npoints, T_min, T_max, H_min, H_max = vd.check_grid_args(
        T_range, H_range, T_npoints, H_npoints, T_values, H_values
    )

    prop_min = prop_values.min()
    prop_max = prop_values.max()

    grid_T, grid_H, grid = _get_complete_grid(
        T_npoints, H_npoints,
        T_values, H_values, prop_values,
        T_min, T_max, H_min, H_max,
        interp_method
    )

    return T_min, T_max, H_min, H_max, prop_min, prop_max, grid_T, grid_H, grid

def _get_complete_grid(
    T_npoints: np.int64,
    H_npoints: np.int64,
    T_values: NDArray[np.float64],
    H_values: NDArray[np.float64],
    prop_values: NDArray[np.float64],
    T_min: np.float64,
    T_max: np.float64,
    H_min: np.float64,
    H_max: np.float64,
    interp_method: str
    ) -> NDArray[np.float64]:
    '''
    Return a grid of prop values interpolated using `interp_method` with
    `T_npoints` in the horizontal direction and `H_npoints` in the vertical
    direction, limited by `T_min`, `T_max`, `H_min`, and `H_max`.

    Any ``NaN`` values left from the `interp_method` are filled by the values
    using the ``'nearest'`` method.

    Returns the T and H grids used to make the interpolated grid, and the
    interpolated grid.
    '''

    grid_T, grid_H, grid = _get_grid(
        T_npoints, H_npoints,
        T_values, H_values, prop_values,
        T_min, T_max, H_min, H_max,
        interp_method
    )

    _, _, nearest_grid = _get_grid(
        T_npoints, H_npoints,
        T_values, H_values, prop_values,
        T_min, T_max, H_min, H_max,
        'nearest'
    )

    grid_nan_idx = np.isnan(grid)
    grid[grid_nan_idx] = nearest_grid[grid_nan_idx]

    return grid_T, grid_H, grid

def _get_grid(
    T_npoints: np.int64,
    H_npoints: np.int64,
    T_values: NDArray[np.float64],
    H_values: NDArray[np.float64],
    prop_values: NDArray[np.float64],
    T_min: np.float64,
    T_max: np.float64,
    H_min: np.float64,
    H_max: np.float64,
    interp_method: str
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    '''
    Construct a grid of prop values interpolated using `interp_method` with
    `T_npoints` in the horizontal direction and `H_npoints` in the vertical
    direction, limited by `T_min`, `T_max`, `H_min`, and `H_max`.

    Returns the T and H grids used to make the interpolated grid, and the
    interpolated grid.
    '''

    grid_step_T = complex(0, T_npoints)
    grid_step_H = complex(0, H_npoints)
    grid_T, grid_H = np.mgrid[T_min:T_max:grid_step_T, H_min:H_max:grid_step_H]
    grid = griddata((T_values, H_values), prop_values, (grid_T, grid_H), method=interp_method)

    return grid_T.T, grid_H.T, grid.T

def _imshow_map(
    ax: Axes,
    center: bool,
    imshow_kwargs: Kwargs,
    T_min: np.float64,
    T_max: np.float64,
    H_min: np.float64,
    H_max: np.float64,
    prop_min: np.float64,
    prop_max: np.float64,
    grid: NDArray[np.float64]
    ) -> AxesImage:
    '''Set default settings and call :meth:`matplotlib.axes.Axes.imshow`.'''

    if center and prop_min < 0 < prop_max:
        v_bound = min(abs(prop_min), abs(prop_max))
        imshow_kwargs['vmin'] = imshow_kwargs.get('vmin', -1 * v_bound)
        imshow_kwargs['vmax'] = imshow_kwargs.get('vmax', v_bound)
        imshow_kwargs['cmap'] = imshow_kwargs.get('cmap', PLOT_DEFAULTS['diverging_map_cmap'])
    else:
        imshow_kwargs['cmap'] = imshow_kwargs.get('cmap', PLOT_DEFAULTS['sequential_map_cmap'])

    imshow_kwargs['extent'] = imshow_kwargs.get('extent', (T_min, T_max, H_min, H_max))
    imshow_kwargs['interpolation'] = imshow_kwargs.get('interpolation', PLOT_DEFAULTS['map_interp'])
    imshow_kwargs['aspect'] = imshow_kwargs.get('aspect', PLOT_DEFAULTS['map_aspect'])
    imshow_kwargs['origin'] = imshow_kwargs.get('origin', PLOT_DEFAULTS['map_origin'])

    return ax.imshow(grid, **imshow_kwargs)

def _contour_map(
    ax: Axes,
    contour_kwargs: Kwargs,
    grid_T: NDArray[np.float64],
    grid_H: NDArray[np.float64],
    grid: NDArray[np.float64]
    ) -> None:
    '''Set default settings and call :meth:`matplotlib.axes.Axes.contour`.'''

    if 'cmap' not in contour_kwargs:
        # override dashed lines for negative values if using default color
        if 'colors' not in contour_kwargs:
            contour_kwargs['linestyles'] = PLOT_DEFAULTS['contour_linestyles']
        # use default color if cmap not supplied
        contour_kwargs['colors'] = contour_kwargs.get('colors', PLOT_DEFAULTS['contour_colors'])

    contour_kwargs['origin'] = contour_kwargs.get('origin', PLOT_DEFAULTS['contour_origin'])

    ax.contour(grid_T, grid_H, grid, **contour_kwargs)

def _label_map(
    ax: Axes,
    data_prop: DataProp,
    data_version: MapDataVersion,
    T_units: str,
    H_units: str,
    prop_units: str
    ) -> None:
    '''Label map plot with appropriate xlabel, ylabel, and title.'''

    ax.set(
        xlabel=PLOT_DEFAULTS['T_xlabel'].format(units_str=T_units),
        ylabel=PLOT_DEFAULTS['H_xlabel'].format(units_str=H_units),
        title=PLOT_DEFAULTS[f'{data_prop}_{data_version}_title'].format(units_str=prop_units)
    )

def _colorbar_map(
    ax: Axes,
    imshow_kwargs: Kwargs,
    colorbar_kwargs: Kwargs,
    prop_cbar_label: str,
    prop_min: np.float64,
    prop_max: np.float64,
    image: AxesImage
    ) -> Colorbar:
    '''Add a color bar to the parent |Figure| of `ax`.'''

    colorbar_kwargs['label'] = colorbar_kwargs.get('label', prop_cbar_label)

    # add arrow to top and/or bottom of colorbar if colors extend beyond limits

    cmin = imshow_kwargs.get('vmin', prop_min)
    cmax = imshow_kwargs.get('vmax', prop_max)

    if cmin > prop_min and cmax < prop_max:
        colorbar_kwargs['extend'] = colorbar_kwargs.get('extend', 'both')
    elif cmin > prop_min:
        colorbar_kwargs['extend'] = colorbar_kwargs.get('extend', 'min')
    elif cmax < prop_max:
        colorbar_kwargs['extend'] = colorbar_kwargs.get('extend', 'max')

    cbar = ax.get_figure().colorbar(image, **colorbar_kwargs)
    cbar.ax.minorticks_off()

    return cbar
