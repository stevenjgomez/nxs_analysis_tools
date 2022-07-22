'''Contains functions to group data by field or temperature.'''

from typing import Union, Optional

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy
from numba import njit

from .typedefs import Presets
from ._classvars import RAW_DF_COLUMNS, DEFAULT_PRESETS

def test_grouping(
    raw_df: pd.DataFrame,
    presets: Presets
    ) -> tuple[dict[str, Union[NDArray[np.float64], np.int64, np.float64]], DataFrameGroupBy]:
    '''For use in ``magentro.test_grouping`` and ``magentro.test_grouping_``.'''

    fds = presets['fields']
    dms = presets['decimals']
    md = presets['max_diff']

    return (
        {'fields': fds, 'decimals': dms, 'max_diff': md},
        group_by(
            raw_df,
            group_col=RAW_DF_COLUMNS['H'],
            groups=fds,
            decimals=dms,
            max_diff=md
        )
    )

def group_by(
    df: pd.DataFrame,
    group_col: str,
    groups: NDArray[np.float64] = DEFAULT_PRESETS['fields'],
    decimals: np.int64 = DEFAULT_PRESETS['decimals'],
    max_diff: np.float64 = DEFAULT_PRESETS['max_diff'],
    match_col: Optional[str] = None,
    match_values: Optional[NDArray[np.float64]] = None
    ) -> DataFrameGroupBy:
    '''
    Group data by temperature or field strength.

    Parameters
    ----------
    df : DataFrame
        |DataFrame| to group. Columns should be ordinary |Series|s, without
        ``PintArray``s, e.g. from ``self.raw_data`` instead of
        ``self._raw_data``.
    group_col : str
        Name of column to use for grouping.
    groups : ndarray
        Expected groups. If empty, the groups are determined automatically
        based on `max_diff` and/or `decimals`.
    decimals : int
        The decimal place to which to round the automatically determined groups.
        (A negative integer specifies the number of positions to the left of the
        decimal point. See :func:`numpy.around` `decimal` parameter documentation.)
        Ignored if `groups` is not empty. If groups is empty and `max_diff` is
        ``np.inf``, groups will be determined by rounding to this decimal place.
    max_diff : float
        If `groups` is not empty, `max_diff` is the maximum difference allowed
        between each value in the `group_col` column and the closest group in
        `groups`.
        Values too far away from any group will be omitted, unless max_diff is
        ``np.inf``.
        If `groups` is empty, `max_diff` is the maximum difference allowed
        between any two items in each group, which is used to determine groups
        automatically.
        Generally, `decimals` is enough to determine groups, but `max_diff` can
        be used for finer control.
    match_col : str, optional
        Name of column to use for matching values within group.
    match_values : ndarray, optional
        Values to match within each group.
        Only the ``len(match_values)`` observations in `match_col` of each group
        that are closest to one of `match_values` will be kept.
        Other observations will be indicated as omitted by filling with ``NaN``.

    Returns
    -------
    grouped_by : DataFrameGroupBy
    '''

    if len(groups) != 0:
        # take the (unique) groups supplied
        unique_groups = np.unique(groups)

        if max_diff != np.inf:
            # remove values that are too far away from any group
            all_diffs = abs(unique_groups[:, np.newaxis] - df.loc[:, group_col].values)
            faraway_mask = np.min(all_diffs, axis=0) <= max_diff
            df = df.loc[faraway_mask, :]

    elif max_diff != np.inf:
        # automatically determine groups
        unique_groups = find_unique_groups(df.loc[:,group_col].values, max_diff, decimals)
    else:
        # determine groups solely by rounding
        unique_groups = np.unique(
            df.loc[:, group_col].round(decimals=decimals)
        )

    # perform the groupby based on which values are closest to a group
    grouped_by = df.copy().groupby(
        lambda x: unique_groups[abs(unique_groups - df.loc[:, group_col].loc[x]).argmin()]
    )

    if match_col is not None and match_values is not None:
        match_col = str(match_col)
        match_values = np.unique(match_values)
        grps = list(grouped_by.groups.keys())
        df = grouped_by.apply(
            match_closest_values,
            group_col,
            match_col,
            match_values,
            grps
        )
        grouped_by = df.groupby(
            lambda x: unique_groups[abs(unique_groups - df.loc[:, group_col].loc[x]).argmin()]
        )

    return grouped_by

@njit(cache=True)
def find_unique_groups(
    temps_or_fields: NDArray[np.float64],
    max_diff: np.float64,
    decimals: np.int64
    ) -> NDArray[np.float64]:
    '''Return unique temperatures or fields based on `max_diff`.'''

    # order values to group
    temps_or_fields = np.sort(temps_or_fields)

    if len(temps_or_fields) < 2:
        return temps_or_fields

    # groups will be a list of lists (of varying length)
    groups = [[temps_or_fields[0]]]
    group_idx = 0

    for tf in temps_or_fields[1:]:
        # create new group if difference between value and first item of current group is large
        if abs(tf - groups[group_idx][0]) > max_diff:
            groups.append([np.float64(x) for x in range(0)]) # for numba's benefit
            group_idx += 1

        # append value to current group
        groups[group_idx].append(tf)

    # take the mean of each sub-list, then round
    rounded_mean_groups = np.zeros(len(groups), dtype=np.float64)
    np.around(
        np.array([np.array(g).mean() for g in groups], dtype=np.float64),
        decimals,
        rounded_mean_groups
    )

    return rounded_mean_groups

def match_closest_values(
    da: pd.DataFrame,
    group_col: str,
    match_col: str,
    match_values: NDArray[np.float64],
    grps: list[np.float64]
    ) -> pd.DataFrame:
    '''
    Passed to ``DataFrameGroupBy.apply``. Primarily for grouping by temperature.

    Finds the closest observations in `match_col` of `da` to any one of
    `match_values`, then limits data to one observation per match value by
    taking the one with the closest value in `group_col` to `grps[0]` for
    each match value.
    '''

    # get the current group and remove from the list
    grp = grps.pop(0)

    # array of lowest distance from a value in match_values for each observation
    match_diffs = abs(match_values[:, np.newaxis] - da.loc[:, match_col].values)
    min_idx = np.argmin(match_diffs, axis=0)

    group_values = da.loc[:, group_col].values
    group_diffs = abs(group_values - grp)
    min_distances = np.sqrt(np.min(match_diffs, axis=0)**2 + np.min(group_diffs)**2)
    closest_idxs = np.zeros(len(match_values), dtype=np.int64)

    for i in range(len(match_values)):
        group_values_subset = group_values.copy()
        group_values_subset[min_idx != i] = np.nan
        group_distances_subset = min_distances.copy()
        group_distances_subset[min_idx != i] = np.nan
        closest_idxs[i] = np.nanargmin(group_distances_subset)

    da = da.iloc[closest_idxs, :]

    return da
