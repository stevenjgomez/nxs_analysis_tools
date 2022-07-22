'''
Contains functions to perform Tikhonov regularization for smoothing data.

This code is based on work by Jonathan Stickel and is a modification of his
`scikit-datasmooth <regularsmooth>`_ package.

Stickel's implementation notes, references, copyright notice, list of
conditions, and disclaimer are copied below.

.. _regularsmooth: https://github.com/jjstickel/scikit-datasmooth/blob/976ab86998d1648506684360ab9d65b8a3ccf078/scikits/datasmooth/regularsmooth.py
'''

#Implementation Notes
#--------------------
#    Smooth data by regularization as described in [1]. Optimal values
#    for the regularization parameter, lambda, can be calulated using
#    the generalized cross-validation method described in [2] or by
#    constraining the standard deviation between the smoothed and
#    measured data as described in [3]. Both methods for calculating
#    lambda are reviewed in [1].
#
#References
#----------
#    [1] Comput. Chem. Eng. (2010) 34, 467
#    [2] Anal. Chem. (2003) 75, 3631
#    [3] AIChE J. (2006) 52, 325
#
#
#Copyright (c) 2010, Jonathan Stickel
#
#All rights reserved.
#
#
#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions
#are met:
#
#  Redistributions of source code must retain the above copyright
#  notice, this list of conditions and the following disclaimer.
#
#  Redistributions in binary form must reproduce the above copyright
#  notice, this list of conditions and the following disclaimer in the
#  documentation and/or other materials provided with the
#  distribution.
#
#  Neither the name of Jonathan Stickel nor the names of any
#  contributors may be used to endorse or promote products derived
#  from this software without specific prior written permission.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from typing import Union, Optional
import logging

import numpy as np
from numpy.typing import NDArray
from scipy import optimize
from numba import njit, prange

from .typedefs import Kwargs, MatchErrStr, ToMinimizeStr

logger = logging.getLogger('magentropy')

def smooth_processed(
    used_fields: NDArray[np.float64],
    df_arrays: tuple[NDArray[np.float64], ...],
    output: NDArray[np.float64],
    d: np.int64,
    lms: NDArray[np.float64],
    lm_g: np.float64,
    we: bool,
    me: Union[bool, NDArray[np.float64], MatchErrStr],
    mk: Kwargs
    ) -> None:
    '''Perform the magnetic momemnt smoothing for the processed data.'''

    for i in range(len(used_fields)):
        _smooth_one(i, used_fields, df_arrays, output, d, lms, lm_g, we, me, mk)

def _smooth_one(
    i: np.int64,
    used_fields: NDArray[np.float64],
    df_arrays: tuple[NDArray[np.float64], ...],
    output: NDArray[np.float64],
    d: np.int64,
    lms: NDArray[np.float64],
    lm_g: np.float64,
    we: bool,
    me: Union[bool, NDArray[np.float64], MatchErrStr],
    mk: Kwargs
    ) -> None:
    '''
    Perform smoothing on one field at index `i`.

    Data indices corresponding to columns are T: 0, M: 2, M_err: 3

    ``output[0,:,0]`` is a 1D ``ndarray`` of the temperature grid.
    '''

    weights = 1. / df_arrays[i][:, 3]**2 if we else np.ones_like(df_arrays[i][:, 3])

    if np.isnan(lms[i]):
        to_minimize, stdev_or_errors = _lm_guess_params(i, df_arrays, me)
        lms[i] = _smooth_guess_lm(
            i, df_arrays, output, d, lm_g,
            weights, stdev_or_errors, to_minimize, mk
        )
    else:
        output[i+1:,:,2] = _solve_regularization(
            x=df_arrays[i][:, 0],
            y=df_arrays[i][:, 2],
            xhat=output[0,:,0],
            weights=weights,
            d_order=d,
            lmbd_s=lms[i],
        )

    logger.info('Processed M(T) at field: %s', used_fields[i])

def _lm_guess_params(
    i: np.int64,
    df_arrays: tuple[NDArray[np.float64], ...],
    me: Union[bool, NDArray[np.float64], MatchErrStr]
    ) -> tuple[str, Optional[np.float64]]:
    '''
    Determine the type of minimization and standard deviation or error
    values to use for determining the optimal :math:`\\lambda`.
    '''

    if isinstance(me, bool):
        if me:
            to_minimize = 'err'
            stdev_or_errors = df_arrays[i][:, 3]
        else:
            to_minimize = 'gcv'
            stdev_or_errors = None
    else:
        to_minimize = 'std'
        if me == 'min':
            stdev_or_errors = df_arrays[i][:, 3].min()
        elif me == 'mean':
            stdev_or_errors = df_arrays[i][:, 3].mean()
        elif me == 'max':
            stdev_or_errors = df_arrays[i][:, 3].max()
        elif len(me) == 1:
            stdev_or_errors = me[0]
        else:
            stdev_or_errors = me[i]

    if stdev_or_errors is None or np.isnan(stdev_or_errors):
        to_minimize = 'gcv'
        stdev_or_errors = None

    return to_minimize, stdev_or_errors

def _smooth_guess_lm(
    i: np.int64,
    df_arrays: tuple[NDArray[np.float64], ...],
    output: NDArray[np.float64],
    d: np.int64,
    lm_g: np.float64,
    weights: NDArray[np.float64],
    stdev_or_errors: Optional[Union[np.float64, NDArray[np.float64]]],
    to_minimize: ToMinimizeStr,
    mk: Kwargs
    ) -> np.float64:
    '''Perform the smoothing when no :math:`\\lambda` is supplied.'''

    lm, output[i+1:,:,2] = _minimize(
        to_minimize=to_minimize,
        x=df_arrays[i][:, 0],
        y=df_arrays[i][:, 2],
        xhat=output[0,:,0],
        weights=weights,
        d_order=d,
        lmbd_s_guess=lm_g,
        stdev_or_errors=stdev_or_errors,
        min_kwargs=mk
    )

    return lm

@njit(parallel=False, cache=True)
def _solve_regularization(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    xhat: NDArray[np.float64],
    weights: NDArray[np.float64],
    d_order: np.int64,
    lmbd_s: np.float64,
    ) -> NDArray[np.float64]:
    '''
    Solve the regularization problem a single time and return yhat,
    the smoothed values, as a 1D ``ndarray``.
    '''

    _, m, mtw, dtd, b, delta_inv = _regularization_setup(x, y, xhat, weights, d_order)
    a = _a_matrix(m, mtw, dtd, delta_inv, lmbd_s)

    return np.linalg.solve(a, b).ravel()

@njit(parallel=False, cache=True)
def _regularization_setup(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    xhat: NDArray[np.float64],
    weights: NDArray[np.float64],
    d_order: np.int64
    ) -> tuple[
        NDArray[np.float64], NDArray[np.float64], NDArray[np.float64],
        NDArray[np.float64], NDArray[np.float64], np.float64
    ]:
    '''Return regularization calculation results that do not depend on :math:`\\lambda`.'''

    y_column = y.copy().reshape(len(y), 1) # copy ensures contiguous array
    m, mtw, dtd, b = _regularization_matrices(x, y_column, xhat, weights, d_order)
    delta_inv = (np.trace(dtd) / len(xhat)**(d_order+2))**(-1)

    return y_column, m, mtw, dtd, b, delta_inv

@njit(parallel=True, cache=True)
def _a_matrix(
    m: NDArray[np.float64],
    mtw: NDArray[np.float64],
    dtd: NDArray[np.float64],
    delta_inv: np.float64,
    lmbd_s: np.float64
    ) -> NDArray[np.float64]:
    '''
    Return the "A" matrix for the regularization calculation, which depends on :math:`\\lambda`.

    :math:`A = M^T W M + \\lambda_s \\delta^{-1} D^T D`
    '''

    return mtw @ m + lmbd_s * delta_inv * dtd

@njit(parallel=False, cache=True)
def _regularization_matrices(
    x: NDArray[np.float64],
    y_column: NDArray[np.float64],
    xhat: NDArray[np.float64],
    weights: NDArray[np.float64],
    d_order: np.int64
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    '''
    Return certain matrices for the regularization calculation that do not
    depend on :math:`\\lambda`.
    '''

    m = _mapping_matrix(x, xhat)
    w = _weight_matrix(weights)
    d = _derivative_matrix(len(xhat), xhat[1]-xhat[0], d_order)

    mtw = m.T @ w
    dtd = d.T @ d
    b = mtw @ y_column

    return m, mtw, dtd, b

@njit(parallel=True, cache=True)
def _mapping_matrix(x: NDArray[np.float64], xhat: NDArray[np.float64]) -> NDArray[np.float64]:
    '''Construct the mapping matrix for regularization.'''

    # indices where x could be inserted in xhat that preserve order
    sort_idxs = np.searchsorted(xhat, x, 'right') - 1
    # for extrapolation
    sort_idxs[sort_idxs==-1] += 1
    sort_idxs[sort_idxs==len(xhat)-1] += -1

    # weights for right-side values
    m2 = (x - xhat[sort_idxs])/(xhat[sort_idxs+1] - xhat[sort_idxs])
    # weights for left-side values
    m1 = 1 - m2

    # create matrix of zeros with proper shape
    m = np.zeros((len(x), len(xhat)), dtype=np.float64)

    # add at the appropriate spots
    for i in range(len(x)):
        m[i, sort_idxs[i]] = m1[i]
        m[i, sort_idxs[i]+1] = m2[i]

    ## this would be more efficient, but advanced indexing is not supported
    #j = np.arange(len(x))
    #m[j, sort_idxs[j]] = m1
    #m[j, sort_idxs[j]+1] = m2

    return m

@njit(parallel=True, cache=True)
def _weight_matrix(weights: NDArray[np.float64]) -> NDArray[np.float64]:
    '''Construct the normalized weight matrix for regularization.'''

    return 1./np.sum(weights) * np.diag(weights)

@njit(parallel=True, cache=True)
def _derivative_matrix(
    output_length: np.int64,
    spacing: np.float64,
    d_order: np.int64
    ) -> NDArray[np.float64]:
    '''Construct the derivative matrix for regularization.'''

    # create matrix of zeros with proper shape
    m = np.zeros((output_length, output_length), dtype=np.float64)

    # last backward difference
    m[-1, -2] = -1
    m[-1, -1] = 1

    # forward differences
    for j in prange(output_length-1):
        m[j, j] = -1
        m[j, j+1] = 1

    return np.linalg.matrix_power(m / spacing, d_order)

def _minimize(
    to_minimize: ToMinimizeStr,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    xhat: NDArray[np.float64],
    weights: NDArray[np.float64],
    d_order: np.int64,
    lmbd_s_guess: np.float64,
    stdev_or_errors: Optional[Union[np.float64, NDArray[np.float64]]],
    min_kwargs: Kwargs
    ) -> tuple[np.float64, NDArray[np.float64]]:
    '''
    Find optimal :math:`\\lambda` by minimizing an objective function.

    This requires solving the regularization problem multiple times.
    '''

    y_column, m, mtw, dtd, b, delta_inv = _regularization_setup(x, y, xhat, weights, d_order)

    min_kwargs = dict(min_kwargs)

    if to_minimize == 'gcv':
        min_kwargs['fun'] = _var_gcv
        min_kwargs['args'] = (len(x), y_column, m, mtw, dtd, b, delta_inv)
    elif to_minimize == 'std':
        min_kwargs['fun'] = _var_std
        min_kwargs['args'] = (len(x), y_column, m, mtw, dtd, b, delta_inv, abs(stdev_or_errors))
    elif to_minimize == 'err':
        min_kwargs['fun'] = _var_err
        min_kwargs['args'] = (len(x), y_column, m, mtw, dtd, b, delta_inv, abs(stdev_or_errors))
    else:
        raise ValueError('to_minimize must be one of ["gcv", "std", "err"]')

    min_kwargs['x0'] = np.array(np.log10(lmbd_s_guess), dtype=np.float64, ndmin=1)

    log10_res = optimize.minimize(**min_kwargs)

    logger.info('scipy.optimize.minimize: %s', log10_res.message)

    lmbd_gcv = 10**log10_res.x[0]
    yhat_gcv = _solve_regularization(x, y, xhat, weights, d_order, lmbd_gcv)

    return lmbd_gcv, yhat_gcv

@njit(parallel=True, cache=True)
def _var_gcv(
    log10_lmbd_s_arr: NDArray[np.float64],
    n: np.int64,
    y_column: NDArray[np.float64],
    m: NDArray[np.float64],
    mtw: NDArray[np.float64],
    dtd: NDArray[np.float64],
    b: NDArray[np.float64],
    delta_inv: np.float64
    ) -> np.float64:
    '''Return the generalized cross-validation variance, for use in ``_minimize``.'''

    a = _a_matrix(m, mtw, dtd, delta_inv, 10**log10_lmbd_s_arr[0])
    yhat = np.linalg.solve(a, b)

    err_term = m @ yhat - y_column
    numerator = err_term.T @ err_term / n
    hat_matrix = m @ np.linalg.solve(a, mtw)
    denominator = (1 - np.trace(hat_matrix) / n)**2

    return numerator / denominator

@njit(parallel=True, cache=True)
def _var_std(
    log10_lmbd_s_arr: NDArray[np.float64],
    n: np.int64,
    y_column: NDArray[np.float64],
    m: NDArray[np.float64],
    mtw: NDArray[np.float64],
    dtd: NDArray[np.float64],
    b: NDArray[np.float64],
    delta_inv: np.float64,
    stdev: np.float64
    ) -> np.float64:
    '''Return the squared standard deviation difference, for use in ``_minimize``.'''

    a = _a_matrix(m, mtw, dtd, delta_inv, 10**log10_lmbd_s_arr[0])
    yhat = np.linalg.solve(a, b)
    y_diff = (m @ yhat - y_column).reshape(n)
    stdev_diff = np.std(y_diff) * n / (n-1) ## no numba support for ddof parameter

    return (stdev_diff - stdev)**2

@njit(parallel=True, cache=True)
def _var_err(
    log10_lmbd_s_arr: NDArray[np.float64],
    n: np.int64,
    y_column: NDArray[np.float64],
    m: NDArray[np.float64],
    mtw: NDArray[np.float64],
    dtd: NDArray[np.float64],
    b: NDArray[np.float64],
    delta_inv: np.float64,
    errors: NDArray[np.float64]
    ) -> np.float64:
    '''
    Return the sum of squared differences between the absolute differences
    and supplied errors, for use in ``_minimize``.
    '''

    a = _a_matrix(m, mtw, dtd, delta_inv, 10**log10_lmbd_s_arr[0])
    yhat = np.linalg.solve(a, b)
    y_diff = (m @ yhat - y_column).reshape(n)

    return np.sum((np.abs(y_diff) - errors)**2)

## There were issues when using JIT with bootstrap.
## Only the first few fields would be computed,
## and reproducibility using a random seed could not be supported.

#@njit(parallel=True, cache=False)
def bootstrap(
    used_fields: NDArray[np.float64],
    df_arrays: tuple[NDArray[np.float64], ...],
    output: NDArray[np.float64],
    d: np.int64,
    lms: NDArray[np.float64],
    we: bool,
    n_bootstrap: np.int64,
    rng: np.random.Generator
    ) -> None:
    '''Perform the magnetic momemnt smoothing for the processed data.'''

    #for i in prange(len(used_fields)):
    #    _bootstrap_one(i, df_arrays, output, d, lms, we, n_bootstrap)
    #    with objmode():
    #        logger.info('Calculated bootstrap estimates at field: %s', used_fields[i])

    for i in range(len(used_fields)):
        _bootstrap_one(i, df_arrays, output, d, lms, we, n_bootstrap, rng)
        logger.info('Calculated bootstrap estimates at field: %s', used_fields[i])

#@njit(parallel=False, cache=True)
def _bootstrap_one(
    i: np.int64,
    df_arrays: tuple[NDArray[np.float64], ...],
    output: NDArray[np.float64],
    d: np.int64,
    lms: NDArray[np.float64],
    we: bool,
    n_bootstrap: np.int64,
    rng: np.random.Generator
    ) -> None:
    '''
    Perform bootstrap calculation on one field at index `i`.

    Data indices corresponding to columns are T: 0, M: 2, M_err: 3

    output[0,:,0] is a 1D array of the temperature grid.
    '''

    weights = 1. / df_arrays[i][:, 3]**2 if we else np.ones_like(df_arrays[i][:, 3])

    stdevs = _bootstrap_stdev(
        x=df_arrays[i][:, 0],
        y=df_arrays[i][:, 2],
        xhat=output[0,:,0],
        weights=weights,
        d_order=d,
        lmbd_s=lms[i],
        n_bootstrap=n_bootstrap,
        rng=rng
    )

    ## numba indexing issue forces this less efficient code
    #for j in range(output.shape[1]):
    #    output[i+1:,j,3] = stdevs[j]

    output[i+1, :, 3] = stdevs

#@njit(parallel=True, cache=True)
def _bootstrap_stdev(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    xhat: NDArray[np.float64],
    weights: NDArray[np.float64],
    d_order: np.int64,
    lmbd_s: np.float64,
    n_bootstrap: np.int64,
    rng: np.random.Generator
    ) -> NDArray[np.float64]:
    '''Compute the bootstrap standard deviation estimate for each `xhat` point.'''

    results = np.zeros((n_bootstrap, len(xhat)))

    for b in prange(n_bootstrap):
        results[b] = _one_bootstrap_regularization(x, y, xhat, weights, d_order, lmbd_s, rng)

    ## numba does not support axis argument for std
    #return np.array([np.std(yhats) for yhats in results.T])

    return np.std(results, axis=0)

#@njit(parallel=False, cache=True)
def _one_bootstrap_regularization(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    xhat: NDArray[np.float64],
    weights: NDArray[np.float64],
    d_order: np.int64,
    lmbd_s: np.float64,
    rng: np.random.Generator
    ) -> NDArray[np.float64]:
    '''Solve the regularization problem for one bootstrap selection of the data.'''

    ## no numba support for random generators
    #idxs = np.random.choice(len(x), len(x))

    idxs = rng.choice(len(x), len(x))

    return _solve_regularization(x[idxs], y[idxs], xhat, weights[idxs], d_order, lmbd_s)
