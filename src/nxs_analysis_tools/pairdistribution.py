"""
Tools for generating single crystal pair distribution functions.
"""
import time
import os
import gc
import math
import warnings
from scipy.ndimage import rotate, affine_transform, binary_dilation
from scipy.spatial import ConvexHull
import scipy
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from nexusformat.nexus import nxsave, NXroot, NXentry, NXdata, NXfield
import numpy as np
from astropy.convolution import Kernel, convolve_fft
import pyfftw
from concurrent.futures import ThreadPoolExecutor
from .datareduction import plot_slice, reciprocal_lattice_params, Padder, \
    array_to_nxdata
from .lineartransformations import ShearTransformer, rotate_plane_affine, mirror_plane_affine

_rotate_plane_affine = rotate_plane_affine
_mirror_plane_affine = mirror_plane_affine

__all__ = ['symmetrize', 'Puncher', 'Interpolator',
           'fourier_transform_nxdata', 'Gaussian3DKernel', 'DeltaPDF',
           'generate_gaussian', '_rotate_plane_affine', '_mirror_plane_affine'
           ]


def _resolve_symmetry_params(
    symmetry=None,
    lattice_angle=None,
    n_fold=None,
    mirror=None,
    mirror_axis=None,
    mirror_angle=None,
    theta_min=None,
    theta_max=None,
):
    """Resolve crystallographic symmetry presets and explicit overrides."""
    preset_lattice_angle = 90.0
    preset_n_fold = None
    preset_mirror = True
    preset_mirror_angle = None
    preset_mirror_axis = None

    if symmetry is not None:
        sym_str = symmetry.lower() if isinstance(symmetry, str) else symmetry
        if sym_str == 'hexagonal':
            preset_lattice_angle = 60.0
            preset_n_fold = 6
            preset_mirror = True
            preset_mirror_angle = 30.0
            preset_mirror_axis = None
        elif sym_str == 'tetragonal':
            preset_lattice_angle = 90.0
            preset_n_fold = 4
            preset_mirror = True
            preset_mirror_angle = 45.0
            preset_mirror_axis = None
        elif sym_str == 'trigonal':
            preset_lattice_angle = 120.0
            preset_n_fold = 3
            preset_mirror = True
            preset_mirror_angle = 30.0
            preset_mirror_axis = None
        elif sym_str == 'orthorhombic':
            preset_lattice_angle = 90.0
            preset_n_fold = 2
            preset_mirror = True
            preset_mirror_axis = 0
            preset_mirror_angle = None
        else:
            raise ValueError(
                f"Unknown symmetry '{symmetry}'. Supported presets are: "
                "'hexagonal', 'tetragonal', 'trigonal', 'orthorhombic'."
            )

    resolved_lattice_angle = lattice_angle if lattice_angle is not None else preset_lattice_angle
    resolved_n_fold = n_fold if n_fold is not None else preset_n_fold
    resolved_mirror = mirror if mirror is not None else preset_mirror
    resolved_mirror_axis = mirror_axis if mirror_axis is not None else preset_mirror_axis
    resolved_mirror_angle = mirror_angle if mirror_angle is not None else preset_mirror_angle

    rotations = None
    if theta_min is not None and theta_max is not None:
        diff = abs(theta_max - theta_min)
        if diff > 0:
            if resolved_mirror:
                rotations = abs(int(360.0 / diff / 2.0))
            else:
                rotations = abs(int(360.0 / diff))
        else:
            rotations = 1
    elif resolved_n_fold is not None:
        rotations = resolved_n_fold

    return (
        resolved_lattice_angle,
        resolved_n_fold,
        resolved_mirror,
        resolved_mirror_axis,
        resolved_mirror_angle,
        rotations,
    )


def _resolve_layer_axis(data, layer_axis):
    """Resolve 3D stacking axis from index or axis name."""
    if layer_axis is None:
        return 2 if data.ndim == 3 else None
    if isinstance(layer_axis, int):
        return layer_axis
    if isinstance(layer_axis, str):
        target = layer_axis.lower().strip()
        for idx, ax in enumerate(data.nxaxes):
            ax_name = ax.nxname.lower().strip()
            if target == ax_name or target in ax_name:
                return idx
        name_map = {'h': 0, 'qh': 0, 'k': 1, 'qk': 1, 'l': 2, 'ql': 2}
        if target in name_map:
            return name_map[target]
        raise ValueError(
            f"Could not resolve layer_axis '{layer_axis}' in data axes: "
            f"{[ax.nxname for ax in data.nxaxes]}"
        )
    return int(layer_axis)


def _get_slice(data, axis_idx, i):
    idx = [slice(None)] * data.ndim
    idx[axis_idx] = i
    return data[tuple(idx)]


def _set_slice(arr, axis_idx, i, val):
    idx = [slice(None)] * arr.ndim
    idx[axis_idx] = i
    arr[tuple(idx)] = val


def _compute_transformed_bounds_2d(
    arr_2d,
    q1,
    q2,
    dq1,
    dq2,
    lattice_angle=90.0,
    n_fold=1,
    mirror=True,
    mirror_angle=None,
    mirror_axis=None,
    aspect=1.0,
):
    """
    Compute pad widths along both in-plane axes to contain all transformed symmetry equivalents
    of non-empty data points.
    """
    gamma = np.radians(lattice_angle if lattice_angle is not None else 90.0)
    asp = aspect if aspect is not None else 1.0

    B_phys = np.array([
        [1.0, asp * np.cos(gamma)],
        [0.0, asp * np.sin(gamma)]
    ], dtype=float)
    B_phys_inv = np.linalg.inv(B_phys)

    fold = n_fold if n_fold is not None else 1
    angle_step = 360.0 / fold if fold > 1 else 360.0

    rot_matrices = []
    for k in range(fold):
        theta = np.radians(angle_step * k)
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=float)
        rot_matrices.append(R)

    all_ops_phys = []
    for R in rot_matrices:
        all_ops_phys.append(B_phys_inv @ R @ B_phys)

    if mirror:
        if mirror_angle is not None:
            psi = np.radians(mirror_angle)
            Ref = np.array([[np.cos(2 * psi), np.sin(2 * psi)], [np.sin(2 * psi), -np.cos(2 * psi)]], dtype=float)
            for R in rot_matrices:
                all_ops_phys.append(B_phys_inv @ (R @ Ref) @ B_phys)
        elif mirror_axis == 0:
            M0 = np.array([[-1.0, 0.0], [0.0, 1.0]], dtype=float)
            for R in rot_matrices:
                all_ops_phys.append(B_phys_inv @ R @ B_phys @ M0)
        elif mirror_axis == 1:
            M1 = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=float)
            for R in rot_matrices:
                all_ops_phys.append(B_phys_inv @ R @ B_phys @ M1)
        elif mirror_axis in ('diagonal', 'transpose'):
            Mdiag = np.array([[0.0, dq2 / dq1 if dq1 != 0 else 1.0], [dq1 / dq2 if dq2 != 0 else 1.0, 0.0]], dtype=float)
            for R in rot_matrices:
                all_ops_phys.append(B_phys_inv @ R @ B_phys @ Mdiag)
        else:
            Ref = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=float)
            for R in rot_matrices:
                all_ops_phys.append(B_phys_inv @ (R @ Ref) @ B_phys)

    non_empty = (arr_2d != 0) & (~np.isnan(arr_2d))
    if not np.any(non_empty):
        return (0, 0), (0, 0)

    if np.all(non_empty):
        pts = np.array([
            [q1[0], q2[0]],
            [q1[0], q2[-1]],
            [q1[-1], q2[0]],
            [q1[-1], q2[-1]],
        ], dtype=float)
    else:
        idx1, idx2 = np.where(non_empty)
        pts = np.column_stack([q1[idx1], q2[idx2]])
        if len(pts) >= 3:
            try:
                hull = ConvexHull(pts)
                pts = pts[hull.vertices]
            except Exception:
                pass

    transformed_pts = []
    for op in all_ops_phys:
        transformed_pts.append(pts @ op.T)
    transformed_pts = np.vstack(transformed_pts)

    min_q1 = float(min(q1.min(), transformed_pts[:, 0].min()))
    max_q1 = float(max(q1.max(), transformed_pts[:, 0].max()))
    min_q2 = float(min(q2.min(), transformed_pts[:, 1].min()))
    max_q2 = float(max(q2.max(), transformed_pts[:, 1].max()))

    if dq1 >= 0:
        pad1_left = int(np.ceil(np.round((q1[0] - min_q1) / dq1, 6))) if min_q1 < q1[0] else 0
        pad1_right = int(np.ceil(np.round((max_q1 - q1[-1]) / dq1, 6))) if max_q1 > q1[-1] else 0
    else:
        pad1_left = int(np.ceil(np.round((max_q1 - q1[0]) / abs(dq1), 6))) if max_q1 > q1[0] else 0
        pad1_right = int(np.ceil(np.round((q1[-1] - min_q1) / abs(dq1), 6))) if min_q1 < q1[-1] else 0

    if dq2 >= 0:
        pad2_left = int(np.ceil(np.round((q2[0] - min_q2) / dq2, 6))) if min_q2 < q2[0] else 0
        pad2_right = int(np.ceil(np.round((max_q2 - q2[-1]) / dq2, 6))) if max_q2 > q2[-1] else 0
    else:
        pad2_left = int(np.ceil(np.round((max_q2 - q2[0]) / abs(dq2), 6))) if max_q2 > q2[0] else 0
        pad2_right = int(np.ceil(np.round((q2[-1] - min_q2) / abs(dq2), 6))) if min_q2 < q2[-1] else 0

    return (pad1_left, pad1_right), (pad2_left, pad2_right)


def _auto_expand(
    dataset,
    lattice_angle=90.0,
    n_fold=1,
    mirror=True,
    mirror_angle=None,
    mirror_axis=None,
    aspect=1.0,
    layer_axis=None,
):
    """
    Pad dataset so that its coordinate bounds encompass all symmetry-transformed equivalents
    of non-empty data points without truncating existing bounds.
    """
    if dataset is None:
        return None

    # Case 1: Raw NumPy ndarray (2D)
    if not hasattr(dataset, 'nxaxes') or not hasattr(dataset, 'nxsignal'):
        if not isinstance(dataset, np.ndarray) or dataset.ndim != 2:
            return dataset
        arr = np.asarray(dataset, dtype=float)
        c = np.array([(arr.shape[0] - 1) / 2.0, (arr.shape[1] - 1) / 2.0], dtype=float)
        q1 = np.arange(arr.shape[0], dtype=float) - c[0]
        q2 = np.arange(arr.shape[1], dtype=float) - c[1]
        dq1, dq2 = 1.0, 1.0
        pad1, pad2 = _compute_transformed_bounds_2d(
            arr,
            q1, q2, dq1, dq2,
            lattice_angle=lattice_angle,
            n_fold=n_fold,
            mirror=mirror,
            mirror_angle=mirror_angle,
            mirror_axis=mirror_axis,
            aspect=aspect
        )
        if pad1 == (0, 0) and pad2 == (0, 0):
            return arr
        return np.pad(arr, (pad1, pad2), mode='constant', constant_values=0.0)

    # Case 2: NXdata (2D or 3D)
    if dataset.ndim == 2:
        ax1_idx, ax2_idx = 0, 1
        layer_idx = None
        arr = np.asarray(dataset.nxsignal.nxdata, dtype=float)
        in_plane_arr = arr
    elif dataset.ndim == 3:
        layer_idx = _resolve_layer_axis(dataset, layer_axis)
        plane_axes = [i for i in range(3) if i != layer_idx]
        ax1_idx, ax2_idx = plane_axes[0], plane_axes[1]
        arr = np.asarray(dataset.nxsignal.nxdata, dtype=float)
        in_plane_arr = ((arr != 0) & (~np.isnan(arr))).any(axis=layer_idx)
    else:
        return dataset

    q1 = np.asarray(dataset.nxaxes[ax1_idx].nxdata, dtype=float)
    q2 = np.asarray(dataset.nxaxes[ax2_idx].nxdata, dtype=float)
    dq1 = float((q1[-1] - q1[0]) / (len(q1) - 1)) if len(q1) > 1 else 1.0
    dq2 = float((q2[-1] - q2[0]) / (len(q2) - 1)) if len(q2) > 1 else 1.0

    pad1, pad2 = _compute_transformed_bounds_2d(
        in_plane_arr,
        q1, q2, dq1, dq2,
        lattice_angle=lattice_angle,
        n_fold=n_fold,
        mirror=mirror,
        mirror_angle=mirror_angle,
        mirror_axis=mirror_axis,
        aspect=aspect
    )

    pad_widths = [None] * dataset.ndim
    pad_widths[ax1_idx] = pad1
    pad_widths[ax2_idx] = pad2

    if layer_idx is not None:
        q_layer = np.asarray(dataset.nxaxes[layer_idx].nxdata, dtype=float)
        dq_layer = float((q_layer[-1] - q_layer[0]) / (len(q_layer) - 1)) if len(q_layer) > 1 else 1.0
        layer_has_data = np.where(((arr != 0) & (~np.isnan(arr))).any(axis=(ax1_idx, ax2_idx)))[0]
        if len(layer_has_data) > 0:
            active_L = q_layer[layer_has_data]
            min_L = min(q_layer.min(), -active_L.max())
            max_L = max(q_layer.max(), -active_L.min())
        else:
            min_L = q_layer.min()
            max_L = q_layer.max()

        if dq_layer >= 0:
            pad_L_left = int(np.ceil(np.round((q_layer[0] - min_L) / dq_layer, 6))) if min_L < q_layer[0] else 0
            pad_L_right = int(np.ceil(np.round((max_L - q_layer[-1]) / dq_layer, 6))) if max_L > q_layer[-1] else 0
        else:
            pad_L_left = int(np.ceil(np.round((max_L - q_layer[0]) / abs(dq_layer), 6))) if max_L > q_layer[0] else 0
            pad_L_right = int(np.ceil(np.round((q_layer[-1] - min_L) / abs(dq_layer), 6))) if min_L < q_layer[-1] else 0
        pad_widths[layer_idx] = (pad_L_left, pad_L_right)

    if all(pw == (0, 0) for pw in pad_widths):
        return dataset

    arr_padded = np.pad(arr, pad_widths, mode='constant', constant_values=0.0)

    new_axes = []
    for idx in range(dataset.ndim):
        ax = dataset.nxaxes[idx]
        q_old = np.asarray(ax.nxdata, dtype=float)
        pw_left, pw_right = pad_widths[idx]
        dq = float((q_old[-1] - q_old[0]) / (len(q_old) - 1)) if len(q_old) > 1 else 1.0
        if dq >= 0:
            q_left = q_old[0] - np.arange(pw_left, 0, -1) * dq if pw_left > 0 else np.array([], dtype=float)
            q_right = q_old[-1] + np.arange(1, pw_right + 1) * dq if pw_right > 0 else np.array([], dtype=float)
        else:
            q_left = q_old[0] + np.arange(pw_left, 0, -1) * abs(dq) if pw_left > 0 else np.array([], dtype=float)
            q_right = q_old[-1] - np.arange(1, pw_right + 1) * abs(dq) if pw_right > 0 else np.array([], dtype=float)
        q_new = np.concatenate([q_left, q_old, q_right])

        new_ax = NXfield(q_new, name=ax.nxname)
        for k, v in ax.attrs.items():
            new_ax.attrs[k] = v
        new_axes.append(new_ax)

    new_sig = NXfield(arr_padded, name=dataset.nxsignal.nxname)
    for k, v in dataset.nxsignal.attrs.items():
        new_sig.attrs[k] = v

    new_data = NXdata(new_sig, tuple(new_axes))
    for k, v in dataset.attrs.items():
        if k not in ('axes', 'signal'):
            new_data.attrs[k] = v

    return new_data


def _symmetrize_2d_slice_average(
    slice_data,
    lattice_angle=90.0,
    n_fold=1,
    mirror=True,
    mirror_angle=None,
    mirror_axis=None,
    aspect=1.0,
    order=1,
    positive_values=True
):
    if hasattr(slice_data, 'nxsignal'):
        arr = np.asarray(slice_data.nxsignal.nxdata, dtype=float)
        q1 = np.asarray(slice_data.nxaxes[0].nxdata, dtype=float)
        q2 = np.asarray(slice_data.nxaxes[1].nxdata, dtype=float)
        dq1 = float((q1[-1] - q1[0]) / (len(q1) - 1)) if len(q1) > 1 else 1.0
        dq2 = float((q2[-1] - q2[0]) / (len(q2) - 1)) if len(q2) > 1 else 1.0
        c = np.array([-q1[0] / dq1, -q2[0] / dq2], dtype=float)
    else:
        arr = np.asarray(slice_data, dtype=float)
        dq1, dq2 = 1.0, 1.0
        c = np.array([(arr.shape[0] - 1) / 2.0, (arr.shape[1] - 1) / 2.0], dtype=float)

    rotations = [arr]

    fold = n_fold if n_fold is not None else 1
    angle_step = 360.0 / fold if fold > 1 else 360.0
    if fold > 1:
        for k in range(1, fold):
            rot = rotate_plane_affine(
                arr,
                lattice_angle=lattice_angle,
                rotation_angle=angle_step * k,
                dq1=dq1,
                dq2=dq2,
                origin=c,
                aspect=aspect,
                order=order,
                cval=0.0
            )
            rotations.append(rot)

    if mirror:
        mirr = mirror_plane_affine(
            arr,
            lattice_angle=lattice_angle,
            mirror_angle=mirror_angle,
            mirror_axis=mirror_axis,
            dq1=dq1,
            dq2=dq2,
            origin=c,
            aspect=aspect,
            order=order,
            cval=0.0
        )
        rotations.append(mirr)
        if fold > 1:
            for k in range(1, fold):
                rot_m = rotate_plane_affine(
                    mirr,
                    lattice_angle=lattice_angle,
                    rotation_angle=angle_step * k,
                    dq1=dq1,
                    dq2=dq2,
                    origin=c,
                    aspect=aspect,
                    order=order,
                    cval=0.0
                )
                rotations.append(rot_m)

    stack = np.array(rotations)
    valid = (stack != 0) & (~np.isnan(stack))
    count_valid = valid.sum(axis=0)
    sum_valid = np.where(valid, stack, 0.0).sum(axis=0)

    averaged = np.zeros_like(sum_valid)
    np.divide(sum_valid, count_valid, out=averaged, where=(count_valid > 0))

    if positive_values:
        averaged[averaged < 0] = 0.0

    return averaged, count_valid


def _symmetrize_2d_average(
    data,
    lattice_angle=90.0,
    n_fold=1,
    mirror=True,
    mirror_angle=None,
    mirror_axis=None,
    aspect=1.0,
    order=1,
    positive_values=True
):
    avg, _ = _symmetrize_2d_slice_average(
        data,
        lattice_angle=lattice_angle,
        n_fold=n_fold,
        mirror=mirror,
        mirror_angle=mirror_angle,
        mirror_axis=mirror_axis,
        aspect=aspect,
        order=order,
        positive_values=False
    )
    if positive_values:
        if np.any(avg < 0):
            warnings.warn(
                "Negative values found in symmetrized dataset and clipped to zero. "
                "Set positive_values=False to override.",
                UserWarning,
                stacklevel=3
            )
        avg[avg < 0] = 0.0
    if hasattr(data, 'nxaxes'):
        return array_to_nxdata(avg, data)
    return avg


def _symmetrize_2d_wedge(
    data,
    theta_min=None,
    theta_max=None,
    lattice_angle=90.0,
    n_fold=None,
    mirror=True,
    mirror_axis=None,
    rotations=None,
    positive_values=True,
    return_diagnostics=False,
):
    mirror = mirror if mirror is not None else True

    if mirror and mirror_axis is None:
        warnings.warn(
            "mirror_axis not specified. Defaulting to 0. "
            "Set mirror_axis explicitly when using mirror=True.",
            UserWarning,
            stacklevel=3
        )
        mirror_axis = 0

    if theta_min is None or theta_max is None:
        fold = n_fold if n_fold is not None else 6
        if mirror:
            theta_min = 0.0 if theta_min is None else theta_min
            theta_max = 360.0 / (2 * fold) if theta_max is None else theta_max
            rotations = fold
        else:
            theta_min = 0.0 if theta_min is None else theta_min
            theta_max = 360.0 / fold if theta_max is None else theta_max
            rotations = fold
    elif rotations is None:
        diff = abs(theta_max - theta_min)
        if diff > 0:
            if mirror:
                rotations = abs(int(360.0 / diff / 2.0))
            else:
                rotations = abs(int(360.0 / diff))
        else:
            rotations = 1

    p = Padder(data)
    padding = tuple(len(axis) for axis in data.nxaxes)
    data_padded = p.pad(padding)

    q1 = data_padded.nxaxes[0]
    q2 = data_padded.nxaxes[1]

    theta = np.arctan2(q1.reshape((-1, 1)), q2.reshape((1, -1)))
    theta = np.mod(theta, 2 * np.pi)

    theta_min_rad = np.deg2rad(theta_min % 360)
    theta_max_rad = np.deg2rad(theta_max % 360)

    if theta_min_rad <= theta_max_rad:
        symmetrization_mask = (theta >= theta_min_rad) & (theta <= theta_max_rad)
    else:
        symmetrization_mask = (theta >= theta_min_rad) | (theta <= theta_max_rad)

    transformer = ShearTransformer(lattice_angle)
    mask = array_to_nxdata(transformer.invert(symmetrization_mask), data_padded)
    unpadded_mask = p.unpad(mask)

    wedge = mask * data_padded
    unpadded_wedge = p.unpad(wedge)
    wedge_arr = wedge[data.nxsignal.nxname].nxdata
    wedge_arr = transformer.apply(wedge_arr)

    scale = wedge_arr.shape[0] / wedge_arr.shape[1]
    wedge_arr = affine_transform(
        wedge_arr,
        Affine2D().scale(scale, 1).get_matrix()[:2, :2],
        offset=[(1 - scale) * wedge_arr.shape[0] / 2, 0],
        order=0,
    )

    reconstructed = np.zeros(wedge_arr.shape)
    rot_step = 360.0 / rotations if rotations > 0 else 360.0
    for _ in range(rotations):
        reconstructed += wedge_arr
        wedge_arr = rotate(wedge_arr, rot_step, reshape=False, order=0)

    if mirror:
        reconstructed = np.where(
            reconstructed == 0,
            reconstructed + np.flip(reconstructed, axis=mirror_axis),
            reconstructed
        )

    reconstructed = affine_transform(
        reconstructed,
        Affine2D().scale(scale, 1).inverted().get_matrix()[:2, :2],
        offset=[-(1 - scale) * wedge_arr.shape[0] / 2 / scale, 0],
        order=0,
    )
    reconstructed = transformer.invert(reconstructed)
    reconstructed = p.unpad(reconstructed)

    sig_max = float(data.nxsignal.nxdata.max())
    reconstructed[reconstructed > sig_max] = sig_max
    if positive_values:
        if np.any(reconstructed < 0):
            warnings.warn(
                "Negative values found in symmetrized dataset and clipped to zero. "
                "Set positive_values=False to override.",
                UserWarning,
                stacklevel=3
            )
        reconstructed[reconstructed < 0] = 0.0

    symmetrized_nx = NXdata(NXfield(reconstructed, name=data.nxsignal.nxname), (data.nxaxes[0], data.nxaxes[1]))
    if return_diagnostics:
        return symmetrized_nx, unpadded_mask, unpadded_wedge
    return symmetrized_nx


def _symmetrize_3d_average(
    data,
    layer_axis=2,
    lattice_angle=90.0,
    n_fold=1,
    mirror=True,
    mirror_angle=None,
    mirror_axis=None,
    aspect=1.0,
    tol=0.01,
    positive_values=True,
    parallel=False,
    num_workers=None
):
    axis_idx = _resolve_layer_axis(data, layer_axis)

    layer_coords = np.asarray(data.nxaxes[axis_idx].nxdata, dtype=float)
    n_layers = len(layer_coords)

    processed = set()
    pairs = []
    for i in range(n_layers):
        if i in processed:
            continue
        L_i = layer_coords[i]
        if np.isclose(L_i, 0.0, atol=tol):
            pairs.append((i, None))
            processed.add(i)
        else:
            diffs = np.abs(layer_coords - (-L_i))
            j = int(np.argmin(diffs))
            if diffs[j] <= tol and j != i:
                pairs.append((i, j))
                processed.add(i)
                processed.add(j)
            else:
                pairs.append((i, None))
                processed.add(i)

    output_array = np.zeros(data.nxsignal.shape, dtype=float)

    def process_pair(pair):
        idx_i, idx_j = pair
        slice_i = _get_slice(data, axis_idx, idx_i)
        avg_i, counts_i = _symmetrize_2d_slice_average(
            slice_i,
            lattice_angle=lattice_angle,
            n_fold=n_fold,
            mirror=mirror,
            mirror_angle=mirror_angle,
            mirror_axis=mirror_axis,
            aspect=aspect,
            positive_values=positive_values
        )
        if idx_j is None:
            return [(idx_i, avg_i)]
        else:
            slice_j = _get_slice(data, axis_idx, idx_j)
            avg_j, counts_j = _symmetrize_2d_slice_average(
                slice_j,
                lattice_angle=lattice_angle,
                n_fold=n_fold,
                mirror=mirror,
                mirror_angle=mirror_angle,
                mirror_axis=mirror_axis,
                aspect=aspect,
                positive_values=positive_values
            )
            total_counts = counts_i + counts_j
            combined = np.zeros_like(avg_i)
            weighted_sum = avg_i * counts_i + avg_j * counts_j
            np.divide(weighted_sum, total_counts, out=combined, where=(total_counts > 0))
            if positive_values:
                combined[combined < 0] = 0.0
            return [(idx_i, combined), (idx_j, combined)]

    if parallel:
        if num_workers is None:
            num_workers = max(1, (os.cpu_count() or 1) - 1)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for pair_results in executor.map(process_pair, pairs):
                for idx_layer, layer_arr in pair_results:
                    _set_slice(output_array, axis_idx, idx_layer, layer_arr)
    else:
        for pair in pairs:
            for idx_layer, layer_arr in process_pair(pair):
                _set_slice(output_array, axis_idx, idx_layer, layer_arr)

    if positive_values:
        if np.any(output_array < 0):
            warnings.warn(
                "Negative values found in symmetrized dataset and clipped to zero. "
                "Set positive_values=False to override.",
                UserWarning,
                stacklevel=3
            )
        output_array[output_array < 0] = 0.0

    return array_to_nxdata(output_array, data)


def _symmetrize_3d_wedge(
    data,
    layer_axis=2,
    theta_min=None,
    theta_max=None,
    lattice_angle=90.0,
    n_fold=None,
    mirror=True,
    mirror_axis=None,
    rotations=None,
    positive_values=True,
):
    axis_idx = _resolve_layer_axis(data, layer_axis)
    plane_axes = [i for i in range(3) if i != axis_idx]
    plane_name = data.nxaxes[plane_axes[0]].nxname + data.nxaxes[plane_axes[1]].nxname
    layer_ax = data.nxaxes[axis_idx]

    starttime = time.time()
    out_array = np.zeros(data.nxsignal.shape, dtype=float)

    print(f'Symmetrizing {plane_name} planes...')
    for idx, value in enumerate(layer_ax):
        print(f'Symmetrizing {layer_ax.nxname}={float(value):.02f}...', end='\r')
        slice_2d = _get_slice(data, axis_idx, idx)
        data_symmetrized = _symmetrize_2d_wedge(
            slice_2d,
            theta_min=theta_min,
            theta_max=theta_max,
            lattice_angle=lattice_angle,
            n_fold=n_fold,
            mirror=mirror,
            mirror_axis=mirror_axis,
            rotations=rotations,
            positive_values=False,
        )
        _set_slice(out_array, axis_idx, idx, data_symmetrized[data.nxsignal.nxname].nxdata)
    print(f'\nSymmetrized {plane_name} planes.')

    if positive_values:
        if np.any(out_array < 0):
            warnings.warn(
                "Negative values found in symmetrized dataset and clipped to zero. "
                "Set positive_values=False to override.",
                UserWarning,
                stacklevel=3
            )
        out_array[out_array < 0] = 0.0

    stoptime = time.time()
    print(f"\nSymmetrization finished in {((stoptime - starttime) / 60):.02f} minutes.")

    return array_to_nxdata(out_array, data)


def symmetrize(
    data,
    symmetry=None,
    method=None,
    lattice_angle=None,
    n_fold=None,
    layer_axis=None,
    mirror=None,
    mirror_axis=None,
    mirror_angle=None,
    theta_min=None,
    theta_max=None,
    tol=0.01,
    aspect=1.0,
    positive_values=True,
    auto_expand=True,
    parallel=False,
    num_workers=None,
    test=False,
    slice_coord=None,
    **kwargs
):
    """
    Symmetrize a 2D or 3D reciprocal space dataset.

    Supports two symmetrization methods:
    - ``'average'``: Averages over all n-fold rotational and mirror symmetry equivalents,
      pairing +L and -L layers in 3D weighted by valid pixel counts. Uses single-pass
      affine transformations to avoid destructive rebinning.
    - ``'wedge'``: Extracts an angular wedge ``(theta_min, theta_max)`` and reconstructs
      the dataset via rotation and mirroring.

    Parameters
    ----------
    data : :class:`nexusformat.nexus.tree.NXdata` or :class:`numpy.ndarray`
        The 2D or 3D dataset to symmetrize.
    symmetry : {'hexagonal', 'tetragonal', 'trigonal', 'orthorhombic'}, optional
        Crystallographic symmetry preset. Default is None.
    method : {'average', 'wedge'}, optional
        Symmetrization algorithm. Defaults to ``'wedge'`` when ``theta_min`` or ``theta_max``
        is specified, otherwise ``'average'``.
    lattice_angle : float, optional
        Angle between the two in-plane principal lattice axes in degrees (gamma).
        Defaults to 90.0, or the preset default if ``symmetry`` is specified.
    n_fold : int, optional
        Rotational symmetry fold (e.g., 6, 4, 3, 2).
    layer_axis : int or str, optional
        The stacking/layer axis for 3D datasets (e.g., ``2``, ``'L'``, ``'Ql'``, ``0``, ``'H'``).
        Defaults to ``2`` for 3D data.
    mirror : bool, optional
        Whether to include mirror reflection symmetry. Defaults to True.
    mirror_axis : int or str, optional
        Discrete mirror axis (``0``, ``1``, or ``'diagonal'``).
    mirror_angle : float, optional
        Cartesian reflection angle in degrees.
    theta_min : float, optional
        Minimum angle in degrees for wedge symmetrization.
    theta_max : float, optional
        Maximum angle in degrees for wedge symmetrization.
    tol : float, optional
        Coordinate tolerance for pairing +L and -L layers in 3D data. Defaults to 0.01.
    aspect : float, optional
        Aspect ratio ``|b*| / |a*|`` of in-plane basis vectors. Defaults to 1.0.
    positive_values : bool, optional
        If True, clips negative values to zero and issues a warning. Defaults to True.
    auto_expand : bool, optional
        Whether to automatically extend coordinate bounds along transformed axes so that
        rotated or mirrored non-empty data is not cut off. Never truncates existing bounds.
        Defaults to True.
    parallel : bool, optional
        Whether to use multi-threaded parallelism for 3D layer processing. Defaults to False.
    num_workers : int, optional
        Number of worker threads when ``parallel=True``. Defaults to ``max(1, os.cpu_count() - 1)``.
    test : bool, optional
        If True, displays a diagnostic visualization of the symmetrization process
        (for 3D data, previews a single 2D cross-section at ``slice_coord``). Defaults to False.
    slice_coord : float, optional
        Coordinate along ``layer_axis`` to preview when ``test=True`` on a 3D dataset.
        Defaults to ``0.0`` (or the nearest grid value).
    **kwargs : dict
        Additional keyword arguments forwarded to :func:`plot_slice` when ``test=True``.

    Returns
    -------
    :class:`nexusformat.nexus.tree.NXdata` or :class:`numpy.ndarray`
        The symmetrized dataset (or the symmetrized 2D cross-section when ``test=True`` on 3D data).
    """
    if data is None:
        raise ValueError("No data provided to symmetrize().")

    if method is None:
        if theta_min is not None or theta_max is not None:
            method = 'wedge'
        else:
            method = 'average'

    (
        resolved_lattice_angle,
        resolved_n_fold,
        resolved_mirror,
        resolved_mirror_axis,
        resolved_mirror_angle,
        rotations,
    ) = _resolve_symmetry_params(
        symmetry=symmetry,
        lattice_angle=lattice_angle,
        n_fold=n_fold,
        mirror=mirror,
        mirror_axis=mirror_axis,
        mirror_angle=mirror_angle,
        theta_min=theta_min,
        theta_max=theta_max,
    )

    fold_for_bounds = resolved_n_fold if resolved_n_fold is not None else (rotations if rotations is not None else 1)

    if data.ndim == 2:
        working_data = (
            _auto_expand(
                data,
                lattice_angle=resolved_lattice_angle,
                n_fold=fold_for_bounds,
                mirror=resolved_mirror,
                mirror_angle=resolved_mirror_angle,
                mirror_axis=resolved_mirror_axis,
                aspect=aspect,
            )
            if auto_expand
            else data
        )

        if method == 'average':
            res = _symmetrize_2d_average(
                working_data,
                lattice_angle=resolved_lattice_angle,
                n_fold=resolved_n_fold,
                mirror=resolved_mirror,
                mirror_angle=resolved_mirror_angle,
                mirror_axis=resolved_mirror_axis,
                aspect=aspect,
                positive_values=positive_values,
            )
            if test:
                fig, axesarr = plt.subplots(1, 2, figsize=(10, 4.5))
                plot_slice(data, skew_angle=resolved_lattice_angle, ax=axesarr[0], title='Original Data', **kwargs)
                plot_slice(res, skew_angle=resolved_lattice_angle, ax=axesarr[1], title='Symmetrized (Average)', **kwargs)
                plt.subplots_adjust(wspace=0.3)
                plt.show()
            return res

        elif method == 'wedge':
            if test:
                res, sym_mask, wedge_data = _symmetrize_2d_wedge(
                    working_data,
                    theta_min=theta_min,
                    theta_max=theta_max,
                    lattice_angle=resolved_lattice_angle,
                    n_fold=resolved_n_fold,
                    mirror=resolved_mirror,
                    mirror_axis=resolved_mirror_axis,
                    rotations=rotations,
                    positive_values=positive_values,
                    return_diagnostics=True,
                )
                fig, axesarr = plt.subplots(2, 2, figsize=(10, 8))
                axes = axesarr.reshape(-1)
                plot_slice(working_data, skew_angle=resolved_lattice_angle, ax=axes[0], title='data', **kwargs)
                filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ('vmin', 'vmax')}
                plot_slice(sym_mask, skew_angle=resolved_lattice_angle, ax=axes[1], title='mask', **filtered_kwargs)
                plot_slice(wedge_data, skew_angle=resolved_lattice_angle, ax=axes[2], title='wedge', **kwargs)
                plot_slice(res, skew_angle=resolved_lattice_angle, ax=axes[3], title='symmetrized', **kwargs)
                plt.subplots_adjust(wspace=0.4)
                plt.show()
                return res
            else:
                return _symmetrize_2d_wedge(
                    working_data,
                    theta_min=theta_min,
                    theta_max=theta_max,
                    lattice_angle=resolved_lattice_angle,
                    n_fold=resolved_n_fold,
                    mirror=resolved_mirror,
                    mirror_axis=resolved_mirror_axis,
                    rotations=rotations,
                    positive_values=positive_values,
                    return_diagnostics=False,
                )
        else:
            raise ValueError(f"Unknown symmetrization method '{method}'. Choose 'average' or 'wedge'.")

    elif data.ndim == 3:
        axis_idx = _resolve_layer_axis(data, layer_axis)

        if test:
            coords = np.asarray(data.nxaxes[axis_idx].nxdata, dtype=float)
            if slice_coord is None:
                slice_coord = 0.0 if (coords.min() <= 0.0 <= coords.max()) else float(coords[len(coords) // 2])
            idx_i = int(np.argmin(np.abs(coords - slice_coord)))
            orig_slice = _get_slice(data, axis_idx, idx_i)
            return symmetrize(
                orig_slice,
                symmetry=symmetry,
                method=method,
                lattice_angle=resolved_lattice_angle,
                n_fold=resolved_n_fold,
                mirror=resolved_mirror,
                mirror_axis=resolved_mirror_axis,
                mirror_angle=resolved_mirror_angle,
                theta_min=theta_min,
                theta_max=theta_max,
                aspect=aspect,
                positive_values=positive_values,
                auto_expand=auto_expand,
                test=True,
                **kwargs,
            )

        working_data = (
            _auto_expand(
                data,
                lattice_angle=resolved_lattice_angle,
                n_fold=fold_for_bounds,
                mirror=resolved_mirror,
                mirror_angle=resolved_mirror_angle,
                mirror_axis=resolved_mirror_axis,
                aspect=aspect,
                layer_axis=axis_idx,
            )
            if auto_expand
            else data
        )

        if method == 'average':
            return _symmetrize_3d_average(
                data=working_data,
                layer_axis=axis_idx,
                lattice_angle=resolved_lattice_angle,
                n_fold=resolved_n_fold,
                mirror=resolved_mirror,
                mirror_angle=resolved_mirror_angle,
                mirror_axis=resolved_mirror_axis,
                aspect=aspect,
                tol=tol,
                positive_values=positive_values,
                parallel=parallel,
                num_workers=num_workers,
            )
        elif method == 'wedge':
            return _symmetrize_3d_wedge(
                data=working_data,
                layer_axis=axis_idx,
                theta_min=theta_min,
                theta_max=theta_max,
                lattice_angle=resolved_lattice_angle,
                n_fold=resolved_n_fold,
                mirror=resolved_mirror,
                mirror_axis=resolved_mirror_axis,
                rotations=rotations,
                positive_values=positive_values,
            )
        else:
            raise ValueError(f"Unknown symmetrization method '{method}'. Choose 'average' or 'wedge'.")
    else:
        raise ValueError(f"symmetrize() supports 2D or 3D datasets, got {data.ndim}D.")


def _identify_axis_name(ax, default_name):
    """Identify physical reciprocal space axis ('h', 'k', 'l') from an axis or field."""
    name = getattr(ax, "nxname", getattr(ax, "name", None))
    if name is None and isinstance(ax, str):
        name = ax
    if name is not None:
        name_lower = str(name).lower()
        if name_lower in ("h", "qh") or name_lower.endswith("_h") or name_lower.startswith("h_"):
            return "h"
        elif name_lower in ("k", "qk") or name_lower.endswith("_k") or name_lower.startswith("k_"):
            return "k"
        elif name_lower in ("l", "ql") or name_lower.endswith("_l") or name_lower.startswith("l_"):
            return "l"
    return default_name


def generate_gaussian(
    H,
    K=None,
    L=None,
    amp=1.0,
    stddev=1.0,
    lattice_params=None,
    coeffs=None,
    center=None,
    return_nxdata=False,
):
    """
    Generate a 3D Gaussian distribution.

    This function creates a 3D Gaussian distribution in reciprocal space based
    on the specified parameters. Supports inputs as three separate coordinate
    arrays/fields or directly as a 3D :class:`nexusformat.nexus.tree.NXdata` object,
    automatically aligning coordinates to reciprocal lattice axes regardless of
    whether axes are ordered ``[Qh, Qk, Ql]`` or ``[Ql, Qk, Qh]``.

    Parameters
    ----------
    H : :class:`nexusformat.nexus.tree.NXdata`, :class:`nexusformat.nexus.tree.NXfield`, or :class:`numpy.ndarray`
        The first coordinate axis, or a 3D :class:`nexusformat.nexus.tree.NXdata` object.
    K : :class:`nexusformat.nexus.tree.NXfield` or :class:`numpy.ndarray`, optional
        The second coordinate axis. If `H` is an NXdata object, this parameter is omitted
        or can positionally provide `amp`.
    L : :class:`nexusformat.nexus.tree.NXfield` or :class:`numpy.ndarray`, optional
        The third coordinate axis. If `H` is an NXdata object, this parameter is omitted
        or can positionally provide `stddev`.
    amp : float, optional
        Amplitude of the Gaussian distribution. Default is 1.0.
    stddev : float, optional
        Standard deviation of the Gaussian distribution. Default is 1.0.
    lattice_params : tuple, optional
        Lattice parameters ``(a, b, c, alpha, beta, gamma)``.
    coeffs : tuple or list, optional
        Coefficients for the quadratic form in reciprocal space:
        ``(A, B, C, D, E, F)`` corresponding to
        ``A*H**2 + B*(b*/a*)*H*K + C*(b*/a*)**2*K**2 + D*(b**c*/a**2)*K*L + E*(c*/a*)**2*L**2 + F*(c*/a*)*L*H``.
        Default is ``(1, 0, 1, 0, 1, 0)``.
    center : tuple or dict, optional
        Center of the Gaussian in physical reciprocal coordinates (H, K, L).
        Default is ``(0, 0, 0)``.
    return_nxdata : bool, optional
        If True and an NXdata object was provided, returns the generated Gaussian
        wrapped in an :class:`nexusformat.nexus.tree.NXdata` object with matching metadata.
        Default is False.

    Returns
    -------
    gaussian : :class:`numpy.ndarray` or :class:`nexusformat.nexus.tree.NXdata`
        3D Gaussian distribution matching the shape and axis orientation of the input.
    """
    data_template = None
    if isinstance(H, NXdata):
        data_template = H
        axes = list(data_template.nxaxes)
        if len(axes) != 3:
            raise ValueError("generate_gaussian requires a 3-dimensional NXdata object.")
        ax0, ax1, ax2 = axes[0], axes[1], axes[2]
        if K is not None and not isinstance(K, (np.ndarray, NXfield, list, tuple)):
            # Positional calling: generate_gaussian(data, amp, stddev, lattice_params, [coeffs], [center])
            center = lattice_params if center is None else center
            coeffs = stddev if coeffs is None else coeffs
            lattice_params = amp if lattice_params is None else lattice_params
            stddev = L if stddev is None else stddev
            amp = K
    else:
        if K is None or L is None:
            raise ValueError("H, K, and L coordinate axes must all be provided if first argument is not NXdata.")
        ax0, ax1, ax2 = H, K, L

    if lattice_params is None and data_template is not None:
        lattice_params = getattr(data_template, "lattice_params", None)
    if lattice_params is None:
        raise ValueError("lattice_params (a, b, c, alpha, beta, gamma) must be provided.")

    if center is None:
        center = (0.0, 0.0, 0.0)

    if coeffs is None:
        coeffs = (1, 0, 1, 0, 1, 0)

    val0 = ax0.nxdata if hasattr(ax0, "nxdata") else np.asarray(ax0)
    val1 = ax1.nxdata if hasattr(ax1, "nxdata") else np.asarray(ax1)
    val2 = ax2.nxdata if hasattr(ax2, "nxdata") else np.asarray(ax2)

    names = [
        _identify_axis_name(ax0, "h"),
        _identify_axis_name(ax1, "k"),
        _identify_axis_name(ax2, "l"),
    ]

    # If axes cannot be uniquely identified as h, k, l, fall back to argument order [h, k, l]
    if set(names) != {"h", "k", "l"}:
        names = ["h", "k", "l"]

    # Reciprocal lattice parameters
    a, b, c, alpha, beta, gamma = lattice_params
    a_, b_, c_, *_ = reciprocal_lattice_params((a, b, c, alpha, beta, gamma))

    # Build coordinate grid matching the input axes ordering
    grid0, grid1, grid2 = np.meshgrid(val0, val1, val2, indexing="ij")
    grids = {names[0]: grid0, names[1]: grid1, names[2]: grid2}

    # Shift coordinates according to physical H, K, L center
    if isinstance(center, dict):
        c_h = center.get("H", center.get("h", 0.0))
        c_k = center.get("K", center.get("k", 0.0))
        c_l = center.get("L", center.get("l", 0.0))
    else:
        c_h, c_k, c_l = center[0], center[1], center[2]

    grid_H = grids["h"] - c_h
    grid_K = grids["k"] - c_k
    grid_L = grids["l"] - c_l

    A, B, C, D, E, F = coeffs

    # Quadratic form in reciprocal space
    quad = (
        A * grid_H**2
        + B * (b_ / a_) * grid_H * grid_K
        + C * (b_ / a_)**2 * grid_K**2
        + D * (b_ * c_ / a_**2) * grid_K * grid_L
        + E * (c_ / a_)**2 * grid_L**2
        + F * (c_ / a_) * grid_L * grid_H
    )

    gaussian = amp * np.exp(-quad / (2 * stddev**2))

    if return_nxdata and data_template is not None:
        return array_to_nxdata(gaussian, data_template, signal_name="gaussian")

    return gaussian

class Puncher:
    """
    A class for applying masks to datasets, typically for data processing in reciprocal space.

    This class provides methods for setting data, applying masks, and generating
     masks based on various criteria. It can be used to "punch" or modify datasets
      by setting specific regions to NaN according to the mask.

    Attributes
    ----------
    data : :class:`nexusformat.nexus.tree.NXdata`
        The input dataset to be processed.
    meshgrid : tuple of :class:`numpy.ndarray`
        Meshgrid arrays for the coordinates of the dataset.
    mask : :class:`numpy.ndarray`
        The mask used for removing regions of the dataset.
    lattice_params : tuple
        The lattice parameters [e.g., (a, b, c, alpha, beta, gamma)].
    reciprocal_lattice_params : tuple
        The reciprocal lattice parameters derived from the lattice parameters.
    a, b, c, al, be, ga : float
        Individual lattice parameters.
    a_star, b_star, c_star, al_star, be_star, ga_star : float
        Individual reciprocal lattice parameters.
    punched : :class:`nexusformat.nexus.tree.NXdata`
        The dataset with regions modified (punched) based on the mask.

    Methods
    -------
    set_data(data)
        Sets the dataset to be processed and initializes the coordinate arrays and mask.
    set_lattice_params(lattice_params)
        Sets the lattice parameters and computes the reciprocal lattice parameters.
    add_mask(maskaddition)
        Adds regions to the current mask using a logical OR operation.
    subtract_mask(masksubtraction)
        Removes regions from the current mask using a logical AND NOT operation.
    generate_bragg_mask(punch_radius, coeffs=None, thresh=None)
        Generates an ellipsoidal mask at integer coordinates.
    generate_intensity_mask(thresh, radius, verbose=True)
        Generates a mask based on intensity thresholds, including a pixel-spherical
         region around high-intensity points.
    generate_mask_at_coord(coordinate, punch_radius, coeffs=None, thresh=None)
        Generates an ellipsoidal mask centered at a specific coordinate in reciprocal space
         with a specified radius.
    punch()
        Applies the mask to the dataset, setting masked regions to NaN.
    """

    def __init__(self):
        """
        Initialize the Puncher object.

        This method sets up the initial state of the Puncher instance, including
         attributes for storing the dataset, lattice parameters, and masks.
          It prepares the object for further data processing and masking operations.

        Attributes
        ----------
        data : :class:`nexusformat.nexus.tree.NXdata`, optional
            The input dataset to be processed, initialized as None.
        meshgrid : tuple of :class:`numpy.ndarray`
            Meshgrid arrays for the coordinates of the dataset. Initialized as None.
        mask : :class:`numpy.ndarray`, optional
            The mask used for removing regions of the dataset, initialized as None.
        reciprocal_lattice_params : tuple, optional
            The reciprocal lattice parameters, initialized as None.
        lattice_params : tuple, optional
            The lattice parameters (a, b, c, alpha, beta, gamma),
             initialized as None.
        a, b, c, al, be, ga : float
            Individual lattice parameters, initialized as None.
        a_star, b_star, c_star, al_star, be_star, ga_star : float
            Individual reciprocal lattice parameters, initialized as None.
        punched : :class:`nexusformat.nexus.tree.NXdata`, optional
            The dataset with modified (punched) regions, initialized as None.
        """
        self.punched = None
        self.data = None
        self.meshgrid = None
        self.mask = None
        self.reciprocal_lattice_params = None
        self.lattice_params = None
        self.a, self.b, self.c, self.al, self.be, self.ga = [None] * 6
        self.a_star, self.b_star, self.c_star, self.al_star, self.be_star, self.ga_star = [None] * 6

    def set_data(self, data):
        """
        Set the dataset and initialize the mask if not already set.

        Parameters
        ----------
        data : :class:`nexusformat.nexus.tree.NXdata`
            The dataset to be processed.

        Notes
        -----
        This method also sets up coordinate meshgrids for the dataset.
        """
        
        self.data = data

        # Initialize mask to zeros of same shape
        if self.mask is None:
            self.mask = np.zeros(data.nxsignal.nxdata.shape)

        # Check if shapes match exactly
        if data.shape == tuple(ax.shape[0] for ax in data.nxaxes):
            axes_inputs = data.nxaxes

        # Check if shapes are 1 less than axes
        elif data.shape == tuple(ax.shape[0] - 1 for ax in data.nxaxes):
            axes_inputs = [ax[:-1] for ax in data.nxaxes]

        # Handle shape mismatch
        else:
            raise ValueError(f"Data shape {data.shape} does not match axes lengths.")

        # Dynamically unpack the axes into np.meshgrid
        self.meshgrid = np.meshgrid(*axes_inputs, indexing='ij')
        

    def set_lattice_params(self, lattice_params):
        """
        Set the lattice parameters [e.g., (a, b, c, alpha, beta, gamma)] and compute the reciprocal
        lattice parameters. These should be provided in the order corresponding to the axes of the
        relevant dataset.

        Parameters
        ----------
        lattice_params : tuple
            Tuple of lattice parameters (a, b, c, alpha, beta, gamma).
        """
        
        self.lattice_params = lattice_params
        self.a, self.b, self.c, self.al, self.be, self.ga = lattice_params

        self.reciprocal_lattice_params = reciprocal_lattice_params(lattice_params)
        self.a_star, self.b_star, self.c_star, \
            self.al_star, self.be_star, self.ga_star = self.reciprocal_lattice_params

    def add_mask(self, maskaddition):
        """
        Add regions to the current mask using a logical OR operation.

        Parameters
        ----------
        maskaddition : :class:`numpy.ndarray`
            The mask to be added.
        """
        self.mask = np.logical_or(self.mask, maskaddition)

    def subtract_mask(self, masksubtraction):
        """
        Remove regions from the current mask using a logical AND NOT operation.

        Parameters
        ----------
        masksubtraction : :class:`numpy.ndarray`
            The mask to be subtracted.
        """
        self.mask = np.logical_and(self.mask, np.logical_not(masksubtraction))

    def generate_bragg_mask(self, punch_radius, coeffs=None, thresh=None):
        """
        Generate an ellipsoidal mask at integer coordinates.

        Parameters
        ----------
        punch_radius : float
            Radius for the mask around each integer coordinate.
        coeffs : list, optional
            Coefficients for the expression of the ellipse/ellipsoid to be removed 
            around each integer coordinate (Bragg position). 
            For 3D: [H, HK, K, KL, L, LH] terms. Default is [1, 0, 1, 0, 1, 0].
            For 2D: [H, HK, K] terms. Default is [1, 0, 1].
        thresh : float, optional
            Intensity threshold for applying the mask.

        Returns
        -------
        mask : :class:`numpy.ndarray`
            Boolean mask identifying ellipsoidal regions to be removed around each integer coordinate.
        """
        data = self.data
        ndim = len(data.nxaxes)
        
        # Extract base parameters
        a_ = self.reciprocal_lattice_params[0]
        b_ = self.reciprocal_lattice_params[1]

        if ndim == 2:
            if coeffs is None:
                coeffs = [1, 0, 1]
                
            HH, KK = self.meshgrid
            dH, dK = HH - np.rint(HH), KK - np.rint(KK)
            
            # 2D Ellipse calculation
            mask = (coeffs[0] * dH ** 2 +
                    coeffs[1] * (b_ / a_) * dH * dK +
                    coeffs[2] * (b_ / a_) ** 2 * dK ** 2) < punch_radius ** 2

        elif ndim == 3:
            if coeffs is None:
                coeffs = [1, 0, 1, 0, 1, 0]
                
            HH, KK, LL = self.meshgrid
            c_ = self.reciprocal_lattice_params[2]
            dH, dK, dL = HH - np.rint(HH), KK - np.rint(KK), LL - np.rint(LL)
            
            # 3D Ellipsoid calculation
            mask = (coeffs[0] * dH ** 2 +
                    coeffs[1] * (b_ / a_) * dH * dK +
                    coeffs[2] * (b_ / a_) ** 2 * dK ** 2 +
                    coeffs[3] * (b_ * c_ / a_ ** 2) * dK * dL +
                    coeffs[4] * (c_ / a_) ** 2 * dL ** 2 +
                    coeffs[5] * (c_ / a_) * dL * dH) < punch_radius ** 2
                    
        else:
            raise ValueError(f"Bragg mask generation is only supported for 2D or 3D data. Found {ndim}D.")

        if thresh is not None:
            mask = np.logical_and(mask, data.nxsignal > thresh)

        return mask

    def generate_intensity_mask(self, thresh, radius, verbose=True):
        """
        Generate a mask based on intensity thresholds.

        Parameters
        ----------
        thresh : float
            Intensity threshold for creating the mask.
        radius : int
            Radius around high-intensity points to include in the mask.
        verbose : bool, optional
            Whether to print progress information.

        Returns
        -------
        mask : :class:`numpy.ndarray`
            Boolean mask highlighting regions with high intensity.
        """
        data = self.data
        counts = data.nxsignal.nxdata

        if verbose:
            print(f"Shape of data is {counts.shape}")

        # Identify all points that exceed the threshold
        base_mask = counts > thresh
        
        if verbose:
            num_peaks = np.sum(base_mask)
            print(f"Found high intensity at {num_peaks} individual points.")

        # Create an N-dimensional cube of 1s with a width of (2 * radius + 1) in every dimension
        structure_size = 2 * radius + 1
        cube_structure = np.ones((structure_size,) * counts.ndim, dtype=bool)

        # Dilate the mask
        expanded_mask = binary_dilation(base_mask, structure=cube_structure)

        if verbose:
            print("Done.")

        # Convert back to 1s and 0s
        return expanded_mask.astype(int)

    def generate_mask_at_coord(self, coordinate, punch_radius, coeffs=None, thresh=None):
        """
        Generate a mask centered at a specific coordinate.

        Parameters
        ----------
        coordinate : tuple of float
            Center coordinate for the mask.
        punch_radius : float
            Radius for the mask.
        coeffs : list, optional
            Coefficients for the expression of the ellipse/ellipsoid to be removed
            around the specific coordinate.
            For 3D: [H, HK, K, KL, L, LH] terms. Default is [1, 0, 1, 0, 1, 0].
            For 2D: [H, HK, K] terms. Default is [1, 0, 1].
        thresh : float, optional
            Intensity threshold for applying the mask.

        Returns
        -------
        mask : :class:`numpy.ndarray`
            Boolean mask for the specified coordinate.
        """
        data = self.data
        ndim = len(self.meshgrid)
        
        if len(coordinate) != ndim:
            raise ValueError(f"Coordinate length ({len(coordinate)}) must match data dimensions ({ndim}).")
            
        # Extract base parameters
        a_ = self.reciprocal_lattice_params[0]
        b_ = self.reciprocal_lattice_params[1]

        if ndim == 2:
            if coeffs is None:
                coeffs = [1, 0, 1]
                
            HH, KK = self.meshgrid
            centerH, centerK = coordinate
            dH, dK = HH - centerH, KK - centerK
            
            # 2D Ellipse calculation
            mask = (coeffs[0] * dH ** 2 +
                    coeffs[1] * (b_ / a_) * dH * dK +
                    coeffs[2] * (b_ / a_) ** 2 * dK ** 2) < punch_radius ** 2

        elif ndim == 3:
            if coeffs is None:
                coeffs = [1, 0, 1, 0, 1, 0]
                
            HH, KK, LL = self.meshgrid
            centerH, centerK, centerL = coordinate
            c_ = self.reciprocal_lattice_params[2]
            dH, dK, dL = HH - centerH, KK - centerK, LL - centerL
            
            # 3D Ellipsoid calculation
            mask = (coeffs[0] * dH ** 2 +
                    coeffs[1] * (b_ / a_) * dH * dK +
                    coeffs[2] * (b_ / a_) ** 2 * dK ** 2 +
                    coeffs[3] * (b_ * c_ / a_ ** 2) * dK * dL +
                    coeffs[4] * (c_ / a_) ** 2 * dL ** 2 +
                    coeffs[5] * (c_ / a_) * dL * dH) < punch_radius ** 2
                    
        else:
            raise ValueError(f"Coordinate mask generation is only supported for 2D or 3D data. Found {ndim}D.")

        if thresh is not None:
            mask = np.logical_and(mask, data.nxsignal > thresh)

        return mask

    def punch(self):
        """
        Apply the mask to the dataset, setting masked regions to NaN.

        This method creates a new dataset where the masked 
        regions are set to NaN, effectively "punching" those regions.
        """
        data = self.data
        
        # Dynamically pack all available axes into a single tuple
        axes_tuple = tuple(data.nxaxes)
        
        self.punched = NXdata(
            NXfield(
                np.where(self.mask, np.nan, data.nxsignal.nxdata),
                name=data.nxsignal.nxname
            ),
            axes_tuple
        )


def _round_up_to_odd_integer(value):
    """
    Round up a given number to the nearest odd integer.

    This function takes a floating-point value and rounds it up to the smallest
     odd integer that is greater than or equal to the given value.

    Parameters
    ----------
    value : float
        The input floating-point number to be rounded up.

    Returns
    -------
    int
        The nearest odd integer greater than or equal to the input value.

    Examples
    --------
    >>> _round_up_to_odd_integer(4.2)
    5

    >>> _round_up_to_odd_integer(5.0)
    5

    >>> _round_up_to_odd_integer(6.7)
    7
    """
    i = int(math.ceil(value))
    if i % 2 == 0:
        return i + 1

    return i

class Gaussian2DKernel(Kernel):
    """
    Initialize a 2D Gaussian kernel.

    This constructor creates a 2D Gaussian kernel with the specified
    standard deviation and size. The Gaussian kernel is generated based on
    the provided coefficients and is then normalized.

    Parameters
    ----------
    stddev : float
        The standard deviation of the Gaussian distribution, which controls
        the width of the kernel.

    size : tuple of int
        The dimensions of the kernel, given as (x_dim, y_dim).

    coeffs : list of float, optional
        Coefficients for the Gaussian expression.
        The default is [1, 0, 1] corresponding to the Gaussian form:
        (1 * X^2 + 0 * X * Y + 1 * Y^2).

    Raises
    ------
    ValueError
        If the dimensions in `size` are not positive integers.

    Notes
    -----
    The kernel is generated over a grid that spans twice the size of
    each dimension, and the resulting array is normalized.
    """
    _separable = True
    _is_bool = False

    def __init__(self, stddev, size, coeffs=None):
        if not coeffs:
            coeffs = [1, 0, 1]
        x_dim, y_dim = size
        x = np.linspace(-x_dim, x_dim, int(x_dim) + 1)
        y = np.linspace(-y_dim, y_dim, int(y_dim) + 1)
        X, Y = np.meshgrid(x, y)
        array = np.exp(-(coeffs[0] * X ** 2 +
                         coeffs[1] * X * Y +
                         coeffs[2] * Y ** 2) / (2 * stddev ** 2)
                       )
        self._default_size = _round_up_to_odd_integer(stddev)
        super().__init__(array)
        self.normalize()
        self._truncation = np.abs(1. - self._array.sum())

class Gaussian3DKernel(Kernel):
    """
    Initialize a 3D Gaussian kernel.

    This constructor creates a 3D Gaussian kernel with the specified
    standard deviation and size. The Gaussian kernel is generated based on
    the provided coefficients and is then normalized.

    Parameters
    ----------
    stddev : float
        The standard deviation of the Gaussian distribution, which controls
        the width of the kernel.

    size : tuple of int
        The dimensions of the kernel, given as (x_dim, y_dim, z_dim).

    coeffs : list of float, optional
        Coefficients for the Gaussian expression.
        The default is [1, 0, 1, 0, 1, 0], corresponding to the Gaussian form:
        (1 * X^2 + 0 * X * Y + 1 * Y^2 + 0 * Y * Z + 1 * Z^2 + 0 * Z * X).

    Raises
    ------
    ValueError
        If the dimensions in `size` are not positive integers.

    Notes
    -----
    The kernel is generated over a grid that spans twice the size of
    each dimension, and the resulting array is normalized.
    """
    _separable = True
    _is_bool = False

    def __init__(self, stddev, size, coeffs=None):
        if not coeffs:
            coeffs = [1, 0, 1, 0, 1, 0]
        x_dim, y_dim, z_dim = size
        x = np.linspace(-x_dim, x_dim, int(x_dim) + 1)
        y = np.linspace(-y_dim, y_dim, int(y_dim) + 1)
        z = np.linspace(-z_dim, z_dim, int(z_dim) + 1)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        array = np.exp(-(coeffs[0] * X ** 2 +
                         coeffs[1] * X * Y +
                         coeffs[2] * Y ** 2 +
                         coeffs[3] * Y * Z +
                         coeffs[4] * Z ** 2 +
                         coeffs[5] * Z * X) / (2 * stddev ** 2)
                       )
        self._default_size = _round_up_to_odd_integer(stddev)
        super().__init__(array)
        self.normalize()
        self._truncation = np.abs(1. - self._array.sum())


class Interpolator:
    """
    A class to perform data interpolation using convolution with a specified
     kernel, and to apply a window function for tapering the data.

    Attributes
    ----------
    interp_time : float or None
        Time taken for the last interpolation operation. Defaults to None.

    window : :class:`numpy.ndarray` or None
        Window function to be applied to the interpolated data. Defaults to None.

    interpolated : :class:`numpy.ndarray` or None
        The result of the interpolation operation. Defaults to None.

    data : :class:`nexusformat.nexus.tree.NXdata` or None
        The dataset to be interpolated. Defaults to None.

    kernel : :class:`numpy.ndarray` or None
        The kernel used for convolution during interpolation. Defaults to None.

    tapered : :class:`numpy.ndarray` or None
        The interpolated data after applying the window function. Defaults to None.
    """

    def __init__(self):
        """
        Initialize an Interpolator object.

        Sets up an instance of the Interpolator class with the
        following attributes initialized to None:

        - interp_time
        - window
        - interpolated
        - data
        - kernel
        - tapered

        """
        self.interp_time = None
        self.window = None
        self.interpolated = None
        self.data = None
        self.kernel = None
        self.tapered = None

    def set_data(self, data):
        """
        Set the dataset to be interpolated.

        Parameters
        ----------
        data : :class:`nexusformat.nexus.tree.NXdata`
            The dataset containing the data to be interpolated.
        """
        self.data = data
        self.interpolated = data
        self.tapered = data

        # Check if shapes match exactly
        if data.shape == tuple(ax.shape[0] for ax in data.nxaxes):
            axes_inputs = data.nxaxes

        # Check if shapes are 1 less than axes
        elif data.shape == tuple(ax.shape[0] - 1 for ax in data.nxaxes):
            axes_inputs = [ax[:-1] for ax in data.nxaxes]

        # Handle shape mismatch
        else:
            raise ValueError(f"Data shape {data.shape} does not match axes lengths.")

        # Dynamically assign self.q0, self.q1, self.q2, etc.
        for i, ax in enumerate(axes_inputs):
            setattr(self, f"q{i}", ax)

    def set_kernel(self, kernel):
        """
        Set the kernel to be used for interpolation.

        Parameters
        ----------
        kernel : :class:`numpy.ndarray`
            The kernel to be used for convolution during interpolation.
        """
        self.kernel = kernel

    def interpolate(self, verbose=True, positive_values=True):
        """
        Perform interpolation on the dataset using the specified kernel.

        This method convolves the dataset with a kernel using `convolve_fft`
        to perform interpolation. The resulting interpolated data is stored
        in the `interpolated` attribute.

        Parameters
        ----------
        verbose : bool, optional
            If True, prints progress messages and timing information
            (default is True).
        positive_values : bool, optional
            If True, sets negative interpolated values to zero
            (default is True).

        Notes
        -----
        - The convolution operation is performed in Fourier space.
        - If a previous interpolation time is recorded, it is displayed
          before starting a new interpolation.

        Returns
        -------
        None
        """
        start = time.time()

        if self.interp_time and verbose:
            print(f"Last interpolation took {self.interp_time / 60:.2f} minutes.")

        print("Running interpolation...") if verbose else None
        result = np.real(
            convolve_fft(self.data.nxsignal.nxdata,
                         self.kernel, allow_huge=True, return_fft=False))
        print("Interpolation finished.") if verbose else None

        end = time.time()
        interp_time = end - start

        print(f'Interpolation took {interp_time / 60:.2f} minutes.') if verbose else None

        if positive_values:
            if np.any(result < 0):
                warnings.warn(
                    "Negative values found in interpolated dataset and clipped to zero. "
                    "Set positive_values=False to override.",
                    UserWarning,
                    stacklevel=2
                )
            result[result < 0] = 0
        self.interpolated = array_to_nxdata(result, self.data)

    def set_tukey_window(self, tukey_alphas=None):
        """
        Set a Tukey window function for data tapering.

        Parameters
        ----------
        tukey_alphas : tuple of floats, optional
            The alpha parameters for the Tukey window in each dimension.
            Defaults to 1.0 for all dimensions.

        """
        data = self.data
        ndim = data.ndim

        # Set defaults for tukey_alphas
        if tukey_alphas is None:
            tukey_alphas = (1.0,) * ndim
        # Use provided tukey_alphas
        elif len(tukey_alphas) != ndim:
            raise ValueError(f"Length of tukey_alphas ({len(tukey_alphas)}) "
                             f"must match data dimensions ({ndim}).")

        # Initialize window accumulator
        window = 1.0

        for i in range(ndim):
            # Fetch axes
            q_axis = getattr(self, f"q{i}")
            axis_length = len(q_axis)

            # Generate the 1D Tukey window for this axis
            win_1d = scipy.signal.windows.tukey(axis_length, alpha=tukey_alphas[i])

            # Create shape tuple with 1's everywhere except the current dimension
            shape = [1] * ndim
            shape[i] = axis_length

            # Reshape the 1D window and multiply it into the main N-D window
            window = window * win_1d.reshape(shape)

        self.window = window

    def set_hexagonal_tukey_window(self, tukey_alphas=None, reverse_axes=False):
        """
        Set a hexagonal Tukey window function.

        Parameters
        ----------
        tukey_alphas : tuple of floats, optional
            The alpha parameters for the Tukey window.
            For 3D: (H, K, HK, L). Default is (1.0, 1.0, 1.0, 1.0).
            For 2D: (H, K, HK). Default is (1.0, 1.0, 1.0).
        reverse_axes : bool, optional
            If True in 3D, uses the first axis (q0) as the out-of-plane axis. 
            Default False, which uses the third axis (q2) as the out-of-plane axis.
            If True in 2D, transposes the applied hexagonal window.
        
        """

        data = self.data
        ndim = len(data.nxaxes)
        
        if ndim not in (2, 3):
            raise ValueError(f"Hexagonal window only supports 2D or 3D data. Found {ndim}D.")

        # Set defaults based on data dimensions
        if tukey_alphas is None:
            tukey_alphas = (1.0, 1.0, 1.0, 1.0) if ndim == 3 else (1.0, 1.0, 1.0)

        # Map axes dynamically depending on geometry and axes reversal
        if ndim == 3:
            if reverse_axes:
                q_H, q_K = getattr(self, "q1"), getattr(self, "q2") # In-plane
                q_L = getattr(self, "q0")                           # Out-of-plane
                shape_2d = (1, len(q_H), len(q_K))                  # Broadcast shape
                shape_1d = (len(q_L), 1, 1)
            else:
                q_H, q_K = getattr(self, "q0"), getattr(self, "q1") # In-plane
                q_L = getattr(self, "q2")                           # Out-of-plane
                shape_2d = (len(q_H), len(q_K), 1)                  # Broadcast shape
                shape_1d = (1, 1, len(q_L))
        else:
            # 2D case
            if reverse_axes:
                q_H, q_K = getattr(self, "q1"), getattr(self, "q0")
            else:
                q_H, q_K = getattr(self, "q0"), getattr(self, "q1")

        len_H, len_K = len(q_H), len(q_K)

        # Taper H and K directions
        win_H = scipy.signal.windows.tukey(len_H, alpha=tukey_alphas[0])[:, None]
        win_K = scipy.signal.windows.tukey(len_K, alpha=tukey_alphas[1])[None, :]
        win_2d = win_H * win_K

        # Taper HK direction
        truncation = int((len_H - int(len_H * np.sqrt(2) / 2)) / 2)
        
        # 1D truncated array padded with zeros
        win_HK_1d = np.concatenate((
            np.zeros(truncation),
            scipy.signal.windows.tukey(len_H - 2 * truncation, alpha=tukey_alphas[2]),
            np.zeros(truncation)
        ))
        
        # Expand 1D array to 2D
        win_HK_2d = np.tile(win_HK_1d[:, None], (1, len_K))
        
        # Rotate the 2D plane by 45 degrees
        win_HK_rot = scipy.ndimage.rotate(
            win_HK_2d, angle=45, reshape=False, mode='nearest'
        )[:len_H, :len_K] # Slicing ensures exact bounds match
        
        win_HK_rot = np.nan_to_num(win_HK_rot)
        
        # Combine the rectangle and the rotated cuts to form the hexagon
        hex_window_2d = win_2d * win_HK_rot

        # Extrude along L axis if needed
        if ndim == 2:
            self.window = hex_window_2d.T if reverse_axes else hex_window_2d
            
        elif ndim == 3:
            # Reshape the 2D hexagon to slot into the correct 3D orientation
            hex_window_3d = hex_window_2d.reshape(shape_2d)
            
            # Generate the regular 1D Tukey for the out-of-plane axis
            win_L = scipy.signal.windows.tukey(len(q_L), alpha=tukey_alphas[3]).reshape(shape_1d)
            
            # Broadcast them together
            self.window = hex_window_3d * win_L

    def set_ellipsoidal_tukey_window(self, tukey_alpha=1.0, coeffs=None):
        """
        Set an ellipsoidal/elliptical Tukey window function.

        Parameters
        ----------
        tukey_alpha : float, optional
            Tapering parameter for the Tukey window, between 0 and 1.
            - `tukey_alpha = 0` results in an ellipsoidal window (no tapering).
            - `tukey_alpha = 1` results in a full cosine taper.
            Default is 1.0.

        coeffs : tuple of float, optional
            Coefficients defining the ellipsoidal quadratic form.
            For 3D (c0, c1, c2, c3, c4, c5):
                R^2 = c0*H^2 + c1*H*K + c2*K^2 + c3*K*L + c4*L^2 + c5*L*H
            For 2D (c0, c1, c2):
                R^2 = c0*H^2 + c1*H*K + c2*K^2
            If None, coefficients are automatically set to match the edges of the
            reciprocal space axes.
            
        Sets
        ----
        self.window : :class:`numpy.ndarray`
            An N-dimensional array of the same shape as the data, containing 
            the Tukey window values between 0 and 1.
        """
        data = self.data
        ndim = len(data.nxaxes)
        
        if ndim not in (2, 3):
            raise ValueError(f"Ellipsoidal window only supports 2D or 3D data. Found {ndim}D.")
            
        # Fetch axes
        q_axes = [getattr(self, f"q{i}") for i in range(ndim)]
        
        # Calculate default coefficients if none are provided
        q_maxes = [q.max() for q in q_axes]
        smallest_extent = np.min(q_maxes)
        
        if coeffs is None:
            if ndim == 2:
                coeffs = ((smallest_extent / q_maxes[0]) ** 2, 0,
                          (smallest_extent / q_maxes[1]) ** 2)
            else:
                coeffs = ((smallest_extent / q_maxes[0]) ** 2, 0,
                          (smallest_extent / q_maxes[1]) ** 2, 0,
                          (smallest_extent / q_maxes[2]) ** 2, 0)
                          
        # Create meshgrids
        grids = np.meshgrid(*q_axes, indexing='ij', sparse=True)
        
        # Calculate the radial distance RR
        if ndim == 2:
            HH, KK = grids
            RR = np.sqrt(
                coeffs[0] * HH ** 2 +
                coeffs[1] * HH * KK +
                coeffs[2] * KK ** 2
            )
        elif ndim == 3:
            HH, KK, LL = grids
            RR = np.sqrt(
                coeffs[0] * HH ** 2 +
                coeffs[1] * HH * KK +
                coeffs[2] * KK ** 2 +
                coeffs[3] * KK * LL +
                coeffs[4] * LL ** 2 +
                coeffs[5] * LL * HH
            )
            
        # Find edges to determine Qmax
        edge_mask = np.zeros(RR.shape, dtype=bool)
        for i, q in enumerate(q_axes):
            # Logically OR the boundary conditions across all available axes
            edge_mask = edge_mask | (grids[i] == q.max())
            
        # Filter for only the edge values to find the minimum Q boundary
        edges = np.where(edge_mask, RR, RR.max())
        Qmax = edges.min()
        
        # Apply the Tukey taper
        alpha = tukey_alpha
        
        # Prevent ZeroDivisionError if alpha is set strictly to 0
        if alpha == 0:
            window = np.where(RR > Qmax, 0.0, 1.0)
        else:
            period = (Qmax * alpha) / np.pi
            
            # The inner taper region
            window = np.where(RR > Qmax * (1 - alpha), 
                              (np.cos((RR - Qmax * (1 - alpha)) / period) + 1) / 2, 
                              1.0)
                              
            # The outer hard cutoff
            window = np.where(RR > Qmax, 0.0, window)
            
        self.window = window

    def set_window(self, window):
        """
        Set a custom window function for data tapering.

        Parameters
        ----------
        window : :class:`numpy.ndarray`
            A custom window function to be applied to the interpolated data.
        """
        self.window = window

    def apply_window(self):
        """
        Apply the window function to the interpolated data.

        The window function, if set, is applied to the `interpolated` data
         to produce the `tapered` result.

        Returns
        -------
        None
        """
        self.tapered = self.interpolated * self.window


def fourier_transform_nxdata(data, method='complete', verbose=True):
    """
    Perform a Fourier Transform on an NXdata object.

    Parameters
    ----------
    data : :class:`nexusformat.nexus.tree.NXdata`
        An NXdata object containing the data to be transformed.

    method : {'complete', 'staged'}, optional
        The FFT method to use:

        - 'complete' (default): performs a full n-dimensional FFT using pyfftw.
        - 'staged': performs a 2+1D FFT in two stages - first within the planes defined by the
          first two axes, then along the third axis. Requires 3D data and the third axis
          must be normal to the plane of the first two axes.

    verbose : bool, optional
        If True, prints progress messages during computation.

    Returns
    -------
    :class:`nexusformat.nexus.tree.NXdata`
        A new NXdata object containing the Fourier-transformed data (real part).

    Notes
    -----
    - Uses `pyfftw` for efficient Fourier Transform computation.
    - Only the real part of the FFT is returned.
    - In version v0.1.15 and beyond, the default method was changed from 
      `method='staged'` to `method='complete'`. Previous behavior can be restored
      by explicitly setting `method='staged'`.
    - `method='staged'` may produce incorrect results if the third coordinate axis
      is not perpendicular to the plane of the first two axes; a warning is issued
      in this case.
    """

    fft_object = None
    fft_array = None

    start = time.time()
    print("FFT started.")

    if method == 'complete':
        print("Performing FFT...") if verbose else None

        # Allocate aligned complex array
        fft_array = pyfftw.empty_aligned(data.shape, dtype='complex128')

        # if axes is None:
        if data.ndim == 3:
            axes = (0,1,2)
        elif data.ndim == 2:
            axes = (0,1)

        # Build FFTW plan
        fft_object = pyfftw.FFTW(
            fft_array,          # input
            fft_array,          # output (in-place)
            axes=axes,
            direction='FFTW_BACKWARD',   # inverse FFT
            flags=('FFTW_MEASURE',),
        )

        # Copy real data into complex array
        fft_array[:] = np.fft.ifftshift(data.nxsignal.nxdata)

        # Execute FFT in-place
        fft_object()

        # Shift once at the end
        fft_array = np.fft.fftshift(fft_array)

        fft_real = fft_array.real

    elif method == 'staged':

        if data.ndim != 3:
            raise ValueError("Staged FFT only implemented for 3D data.")
        
        warnings.warn(
            "method='staged' is only valid when the third coordinate axis is normal to the "
            "plane spanned by the first two axes. Use method=='complete' for general cases.")

        input = data.nxsignal.nxdata

        fft_real = np.zeros(input.shape)

        print(f"Performing FFT on {data.nxaxes[0].nxname}{data.nxaxes[1].nxname} plane...")
        for k in range(0, input.shape[2]):
            fft_real[:, :, k] = np.real(
                np.fft.fftshift(
                    pyfftw.interfaces.numpy_fft.ifftn(np.fft.ifftshift(input[:, :, k]),
                                                    planner_effort='FFTW_MEASURE'))
            )
            print(f'k={k}/{input.shape[2]}                  ', end='\r')

        print(f"Performing FFT on {data.nxaxes[2].nxname} axis...")
        for i in range(0, input.shape[0]):
            for j in range(0, input.shape[1]):
                f_slice = fft_real[i, j, :]
                fft_real[i, j, :] = np.real(
                    np.fft.fftshift(
                        pyfftw.interfaces.numpy_fft.ifftn(np.fft.fftshift(f_slice),
                                                        planner_effort='FFTW_MEASURE')
                    )
                )
                print(f'i={i}/{input.shape[0]}                  ', end='\r')

    end = time.time()
    print(f'FFT completed in {(end - start):.3f} seconds.')

    coords = []
    for i in range(data.ndim):
        step = data.nxaxes[i].nxdata[1] - data.nxaxes[i].nxdata[0]
        axis = NXfield(np.linspace(-0.5 / step, 0.5 / step, data.shape[i]))
        coords.append(axis)

    fft = NXdata(
        NXfield(fft_real),
        tuple(coords)
    )

    # Clean up
    del fft_real, fft_array, fft_object

    gc.collect()

    return fft


class DeltaPDF:
    """
        A class for processing and analyzing 3D diffraction data using various\
        operations, including masking, interpolation, padding, and Fourier
        transformation.

        Attributes
        ----------
        fft : :class:`nexusformat.nexus.tree.NXdata` or None
            The Fourier transformed data.
        data : :class:`nexusformat.nexus.tree.NXdata` or None
            The input diffraction data.
        lattice_params : tuple or None
            Lattice parameters [e.g., (a, b, c, alpha, beta, gamma)]. These should be provided
            in the order corresponding to the axes of the relevant dataset.
        reciprocal_lattice_params : tuple or None
            Reciprocal lattice parameters [e.g., (a*, b*, c*, al*, be*, ga*)].
        puncher : Puncher
            An instance of the Puncher class for generating masks and punching
            the data.
        interpolator : Interpolator
            An instance of the Interpolator class for interpolating and applying
            windows to the data.
        padder : Padder
            An instance of the Padder class for padding the data.
        mask : :class:`numpy.ndarray` or None
            The mask used for data processing.
        kernel : Kernel or None
            The kernel used for interpolation.
        window : :class:`numpy.ndarray` or None
            The window applied to the interpolated data.
        padded : :class:`numpy.ndarray` or None
            The padded data.
        tapered : :class:`numpy.ndarray` or None
            The data after applying the window.
        interpolated : :class:`nexusformat.nexus.tree.NXdata` or None
            The interpolated data.
        punched : :class:`nexusformat.nexus.tree.NXdata` or None
            The punched data.

        """

    def __init__(self):
        """
        Initialize a DeltaPDF object with default attributes.
        """
        self.reciprocal_lattice_params = None
        self.fft = None
        self.data = None
        self.lattice_params = None
        self.puncher = Puncher()
        self.interpolator = Interpolator()
        self.padder = Padder()
        self.mask = None
        self.kernel = None
        self.window = None
        self.padded = None
        self.tapered = None
        self.interpolated = None
        self.punched = None

    def set_data(self, data):
        """
        Set the input diffraction data and update the Puncher and Interpolator
         with the data.

        Parameters
        ----------
        data : :class:`nexusformat.nexus.tree.NXdata`
            The diffraction data to be processed.
        """
        self.data = data
        self.puncher.set_data(data)
        self.interpolator.set_data(data)
        self.padder.set_data(data)
        self.tapered = data
        self.padded = data
        self.interpolated = data
        self.punched = data
        
        # Check if shapes match exactly
        if data.shape == tuple(ax.shape[0] for ax in data.nxaxes):
            axes_inputs = data.nxaxes

        # Check if shapes are 1 less than axes
        elif data.shape == tuple(ax.shape[0] - 1 for ax in data.nxaxes):
            axes_inputs = [ax[:-1] for ax in data.nxaxes]

        # Handle shape mismatch
        else:
            raise ValueError(f"Data shape {data.shape} does not match axes lengths.")

        # Assign axes
        for i, ax in enumerate(axes_inputs):
            setattr(self, f"q{i}", ax)

    def set_lattice_params(self, lattice_params):
        """
        Sets the lattice parameters and calculates the reciprocal lattice
         parameters.

        Parameters
        ----------
        lattice_params : tuple of float
            The lattice parameters [e.g., (a, b, c, alpha, beta, gamma)] in real space. These 
            should be provided in the order corresponding to the axes of the relevant dataset.
        """
        self.lattice_params = lattice_params
        self.puncher.set_lattice_params(lattice_params)
        self.reciprocal_lattice_params = self.puncher.reciprocal_lattice_params

    def add_mask(self, maskaddition):
        """
         Add regions to the current mask using a logical OR operation.

         Parameters
         ----------
         maskaddition : :class:`numpy.ndarray`
             The mask to be added.
         """
        self.puncher.add_mask(maskaddition)
        self.mask = self.puncher.mask

    def subtract_mask(self, masksubtraction):
        """
        Remove regions from the current mask using a logical AND NOT operation.

        Parameters
        ----------
        masksubtraction : :class:`numpy.ndarray`
            The mask to be subtracted.
        """
        self.puncher.subtract_mask(masksubtraction)
        self.mask = self.puncher.mask

    def generate_bragg_mask(self, punch_radius, coeffs=None, thresh=None):
        """
        Generate a mask for Bragg peaks.

        Parameters
        ----------
        punch_radius : float
            Radius for the Bragg peak mask.
        coeffs : list, optional
            Coefficients for the expression of the sphere to be removed
             around each Bragg position, corresponding to coefficients
              for H, HK, K, KL, L, and LH terms. Default is [1, 0, 1, 0, 1, 0].
        thresh : float, optional
            Intensity threshold for applying the mask.

        Returns
        -------
        mask : :class:`numpy.ndarray`
            Boolean mask identifying the Bragg peaks.
        """
        return self.puncher.generate_bragg_mask(punch_radius, coeffs, thresh)

    def generate_intensity_mask(self, thresh, radius, verbose=True):
        """
        Generate a mask based on intensity thresholds.

        Parameters
        ----------
        thresh : float
            Intensity threshold for creating the mask.
        radius : int
            Radius around high-intensity points to include in the mask.
        verbose : bool, optional
            Whether to print progress information.

        Returns
        -------
        mask : :class:`numpy.ndarray`
            Boolean mask highlighting regions with high intensity.
        """
        return self.puncher.generate_intensity_mask(thresh, radius, verbose)

    def generate_mask_at_coord(self, coordinate, punch_radius, coeffs=None, thresh=None):
        """
        Generate a mask centered at a specific coordinate.

        Parameters
        ----------
        coordinate : tuple of float
            Center coordinate for the mask.
        punch_radius : float
            Radius for the mask.
        coeffs : list, optional
            Coefficients for the expression of the ellipse/ellipsoid to be removed
            around the specific coordinate.
            For 3D: [H, HK, K, KL, L, LH] terms. Default is [1, 0, 1, 0, 1, 0].
            For 2D: [H, HK, K] terms. Default is [1, 0, 1].
        thresh : float, optional
            Intensity threshold for applying the mask.

        Returns
        -------
        mask : :class:`numpy.ndarray`
            Boolean mask for the specified coordinate.
        """
        return self.puncher.generate_mask_at_coord(coordinate, punch_radius, coeffs, thresh)

    def punch(self):
        """
        Apply the mask to the dataset, setting masked regions to NaN.

        This method creates a new dataset where the masked regions are set to
         NaN, effectively "punching" those regions.
        """
        self.puncher.punch()
        self.punched = self.puncher.punched
        self.interpolator.set_data(self.punched)

    def set_kernel(self, kernel):
        """
        Set the kernel to be used for interpolation.

        Parameters
        ----------
        kernel : :class:`numpy.ndarray`
            The kernel to be used for convolution during interpolation.
        """
        self.interpolator.set_kernel(kernel)
        self.kernel = kernel

    def interpolate(self, verbose=True, positive_values=True):
        """
        Perform interpolation on the dataset using the specified kernel.

        This method convolves the dataset with a kernel using `convolve_fft`
        to perform interpolation. The resulting interpolated data is stored
        in the `interpolated` attribute.

        Parameters
        ----------
        verbose : bool, optional
            If True, prints progress messages and timing information
            (default is True).
        positive_values : bool, optional
            If True, sets negative interpolated values to zero
            (default is True).

        Notes
        -----
        - The convolution operation is performed in Fourier space.
        - If a previous interpolation time is recorded, it is displayed
          before starting a new interpolation.

        Returns
        -------
        None
        """
        self.interpolator.interpolate(verbose, positive_values)
        self.interpolated = self.interpolator.interpolated

    def set_tukey_window(self, tukey_alphas=None):
        """
        Set a Tukey window function for data tapering.

        Parameters
        ----------
        tukey_alphas : tuple of floats, optional
            The alpha parameters for the Tukey window in each dimension.
            Defaults to 1.0 for all dimensions.

        """
        self.interpolator.set_tukey_window(tukey_alphas)
        self.window = self.interpolator.window

    def set_hexagonal_tukey_window(self, tukey_alphas=None, reverse_axes=False):
        """
        Set a hexagonal Tukey window function.

        Parameters
        ----------
        tukey_alphas : tuple of floats, optional
            The alpha parameters for the Tukey window.
            For 3D: (H, K, HK, L). Default is (1.0, 1.0, 1.0, 1.0).
            For 2D: (H, K, HK). Default is (1.0, 1.0, 1.0).
        reverse_axes : bool, optional
            If True in 3D, uses the first axis (q0) as the out-of-plane axis. 
            Default False, which uses the third axis (q2) as the out-of-plane axis.
            If True in 2D, transposes the applied hexagonal window.
        
        """
        self.interpolator.set_hexagonal_tukey_window(tukey_alphas, reverse_axes)
        self.window = self.interpolator.window

    def set_ellipsoidal_tukey_window(self, tukey_alpha=1.0, coeffs=None):
        """
        Set an ellipsoidal/elliptical Tukey window function.

        Parameters
        ----------
        tukey_alpha : float, optional
            Tapering parameter for the Tukey window, between 0 and 1.
            - `tukey_alpha = 0` results in an ellipsoidal window (no tapering).
            - `tukey_alpha = 1` results in a full cosine taper.
            Default is 1.0.

        coeffs : tuple of float, optional
            Coefficients defining the ellipsoidal quadratic form.
            For 3D (c0, c1, c2, c3, c4, c5):
                R^2 = c0*H^2 + c1*H*K + c2*K^2 + c3*K*L + c4*L^2 + c5*L*H
            For 2D (c0, c1, c2):
                R^2 = c0*H^2 + c1*H*K + c2*K^2
            If None, coefficients are automatically set to match the edges of the
            reciprocal space axes.
            
        Sets
        ----
        self.window : :class:`numpy.ndarray`
            An N-dimensional array of the same shape as the data, containing 
            the Tukey window values between 0 and 1.
        """
        self.interpolator.set_ellipsoidal_tukey_window(tukey_alpha=tukey_alpha, coeffs=coeffs)
        self.window = self.interpolator.window


    def set_window(self, window):
        """
        Set a custom window function for data tapering.

        Parameters
        ----------
        window : :class:`numpy.ndarray`
            A custom window function to be applied to the interpolated data.
        """
        self.interpolator.set_window(window)

    def apply_window(self):
        """
        Apply the window function to the interpolated data.

        The window function, if set, is applied to the `interpolated` data to
         produce the `tapered` result.

        Returns
        -------
        None
        """
        self.interpolator.apply_window()
        self.tapered = self.interpolator.tapered
        self.padder.set_data(self.tapered)

    def pad(self, padding):
        """
        Symmetrically pads the data with zero values.

        Parameters
        ----------
        padding : tuple
            The number of zero-value pixels to add along each edge of the array.

        Returns
        -------
        :class:`nexusformat.nexus.tree.NXdata`
            The padded data with symmetric zero padding.
        """
        self.padded = self.padder.pad(padding)

    def perform_fft(self, **kwargs):
        """
        Perform a 3D Fourier Transform on the padded data.

        Parameters
        ----------
        method : {'complete', 'staged'}, optional
            The FFT method to use:
            - 'complete' (default): performs a full n-dimensional FFT using pyfftw.
            - 'staged': performs a 2+1D FFT in two stages—first within the planes defined by the
            first two axes, then along the third axis. Requires 3D data and the third axis
            must be normal to the plane of the first two axes.

        verbose : bool, optional
            If True, prints progress messages during computation.

        Returns
        -------
        None

        Notes
        -----
        - Calls `fourier_transform_nxdata` to perform the transformation.
        - The output includes frequency components computed from the step
         sizes of the original data axes.

        """
        self.fft = fourier_transform_nxdata(self.padded, **kwargs)
