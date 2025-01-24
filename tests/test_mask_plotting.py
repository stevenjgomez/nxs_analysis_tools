import time
import os
import gc
import math
from scipy import ndimage
import scipy
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from nexusformat.nexus import nxsave, NXroot, NXentry, NXdata, NXfield
import numpy as np
from astropy.convolution import Kernel, convolve_fft
import pyfftw
from nxs_analysis_tools import *
from nxs_analysis_tools.datareduction import Padder, array_to_nxdata

"""
Tools for generating single crystal pair distribution functions.
"""
import time
import os
import gc
import math
from scipy import ndimage
import scipy
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from nexusformat.nexus import nxsave, NXroot, NXentry, NXdata, NXfield
import numpy as np
from astropy.convolution import Kernel, convolve_fft
import pyfftw
# from .datareduction import plot_slice, reciprocal_lattice_params, Padder, \
#     array_to_nxdata

# __all__ = ['Symmetrizer2D', 'Symmetrizer3D', 'Puncher', 'Interpolator',
#            'fourier_transform_nxdata', 'Gaussian3DKernel', 'DeltaPDF',
#            'generate_gaussian'
#            ]


class Symmetrizer2D:
    """
    A class for symmetrizing 2D datasets.

    The `Symmetrizer2D` class provides functionality to apply symmetry
    operations such as rotation and mirroring to 2D datasets.

    Attributes
    ----------
    mirror_axis : int or None
        The axis along which mirroring is performed. Default is None, meaning
        no mirroring is applied.
    symmetrized : NXdata or None
        The symmetrized dataset after applying the symmetrization operations.
        Default is None until symmetrization is performed.
    wedges : NXdata or None
        The wedges extracted from the dataset based on the angular limits.
        Default is None until symmetrization is performed.
    rotations : int or None
        The number of rotations needed to reconstruct the full dataset from
        a single wedge. Default is None until parameters are set.
    transform : Affine2D or None
        The transformation matrix used for skewing and scaling the dataset.
        Default is None until parameters are set.
    mirror : bool or None
        Indicates whether mirroring is performed during symmetrization.
        Default is None until parameters are set.
    skew_angle : float or None
        The skew angle (in degrees) between the principal axes of the plane
         to be symmetrized. Default is None until parameters are set.
    theta_max : float or None
        The maximum angle (in degrees) for symmetrization. Default is None
         until parameters are set.
    theta_min : float or None
        The minimum angle (in degrees) for symmetrization. Default is None
         until parameters are set.
    wedge : NXdata or None
        The dataset wedge used in the symmetrization process. Default is
         None until symmetrization is performed.
    symmetrization_mask : NXdata or None
        The mask used for selecting the region of the dataset to be symmetrized.
         Default is None until symmetrization is performed.

    Methods
    -------
    __init__(**kwargs):
        Initializes the Symmetrizer2D object and optionally sets the parameters
         using `set_parameters`.
    set_parameters(theta_min, theta_max, lattice_angle=90, mirror=True, mirror_axis=0):
        Sets the parameters for the symmetrization operation, including angle limits,
         lattice angle, and mirroring options.
    symmetrize_2d(data):
        Symmetrizes a 2D dataset based on the set parameters.
    test(data, **kwargs):
        Performs a test visualization of the symmetrization process, displaying the
         original data, mask, wedge, and symmetrized result.
    """
    symmetrization_mask: NXdata

    def __init__(self, **kwargs):
        """
        Initializes the Symmetrizer2D object.

        Parameters
        ----------
        **kwargs : dict, optional
            Keyword arguments that can be passed to the `set_parameters` method to
             set the symmetrization parameters during initialization.
        """
        self.mirror_axis = None
        self.symmetrized = None
        self.wedges = None
        self.rotations = None
        self.transform = None
        self.mirror = None
        self.skew_angle = None
        self.theta_max = None
        self.theta_min = None
        self.wedge = None
        if kwargs:
            self.set_parameters(**kwargs)

    def set_parameters(self, theta_min, theta_max, lattice_angle=90, mirror=True, mirror_axis=0):
        """
        Sets the parameters for the symmetrization operation, and calculates the
         required transformations and rotations.

        Parameters
        ----------
        theta_min : float
            The minimum angle in degrees for symmetrization.
        theta_max : float
            The maximum angle in degrees for symmetrization.
        lattice_angle : float, optional
            The angle in degrees between the two principal axes of the plane to be
             symmetrized (default: 90).
        mirror : bool, optional
            If True, perform mirroring during symmetrization (default: True).
        mirror_axis : int, optional
            The axis along which to perform mirroring (default: 0).
        """
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.skew_angle = lattice_angle
        self.mirror = mirror
        self.mirror_axis = mirror_axis

        # Define Transformation
        skew_angle_adj = 90 - lattice_angle
        t = Affine2D()
        # Scale y-axis to preserve norm while shearing
        t += Affine2D().scale(1, np.cos(skew_angle_adj * np.pi / 180))
        # Shear along x-axis
        t += Affine2D().skew_deg(skew_angle_adj, 0)
        # Return to original y-axis scaling
        t += Affine2D().scale(1, np.cos(skew_angle_adj * np.pi / 180)).inverted()
        self.transform = t

        # Calculate number of rotations needed to reconstruct the dataset
        if mirror:
            rotations = abs(int(360 / (theta_max - theta_min) / 2))
        else:
            rotations = abs(int(360 / (theta_max - theta_min)))
        self.rotations = rotations

        self.symmetrization_mask = None

        self.wedges = None

        self.symmetrized = None

    def symmetrize_2d(self, data):
        """
        Symmetrizes a 2D dataset based on the set parameters, applying padding
         to prevent rotation cutoff and handling overlapping pixels.

        Parameters
        ----------
        data : NXdata
            The input 2D dataset to be symmetrized.

        Returns
        -------
        symmetrized : NXdata
            The symmetrized 2D dataset.
        """
        theta_min = self.theta_min
        theta_max = self.theta_max
        mirror = self.mirror
        mirror_axis = self.mirror_axis
        t = self.transform
        rotations = self.rotations

        # Pad the dataset so that rotations don't get cutoff if they extend
        # past the extent of the dataset
        p = Padder(data)
        padding = tuple(len(data[axis]) for axis in data.axes)
        data_padded = p.pad(padding)

        # Define axes that span the plane to be transformed
        q1 = data_padded[data.axes[0]]
        q2 = data_padded[data.axes[1]]

        # Calculate the angle for each data point
        theta = np.arctan2(q1.reshape((-1, 1)), q2.reshape((1, -1)))
        # Create a boolean array for the range of angles
        symmetrization_mask = np.logical_and(theta >= theta_min * np.pi / 180,
                                             theta <= theta_max * np.pi / 180)

        # Define signal to be transformed
        counts = symmetrization_mask

        # Scale and skew counts
        skew_angle_adj = 90 - self.skew_angle
        counts_skew = ndimage.affine_transform(counts,
                                               t.get_matrix()[:2, :2],
                                               offset=[-counts.shape[0] / 2
                                                       * np.sin(skew_angle_adj * np.pi / 180), 0],
                                               order=0,
                                               )
        scale1 = np.cos(skew_angle_adj * np.pi / 180)
        mask = ndimage.affine_transform(counts_skew,
                                         Affine2D().scale(scale1, 1).inverted().get_matrix()[:2, :2],
                                         offset=[-(1 - scale1) * counts.shape[0] / 2, 0],
                                         order=0,
                                         )

        scale2 = counts.shape[0] / counts.shape[1]
        mask = ndimage.affine_transform(mask,
                                         Affine2D().scale(scale2, 1).inverted().get_matrix()[:2, :2],
                                         offset=[-(1 - scale2) * counts.shape[0] / 2, 0],
                                         order=0,
                                         )

        # Convert mask to nxdata
        mask = array_to_nxdata(mask, data_padded)

        # Save mask for user interaction
        self.symmetrization_mask = p.unpad(mask)

        # Perform masking
        wedge = mask*data_padded

        # Save wedge for user interaction
        self.wedge = p.unpad(wedge)

        # Convert wedge back to array for further transformations
        wedge = wedge[data.signal].nxdata

        # Define signal to be transformed
        counts = wedge

        # Scale and skew counts
        skew_angle_adj = 90 - self.skew_angle
        counts_skew = ndimage.affine_transform(counts,
                                               t.inverted().get_matrix()[:2, :2],
                                               offset=[counts.shape[0] / 2
                                                       * np.sin(skew_angle_adj * np.pi / 180), 0],
                                               order=0,
                                               )
        scale1 = np.cos(skew_angle_adj * np.pi / 180)
        wedge = ndimage.affine_transform(counts_skew,
                                        Affine2D().scale(scale1, 1).get_matrix()[:2, :2],
                                        offset=[(1 - scale1) * counts.shape[0] / 2, 0],
                                        order=0,
                                        )

        scale2 = counts.shape[0] / counts.shape[1]
        wedge = ndimage.affine_transform(wedge,
                                        Affine2D().scale(scale2, 1).get_matrix()[:2, :2],
                                        offset=[(1 - scale2) * counts.shape[0] / 2, 0],
                                        order=0,
                                        )

        # Reconstruct full dataset from wedge
        reconstructed = np.zeros(counts.shape)
        for _ in range(0, rotations):
            # The following are attempts to combine images with minimal overlapping pixels
            reconstructed += wedge
            # reconstructed = np.where(reconstructed == 0, reconstructed + wedge, reconstructed)

            wedge = ndimage.rotate(wedge, 360 / rotations, reshape=False, order=0)

        # self.rotated_only = NXdata(NXfield(reconstructed, name=data.signal),
        #                            (q1, q2))

        if mirror:
            # The following are attempts to combine images with minimal overlapping pixels
            reconstructed = np.where(reconstructed == 0,
                                     reconstructed + np.flip(reconstructed, axis=mirror_axis),
                                     reconstructed)
            # reconstructed += np.flip(reconstructed, axis=0)

        # self.rotated_and_mirrored = NXdata(NXfield(reconstructed, name=data.signal),
        #                                    (q1, q2))

        reconstructed = ndimage.affine_transform(reconstructed,
                                                 Affine2D().scale(
                                                     scale2, 1
                                                 ).inverted().get_matrix()[:2, :2],
                                                 offset=[-(1 - scale2) * counts.shape[
                                                     0] / 2 / scale2, 0],
                                                 order=0,
                                                 )
        reconstructed = ndimage.affine_transform(reconstructed,
                                                 Affine2D().scale(
                                                     scale1, 1
                                                 ).inverted().get_matrix()[:2, :2],
                                                 offset=[-(1 - scale1) * counts.shape[
                                                     0] / 2 / scale1, 0],
                                                 order=0,
                                                 )
        reconstructed = ndimage.affine_transform(reconstructed,
                                                 t.get_matrix()[:2, :2],
                                                 offset=[(-counts.shape[0] / 2
                                                          * np.sin(skew_angle_adj * np.pi / 180)),
                                                         0],
                                                 order=0,
                                                 )

        reconstructed_unpadded = p.unpad(reconstructed)

        # Fix any overlapping pixels by truncating counts to max
        reconstructed_unpadded[reconstructed_unpadded > data[data.signal].nxdata.max()] \
            = data[data.signal].nxdata.max()

        symmetrized = NXdata(NXfield(reconstructed_unpadded, name=data.signal),
                             (data[data.axes[0]],
                              data[data.axes[1]]))

        return symmetrized

    def test(self, data, **kwargs):
        """
        Performs a test visualization of the symmetrization process to help assess
         the effect of the parameters.

        Parameters
        ----------
        data : ndarray
            The input 2D dataset to be used for the test visualization.
        **kwargs : dict
            Additional keyword arguments to be passed to the plot_slice function.

        Returns
        -------
        fig : Figure
            The matplotlib Figure object that contains the test visualization plot.
        axesarr : ndarray
            The numpy array of Axes objects representing the subplots in the test
             visualization.

        Notes
        -----
        This method uses the `symmetrize_2d` method to perform the symmetrization on
         the input data and visualize the process.

        The test visualization plot includes the following subplots:
        - Subplot 1: The original dataset.
        - Subplot 2: The symmetrization mask.
        - Subplot 3: The wedge slice used for reconstruction of the full symmetrized dataset.
        - Subplot 4: The symmetrized dataset.

        Example usage:
        ```
        s = Symmetrizer2D()
        s.set_parameters(theta_min, theta_max, skew_angle, mirror)
        s.test(data)
        ```
        """
        s = self
        symm_test = s.symmetrize_2d(data)
        fig, axesarr = plt.subplots(2, 2, figsize=(10, 8))
        axes = axesarr.reshape(-1)
        plot_slice(data, skew_angle=s.skew_angle, ax=axes[0], title='data', **kwargs)
        plot_slice(s.symmetrization_mask, skew_angle=s.skew_angle, ax=axes[1], title='mask')
        plot_slice(s.wedge, skew_angle=s.skew_angle, ax=axes[2], title='wedge', **kwargs)
        plot_slice(symm_test, skew_angle=s.skew_angle, ax=axes[3], title='symmetrized', **kwargs)
        plt.subplots_adjust(wspace=0.4)
        plt.show()
        return fig, axesarr


data = load_transform(r'K:\wilson-3947-a\nxrefine\LaCd3P3\MLA1\LaCd3P3_300.nxs')[:,:,7.9:8.1]
data = rotate_data(data, lattice_angle=60, rotation_angle=120, rotation_axis=2, printout=True)

# from nxs_analysis_tools.pairdistribution import Symmetrizer2D
s2d = Symmetrizer2D(theta_min=-90, theta_max=-90+60, mirror=True, lattice_angle=60, mirror_axis=1)
s2d.test(data[:,:,8.0], vmin=0, vmax=100, xlim=(-3,2))