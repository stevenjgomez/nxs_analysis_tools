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
from .datareduction import plot_slice, reciprocal_lattice_params, Padder, \
    array_to_nxdata

__all__ = ['Symmetrizer2D', 'Symmetrizer3D', 'Puncher', 'Interpolator',
           'fourier_transform_nxdata', 'Gaussian3DKernel', 'DeltaPDF',
           'generate_gaussian'
           ]


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

        scale2 = 1 # q1.max()/q2.max() # TODO: Need to double check this
        counts_unscaled2 = ndimage.affine_transform(counts,
                                                    Affine2D().scale(scale2, 1).inverted().get_matrix()[:2, :2],
                                                    offset=[-(1 - scale2) * counts.shape[
                                                        0] / 2 / scale2, 0],
                                                    order=0,
                                                    )

        scale1 = np.cos(skew_angle_adj * np.pi / 180)
        counts_unscaled1 = ndimage.affine_transform(counts_unscaled2,
                                                    Affine2D().scale(scale1, 1).inverted().get_matrix()[:2, :2],
                                                    offset=[-(1 - scale1) * counts.shape[
                                                        0] / 2 / scale1, 0],
                                                    order=0,
                                                    )

        mask = ndimage.affine_transform(counts_unscaled1,
                                        t.get_matrix()[:2, :2],
                                        offset=[-counts.shape[0] / 2
                                                * np.sin(skew_angle_adj * np.pi / 180), 0],
                                        order=0,
                                        )

        # Convert mask to nxdata
        mask = array_to_nxdata(mask, data_padded)

        # Save mask for user interaction
        self.symmetrization_mask = p.unpad(mask)

        # Perform masking
        wedge = mask * data_padded

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

        scale2 = counts.shape[0]/counts.shape[1]
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

        # Plot the data
        plot_slice(data, skew_angle=s.skew_angle, ax=axes[0], title='data', **kwargs)

        # Filter kwargs to exclude 'vmin' and 'vmax'
        filtered_kwargs = {key: value for key, value in kwargs.items() if key not in ('vmin', 'vmax')}
        # Plot the mask
        plot_slice(s.symmetrization_mask, skew_angle=s.skew_angle, ax=axes[1], title='mask', **filtered_kwargs)

        # Plot the wedge
        plot_slice(s.wedge, skew_angle=s.skew_angle, ax=axes[2], title='wedge', **kwargs)

        # Plot the symmetrized data
        plot_slice(symm_test, skew_angle=s.skew_angle, ax=axes[3], title='symmetrized', **kwargs)
        plt.subplots_adjust(wspace=0.4)
        plt.show()
        return fig, axesarr

class Symmetrizer3D:
    """
    A class to symmetrize 3D datasets by performing sequential 2D symmetrization on
     different planes.

    This class applies 2D symmetrization on the three principal planes of a 3D dataset,
    effectively enhancing the symmetry of the data across all axes.
    """

    def __init__(self, data=None):
        """
        Initialize the Symmetrizer3D object with an optional 3D dataset.

        If data is provided, the corresponding q-vectors and planes are automatically
         set up for symmetrization.

        Parameters
        ----------
        data : NXdata, optional
            The input 3D dataset to be symmetrized.
        """

        assert data is not None, "Symmetrizer3D requires a 3D NXdata object for initialization."

        self.a, self.b, self.c, self.al, self.be, self.ga = [None] * 6
        self.a_star, self.b_star, self.c_star, self.al_star, self.be_star, self.ga_star = [None] * 6
        self.lattice_params = None
        self.reciprocal_lattice_params = None
        self.symmetrized = None
        self.data = data
        self.plane1symmetrizer = Symmetrizer2D()
        self.plane2symmetrizer = Symmetrizer2D()
        self.plane3symmetrizer = Symmetrizer2D()

        if data is not None:
            self.q1 = data[data.axes[0]]
            self.q2 = data[data.axes[1]]
            self.q3 = data[data.axes[2]]
            self.plane1 = self.q1.nxname + self.q2.nxname
            self.plane2 = self.q1.nxname + self.q3.nxname
            self.plane3 = self.q2.nxname + self.q3.nxname

        print("Plane 1: " + self.plane1)
        print("Plane 2: " + self.plane2)
        print("Plane 3: " + self.plane3)

    def set_data(self, data):
        """
        Sets the 3D dataset to be symmetrized and updates the corresponding q-vectors and planes.

        Parameters
        ----------
        data : NXdata
            The input 3D dataset to be symmetrized.
        """
        self.data = data
        self.q1 = data[data.axes[0]]
        self.q2 = data[data.axes[1]]
        self.q3 = data[data.axes[2]]
        self.plane1 = self.q1.nxname + self.q2.nxname
        self.plane2 = self.q1.nxname + self.q3.nxname
        self.plane3 = self.q2.nxname + self.q3.nxname

        print("Plane 1: " + self.plane1)
        print("Plane 2: " + self.plane2)
        print("Plane 3: " + self.plane3)

    def set_lattice_params(self, lattice_params):
        """
        Sets the lattice parameters and calculates the reciprocal lattice parameters.

        Parameters
        ----------
        lattice_params : tuple of float
            The lattice parameters (a, b, c, alpha, beta, gamma) in real space.
        """
        self.a, self.b, self.c, self.al, self.be, self.ga = lattice_params
        self.lattice_params = lattice_params
        self.reciprocal_lattice_params = reciprocal_lattice_params(lattice_params)
        self.a_star, self.b_star, self.c_star, \
            self.al_star, self.be_star, self.ga_star = self.reciprocal_lattice_params

    def symmetrize(self, positive_values=True):
        """
        Symmetrize the 3D dataset by sequentially applying 2D symmetrization
        on the three principal planes.

        This method performs symmetrization along the (q1-q2), (q1-q3),
        and (q2-q3) planes, ensuring that the dataset maintains expected
        symmetry properties. Optionally, negative values resulting from the
        symmetrization process can be set to zero.

        Parameters
        ----------
        positive_values : bool, optional
            If True, sets negative symmetrized values to zero (default is True).

        Returns
        -------
        NXdata
            The symmetrized 3D dataset stored in the `symmetrized` attribute.

        Notes
        -----
        - Symmetrization is performed sequentially across three principal
          planes using corresponding 2D symmetrization methods.
        - The process prints progress updates and timing information.
        - If `theta_max` is not set for a particular plane symmetrizer,
          that plane is skipped.
        """

        starttime = time.time()
        data = self.data
        q1, q2, q3 = self.q1, self.q2, self.q3
        out_array = np.zeros(data[data.signal].shape)

        if self.plane1symmetrizer.theta_max is not None:
            print('Symmetrizing ' + self.plane1 + ' planes...')
            for k, value in enumerate(q3):
                print(f'Symmetrizing {q3.nxname}={value:.02f}.'
                      f'..', end='\r')
                data_symmetrized = self.plane1symmetrizer.symmetrize_2d(data[:, :, k])
                out_array[:, :, k] = data_symmetrized[data.signal].nxdata
            print('\nSymmetrized ' + self.plane1 + ' planes.')

        if self.plane2symmetrizer.theta_max is not None:
            print('Symmetrizing ' + self.plane2 + ' planes...')
            for j, value in enumerate(q2):
                print(f'Symmetrizing {q2.nxname}={value:.02f}...', end='\r')
                data_symmetrized = self.plane2symmetrizer.symmetrize_2d(
                    NXdata(NXfield(out_array[:, j, :], name=data.signal), (q1, q3))
                )
                out_array[:, j, :] = data_symmetrized[data.signal].nxdata
            print('\nSymmetrized ' + self.plane2 + ' planes.')

        if self.plane3symmetrizer.theta_max is not None:
            print('Symmetrizing ' + self.plane3 + ' planes...')
            for i, value in enumerate(q1):
                print(f'Symmetrizing {q1.nxname}={value:.02f}...', end='\r')
                data_symmetrized = self.plane3symmetrizer.symmetrize_2d(
                    NXdata(NXfield(out_array[i, :, :], name=data.signal), (q2, q3))
                )
                out_array[i, :, :] = data_symmetrized[data.signal].nxdata
            print('\nSymmetrized ' + self.plane3 + ' planes.')

        if positive_values:
            out_array[out_array < 0] = 0

        stoptime = time.time()
        print(f"\nSymmetrization finished in {((stoptime - starttime) / 60):.02f} minutes.")

        self.symmetrized = NXdata(NXfield(out_array, name=data.signal),
                                  tuple(data[axis] for axis in data.axes))

        return self.symmetrized

    def save(self, fout_name=None):
        """
        Save the symmetrized dataset to a NeXus file.

        Parameters
        ----------
        fout_name : str, optional
            The name of the output file. If not provided,
            the default name 'symmetrized.nxs' will be used.
        """
        print("Saving file...")

        f = NXroot()
        f['entry'] = NXentry()
        f['entry']['data'] = self.symmetrized
        if fout_name is None:
            fout_name = 'symmetrized.nxs'
        nxsave(fout_name, f)
        print("Output file saved to: " + os.path.join(os.getcwd(), fout_name))


def generate_gaussian(H, K, L, amp, stddev, lattice_params, coeffs=None, center=None):
    """
    Generate a 3D Gaussian distribution.

    This function creates a 3D Gaussian distribution in reciprocal space based
     on the specified parameters.

    Parameters
    ----------
    H, K, L : ndarray
        Arrays specifying the values of H, K, and L coordinates in reciprocal space.
    amp : float
        Amplitude of the Gaussian distribution.
    stddev : float
        Standard deviation of the Gaussian distribution.
    lattice_params : tuple
        Tuple of lattice parameters (a, b, c, alpha, beta, gamma) for the
         reciprocal lattice.
    coeffs : list, optional
        Coefficients for the Gaussian expression, including cross-terms between axes.
         Default is [1, 0, 1, 0, 1, 0],
         corresponding to (1*H**2 + 0*H*K + 1*K**2 + 0*K*L + 1*L**2 + 0*L*H).
    center : tuple
        Tuple of coordinates for the center of the Gaussian. Default is (0,0,0).

    Returns
    -------
    gaussian : ndarray
        3D Gaussian distribution array.
    """
    if center is None:
        center=(0,0,0)
    if coeffs is None:
        coeffs = [1, 0, 1, 0, 1, 0]
    a, b, c, al, be, ga = lattice_params
    a_, b_, c_, _, _, _ = reciprocal_lattice_params((a, b, c, al, be, ga))
    H = H-center[0]
    K = K-center[1]
    L = L-center[2]
    H, K, L = np.meshgrid(H, K, L, indexing='ij')
    gaussian = amp * np.exp(-(coeffs[0] * H ** 2 +
                              coeffs[1] * (b_ * a_ / (a_ ** 2)) * H * K +
                              coeffs[2] * (b_ / a_) ** 2 * K ** 2 +
                              coeffs[3] * (b_ * c_ / (a_ ** 2)) * K * L +
                              coeffs[4] * (c_ / a_) ** 2 * L ** 2 +
                              coeffs[5] * (c_ * a_ / (a_ ** 2)) * L * H) / (2 * stddev ** 2))
    if gaussian.ndim == 3:
        gaussian = gaussian.transpose(1, 0, 2)
    elif gaussian.ndim == 2:
        gaussian = gaussian.transpose()
    return gaussian.transpose(1, 0, 2)

class Puncher:
    """
    A class for applying masks to 3D datasets, typically for data processing in reciprocal space.

    This class provides methods for setting data, applying masks, and generating
     masks based on various criteria. It can be used to "punch" or modify datasets
      by setting specific regions to NaN according to the mask.

    Attributes
    ----------
    punched : NXdata, optional
        The dataset with regions modified (punched) based on the mask.
    data : NXdata, optional
        The input dataset to be processed.
    HH, KK, LL : ndarray
        Meshgrid arrays representing the H, K, and L coordinates in reciprocal space.
    mask : ndarray, optional
        The mask used for identifying and modifying specific regions in the dataset.
    reciprocal_lattice_params : tuple, optional
        The reciprocal lattice parameters derived from the lattice parameters.
    lattice_params : tuple, optional
        The lattice parameters (a, b, c, alpha, beta, gamma).
    a, b, c, al, be, ga : float
        Individual lattice parameters.
    a_star, b_star, c_star, al_star, be_star, ga_star : float
        Individual reciprocal lattice parameters.

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
        Generates a mask for Bragg peaks based on a Gaussian distribution in
         reciprocal space.
    generate_intensity_mask(thresh, radius, verbose=True)
        Generates a mask based on intensity thresholds, including a spherical
         region around high-intensity points.
    generate_mask_at_coord(coordinate, punch_radius, coeffs=None, thresh=None)
        Generates a mask centered at a specific coordinate in reciprocal space
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
        punched : NXdata, optional
            The dataset with modified (punched) regions, initialized as None.
        data : NXdata, optional
            The input dataset to be processed, initialized as None.
        HH, KK, LL : ndarray, optional
            Arrays representing the H, K, and L coordinates in reciprocal space,
             initialized as None.
        mask : ndarray, optional
            The mask for identifying and modifying specific regions in the dataset,
             initialized as None.
        reciprocal_lattice_params : tuple, optional
            The reciprocal lattice parameters, initialized as None.
        lattice_params : tuple, optional
            The lattice parameters (a, b, c, alpha, beta, gamma),
             initialized as None.
        a, b, c, al, be, ga : float
            Individual lattice parameters, initialized as None.
        a_star, b_star, c_star, al_star, be_star, ga_star : float
            Individual reciprocal lattice parameters, initialized as None.
        """
        self.punched = None
        self.data = None
        self.HH, self.KK, self.LL = [None] * 3
        self.mask = None
        self.reciprocal_lattice_params = None
        self.lattice_params = None
        self.a, self.b, self.c, self.al, self.be, self.ga = [None] * 6
        self.a_star, self.b_star, self.c_star, self.al_star, self.be_star, self.ga_star = [None] * 6

    def set_data(self, data):
        """
        Set the 3D dataset and initialize the mask if not already set.

        Parameters
        ----------
        data : NXdata
            The dataset to be processed.

        Notes
        -----
        This method also sets up the H, K, and L coordinate grids for the dataset.
        """
        self.data = data
        if self.mask is None:
            self.mask = np.zeros(data[data.signal].nxdata.shape)
        self.HH, self.KK, self.LL = np.meshgrid(data[data.axes[0]],
                                                data[data.axes[1]],
                                                data[data.axes[2]],
                                                indexing='ij')

    def set_lattice_params(self, lattice_params):
        """
        Set the lattice parameters and compute the reciprocal lattice parameters.

        Parameters
        ----------
        lattice_params : tuple
            Tuple of lattice parameters (a, b, c, alpha, beta, gamma).
        """
        self.a, self.b, self.c, self.al, self.be, self.ga = lattice_params
        self.lattice_params = lattice_params
        self.reciprocal_lattice_params = reciprocal_lattice_params(lattice_params)
        self.a_star, self.b_star, self.c_star, \
            self.al_star, self.be_star, self.ga_star = self.reciprocal_lattice_params

    def add_mask(self, maskaddition):
        """
        Add regions to the current mask using a logical OR operation.

        Parameters
        ----------
        maskaddition : ndarray
            The mask to be added.
        """
        self.mask = np.logical_or(self.mask, maskaddition)

    def subtract_mask(self, masksubtraction):
        """
        Remove regions from the current mask using a logical AND NOT operation.

        Parameters
        ----------
        masksubtraction : ndarray
            The mask to be subtracted.
        """
        self.mask = np.logical_and(self.mask, np.logical_not(masksubtraction))

    def generate_bragg_mask(self, punch_radius, coeffs=None, thresh=None):
        """
        Generate a mask for Bragg peaks.

        Parameters
        ----------
        punch_radius : float
            Radius for the Bragg peak mask.
        coeffs : list, optional
            Coefficients for the expression of the sphere to be removed around
            each Bragg position, corresponding to coefficients for H, HK, K, KL, L, and LH terms.
            Default is [1, 0, 1, 0, 1, 0].
        thresh : float, optional
            Intensity threshold for applying the mask.

        Returns
        -------
        mask : ndarray
            Boolean mask identifying the Bragg peaks.
        """
        if coeffs is None:
            coeffs = [1, 0, 1, 0, 1, 0]
        data = self.data
        H, K, L = self.HH, self.KK, self.LL
        a_, b_, c_, _, _, _ = self.reciprocal_lattice_params

        mask = (coeffs[0] * (H - np.rint(H)) ** 2 +
                coeffs[1] * (b_ * a_ / (a_ ** 2)) * (H - np.rint(H)) * (K - np.rint(K)) +
                coeffs[2] * (b_ / a_) ** 2 * (K - np.rint(K)) ** 2 +
                coeffs[3] * (b_ * c_ / (a_ ** 2)) * (K - np.rint(K)) * (L - np.rint(L)) +
                coeffs[4] * (c_ / a_) ** 2 * (L - np.rint(L)) ** 2 +
                coeffs[5] * (c_ * a_ / (a_ ** 2)) * (L - np.rint(L)) * (H - np.rint(H))) \
               < punch_radius ** 2

        if thresh:
            mask = np.logical_and(mask, data[data.signal] > thresh)

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
        mask : ndarray
            Boolean mask highlighting regions with high intensity.
        """
        data = self.data
        counts = data[data.signal].nxdata
        mask = np.zeros(counts.shape)

        print(f"Shape of data is {counts.shape}") if verbose else None
        for i in range(counts.shape[0]):
            for j in range(counts.shape[1]):
                for k in range(counts.shape[2]):
                    if counts[i, j, k] > thresh:
                        # Set the pixels within the sphere to NaN
                        for x in range(max(i - radius, 0),
                                       min(i + radius + 1, counts.shape[0])):
                            for y in range(max(j - radius, 0),
                                           min(j + radius + 1, counts.shape[1])):
                                for z in range(max(k - radius, 0),
                                               min(k + radius + 1, counts.shape[2])):
                                    mask[x, y, z] = 1
                        print(f"Found high intensity at ({i}, {j}, {k}).\t\t", end='\r') \
                            if verbose else None
        print("\nDone.")
        return mask

    def generate_mask_at_coord(self, coordinate, punch_radius, coeffs=None, thresh=None):
        """
        Generate a mask centered at a specific coordinate.

        Parameters
        ----------
        coordinate : tuple of float
            Center coordinate (H, K, L) for the mask.
        punch_radius : float
            Radius for the mask.
        coeffs : list, optional
            Coefficients for the expression of the sphere to be removed around
            each Bragg position,
            corresponding to coefficients for H, HK, K, KL, L, and LH terms.
            Default is [1, 0, 1, 0, 1, 0].
        thresh : float, optional
            Intensity threshold for applying the mask.

        Returns
        -------
        mask : ndarray
            Boolean mask for the specified coordinate.
        """
        if coeffs is None:
            coeffs = [1, 0, 1, 0, 1, 0]
        data = self.data
        H, K, L = self.HH, self.KK, self.LL
        a_, b_, c_, _, _, _ = self.reciprocal_lattice_params
        centerH, centerK, centerL = coordinate
        mask = (coeffs[0] * (H - centerH) ** 2 +
                coeffs[1] * (b_ * a_ / (a_ ** 2)) * (H - centerH) * (K - centerK) +
                coeffs[2] * (b_ / a_) ** 2 * (K - centerK) ** 2 +
                coeffs[3] * (b_ * c_ / (a_ ** 2)) * (K - centerK) * (L - centerL) +
                coeffs[4] * (c_ / a_) ** 2 * (L - centerL) ** 2 +
                coeffs[5] * (c_ * a_ / (a_ ** 2)) * (L - centerL) * (H - centerH)) \
               < punch_radius ** 2

        if thresh:
            mask = np.logical_and(mask, data[data.signal] > thresh)

        return mask

    def punch(self):
        """
        Apply the mask to the dataset, setting masked regions to NaN.

        This method creates a new dataset where the masked regions are set to
         NaN, effectively "punching" those regions.
        """
        data = self.data
        self.punched = NXdata(NXfield(
            np.where(self.mask, np.nan, data[data.signal].nxdata),
            name=data.signal),
            (data[data.axes[0]], data[data.axes[1]], data[data.axes[2]])
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
        X, Y, Z = np.meshgrid(x, y, z)
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
     kernel.

    Attributes
    ----------
    interp_time : float or None
        Time taken for the last interpolation operation. Defaults to None.

    window : ndarray or None
        Window function to be applied to the interpolated data. Defaults to None.

    interpolated : ndarray or None
        The result of the interpolation operation. Defaults to None.

    data : NXdata or None
        The dataset to be interpolated. Defaults to None.

    kernel : ndarray or None
        The kernel used for convolution during interpolation. Defaults to None.

    tapered : ndarray or None
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
        data : NXdata
            The dataset containing the data to be interpolated.
        """
        self.data = data

    def set_kernel(self, kernel):
        """
        Set the kernel to be used for interpolation.

        Parameters
        ----------
        kernel : ndarray
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
            convolve_fft(self.data[self.data.signal].nxdata,
                         self.kernel, allow_huge=True, return_fft=False))
        print("Interpolation finished.") if verbose else None

        end = time.time()
        interp_time = end - start

        print(f'Interpolation took {interp_time / 60:.2f} minutes.') if verbose else None

        if positive_values:
            result[result < 0] = 0
        self.interpolated = array_to_nxdata(result, self.data)

    def set_tukey_window(self, tukey_alphas=(1.0, 1.0, 1.0)):
        """
        Set a Tukey window function for data tapering.

        Parameters
        ----------
        tukey_alphas : tuple of floats, optional
            The alpha parameters for the Tukey window in each
             dimension (H, K, L). Default is (1.0, 1.0, 1.0).

        Notes
        -----
        The window function is generated based on the size of the dataset in each dimension.
        """
        data = self.data
        tukey_H = np.tile(
            scipy.signal.windows.tukey(len(data[data.axes[0]]), alpha=tukey_alphas[0])[:, None, None],
            (1, len(data[data.axes[1]]), len(data[data.axes[2]]))
        )
        tukey_K = np.tile(
            scipy.signal.windows.tukey(len(data[data.axes[1]]), alpha=tukey_alphas[1])[None, :, None],
            (len(data[data.axes[0]]), 1, len(data[data.axes[2]]))
        )
        window = tukey_H * tukey_K

        del tukey_H, tukey_K
        gc.collect()

        tukey_L = np.tile(
            scipy.signal.windows.tukey(len(data[data.axes[2]]), alpha=tukey_alphas[2])[None, None, :],
            (len(data[data.axes[0]]), len(data[data.axes[1]]), 1))
        window = window * tukey_L

        self.window = window

    def set_hexagonal_tukey_window(self, tukey_alphas=(1.0, 1.0, 1.0, 1.0)):
        """
        Set a hexagonal Tukey window function for data tapering.

        Parameters
        ----------
        tukey_alphas : tuple of floats, optional
            The alpha parameters for the Tukey window in each dimension and
            for the hexagonal truncation (H, HK, K, L).
            Default is (1.0, 1.0, 1.0, 1.0).

        Notes
        -----
        The hexagonal Tukey window is applied to the dataset in a manner that
        preserves hexagonal symmetry.

        """
        data = self.data
        H_ = data[data.axes[0]]
        K_ = data[data.axes[1]]
        L_ = data[data.axes[2]]

        tukey_H = np.tile(
            scipy.signal.windows.tukey(len(data[data.axes[0]]), alpha=tukey_alphas[0])[:, None, None],
            (1, len(data[data.axes[1]]), len(data[data.axes[2]]))
        )
        tukey_K = np.tile(
            scipy.signal.windows.tukey(len(data[data.axes[1]]), alpha=tukey_alphas[1])[None, :, None],
            (len(data[data.axes[0]]), 1, len(data[data.axes[2]]))
        )
        window = tukey_H * tukey_K

        del tukey_H, tukey_K
        gc.collect()

        truncation = int((len(H_) - int(len(H_) * np.sqrt(2) / 2)) / 2)

        tukey_HK = scipy.ndimage.rotate(
            np.tile(
                np.concatenate(
                    (np.zeros(truncation)[:, None, None],
                     scipy.signal.windows.tukey(len(H_) - 2 * truncation,
                                        alpha=tukey_alphas[2])[:, None, None],
                     np.zeros(truncation)[:, None, None])),
                (1, len(K_), len(L_))
            ),
            angle=45, reshape=False, mode='nearest',
        )[0:len(H_), 0:len(K_), :]
        tukey_HK = np.nan_to_num(tukey_HK)
        window = window * tukey_HK

        del tukey_HK
        gc.collect()

        tukey_L = np.tile(
            scipy.signal.windows.tukey(len(data[data.axes[2]]), alpha=tukey_alphas[3])[None, None, :],
            (len(data[data.axes[0]]), len(data[data.axes[1]]), 1)
        )
        window = window * tukey_L

        del tukey_L
        gc.collect()

        self.window = window

    def set_window(self, window):
        """
        Set a custom window function for data tapering.

        Parameters
        ----------
        window : ndarray
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


def fourier_transform_nxdata(data, is_2d=False):
    """
    Perform a 3D Fourier Transform on the given NXdata object.

    This function applies an inverse Fourier Transform to the input data
    using the `pyfftw` library to optimize performance. The result is a
    transformed array with spatial frequency components calculated along
    each axis.

    Parameters
    ----------
    data : NXdata
        An NXdata object containing the data to be transformed. It should
        include the `signal` field for the data and `axes` fields
        specifying the coordinate axes.

    is_2d : bool
        If true, skip FFT on out-of-plane direction and only do FFT
        on axes 0 and 1. Default False.

    Returns
    -------
    NXdata
        A new NXdata object containing the Fourier Transformed data. The
        result includes:

        - `dPDF`: The transformed data array.
        - `x`, `y`, `z`: Arrays representing the real-space components along each axis.

    Notes
    -----
    - The FFT is performed in two stages: first along the last dimension of the input array and then along the first two dimensions.
    - The function uses `pyfftw` for efficient computation of the Fourier Transform.
    - The output frequency components are computed based on the step sizes of the original data axes.

    """
    start = time.time()
    print("Starting FFT.")

    padded = data[data.signal].nxdata

    fft_array = np.zeros(padded.shape)

    print("FFT on axes 1,2")

    for k in range(0, padded.shape[2]):
        fft_array[:, :, k] = np.real(
            np.fft.fftshift(
                pyfftw.interfaces.numpy_fft.ifftn(np.fft.fftshift(padded[:, :, k]),
                                                  planner_effort='FFTW_MEASURE'))
        )
        print(f'k={k}                  ', end='\r')

    if not is_2d:
        print("FFT on axis 3")
        for i in range(0, padded.shape[0]):
            for j in range(0, padded.shape[1]):
                f_slice = fft_array[i, j, :]
                print(f'i={i}                  ', end='\r')
                fft_array[i, j, :] = np.real(
                    np.fft.fftshift(
                        pyfftw.interfaces.numpy_fft.ifftn(np.fft.fftshift(f_slice),
                                                          planner_effort='FFTW_MEASURE')
                    )
                )

    end = time.time()
    print("FFT complete.")
    print('FFT took ' + str(end - start) + ' seconds.')

    H_step = data[data.axes[0]].nxdata[1] - data[data.axes[0]].nxdata[0]
    K_step = data[data.axes[1]].nxdata[1] - data[data.axes[1]].nxdata[0]
    L_step = data[data.axes[2]].nxdata[1] - data[data.axes[2]].nxdata[0]

    fft = NXdata(NXfield(fft_array, name='dPDF'),
                 (NXfield(np.linspace(-0.5 / H_step, 0.5 / H_step, padded.shape[0]), name='x'),
                  NXfield(np.linspace(-0.5 / K_step, 0.5 / K_step, padded.shape[1]), name='y'),
                  NXfield(np.linspace(-0.5 / L_step, 0.5 / L_step, padded.shape[2]), name='z'),
                  )
                 )
    return fft


class DeltaPDF:
    """
        A class for processing and analyzing 3D diffraction data using various\
        operations, including masking, interpolation, padding, and Fourier
        transformation.

        Attributes
        ----------
        fft : NXdata or None
            The Fourier transformed data.
        data : NXdata or None
            The input diffraction data.
        lattice_params : tuple or None
            Lattice parameters (a, b, c, al, be, ga).
        reciprocal_lattice_params : tuple or None
            Reciprocal lattice parameters (a*, b*, c*, al*, be*, ga*).
        puncher : Puncher
            An instance of the Puncher class for generating masks and punching
            the data.
        interpolator : Interpolator
            An instance of the Interpolator class for interpolating and applying
            windows to the data.
        padder : Padder
            An instance of the Padder class for padding the data.
        mask : ndarray or None
            The mask used for data processing.
        kernel : Kernel or None
            The kernel used for interpolation.
        window : ndarray or None
            The window applied to the interpolated data.
        padded : ndarray or None
            The padded data.
        tapered : ndarray or None
            The data after applying the window.
        interpolated : NXdata or None
            The interpolated data.
        punched : NXdata or None
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
        data : NXdata
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

    def set_lattice_params(self, lattice_params):
        """
        Sets the lattice parameters and calculates the reciprocal lattice
         parameters.

        Parameters
        ----------
        lattice_params : tuple of float
            The lattice parameters (a, b, c, alpha, beta, gamma) in real space.
        """
        self.lattice_params = lattice_params
        self.puncher.set_lattice_params(lattice_params)
        self.reciprocal_lattice_params = self.puncher.reciprocal_lattice_params

    def add_mask(self, maskaddition):
        """
         Add regions to the current mask using a logical OR operation.

         Parameters
         ----------
         maskaddition : ndarray
             The mask to be added.
         """
        self.puncher.add_mask(maskaddition)
        self.mask = self.puncher.mask

    def subtract_mask(self, masksubtraction):
        """
        Remove regions from the current mask using a logical AND NOT operation.

        Parameters
        ----------
        masksubtraction : ndarray
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
        mask : ndarray
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
        mask : ndarray
            Boolean mask highlighting regions with high intensity.
        """
        return self.puncher.generate_intensity_mask(thresh, radius, verbose)

    def generate_mask_at_coord(self, coordinate, punch_radius, coeffs=None, thresh=None):
        """
        Generate a mask centered at a specific coordinate.

        Parameters
        ----------
        coordinate : tuple of float
            Center coordinate (H, K, L) for the mask.
        punch_radius : float
            Radius for the mask.
        coeffs : list, optional
            Coefficients for the expression of the sphere to be removed around
             each Bragg position, corresponding to coefficients for
             H, HK, K, KL, L, and LH terms. Default is [1, 0, 1, 0, 1, 0].
        thresh : float, optional
            Intensity threshold for applying the mask.

        Returns
        -------
        mask : ndarray
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
        kernel : ndarray
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

    def set_tukey_window(self, tukey_alphas=(1.0, 1.0, 1.0)):
        """
        Set a Tukey window function for data tapering.

        Parameters
        ----------
        tukey_alphas : tuple of floats, optional
            The alpha parameters for the Tukey window in each dimension
             (H, K, L). Default is (1.0, 1.0, 1.0).

        Notes
        -----
        The window function is generated based on the size of the dataset
        in each dimension.
        """
        self.interpolator.set_tukey_window(tukey_alphas)
        self.window = self.interpolator.window

    def set_hexagonal_tukey_window(self, tukey_alphas=(1.0, 1.0, 1.0, 1.0)):
        """
        Set a hexagonal Tukey window function for data tapering.

        Parameters
        ----------
        tukey_alphas : tuple of floats, optional
            The alpha parameters for the Tukey window in each dimension and
             for the hexagonal truncation (H, HK, K, L). Default is (1.0, 1.0, 1.0, 1.0).

        Notes
        -----
        The hexagonal Tukey window is applied to the dataset in a manner that
         preserves hexagonal symmetry.
        """
        self.interpolator.set_hexagonal_tukey_window(tukey_alphas)
        self.window = self.interpolator.window

    def set_window(self, window):
        """
        Set a custom window function for data tapering.

        Parameters
        ----------
        window : ndarray
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
        NXdata
            The padded data with symmetric zero padding.
        """
        self.padded = self.padder.pad(padding)

    def perform_fft(self, is_2d=False):
        """
        Perform a 3D Fourier Transform on the padded data.

        This method applies an inverse Fourier Transform to the padded data
        using `pyfftw` for optimized performance. The result is stored in
        the `fft` attribute as an NXdata object containing the transformed
        spatial frequency components.

        Parameters
        ----------
        is_2d : bool, optional
           If True, performs the FFT only along the first two axes,
           skipping the out-of-plane direction (default is False).

        Returns
        -------
        None

        Notes
        -----
        - Calls `fourier_transform_nxdata` to perform the transformation.
        - The FFT is computed in two stages: first along the last dimension,
         then along the first two dimensions.
        - The output includes frequency components computed from the step
         sizes of the original data axes.

        """

        self.fft = fourier_transform_nxdata(self.padded, is_2d=is_2d)
