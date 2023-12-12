"""
Tools for generating single crystal pair distribution functions.
"""
import time
import os
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from nexusformat.nexus import nxsave, NXroot, NXentry, NXdata, NXfield
import numpy as np
from .datareduction import plot_slice, reciprocal_lattice_params, Padder

class Symmetrizer2D:
    """
    A class for symmetrizing 2D datasets.
    """
    symmetrization_mask: NXdata

    def __init__(self, **kwargs):
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
        Sets the parameters for the symmetrization operation.

        Parameters
        ----------
        theta_min : float
            The minimum angle in degrees for symmetrization.
        theta_max : float
            The maximum angle in degrees for symmetrization.
        lattice_angle : float, optional
            The angle in degrees between the two principal axes of the plane to be symmetrized (default: 90).
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
        Symmetrizes a 2D dataset based on the set parameters.

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

        # Pad the dataset so that rotations don't get cutoff if they extend past the extent of the dataset
        p = Padder(data)
        padding = tuple([len(data[axis]) for axis in data.axes])
        data_padded = p.pad(padding)

        # Define axes that span the plane to be transformed
        q1 = data_padded[data.axes[0]]
        q2 = data_padded[data.axes[1]]

        # Define signal to be symmetrized
        counts = data_padded[data.signal].nxdata

        # Calculate the angle for each data point
        theta = np.arctan2(q1.reshape((-1, 1)), q2.reshape((1, -1)))
        # Create a boolean array for the range of angles
        symmetrization_mask = np.logical_and(theta >= theta_min * np.pi / 180, theta <= theta_max * np.pi / 180)
        self.symmetrization_mask = NXdata(NXfield(p.unpad(symmetrization_mask), name='mask'),
                                          (data[data.axes[0]], data[data.axes[1]]))

        self.wedge = NXdata(NXfield(p.unpad(counts * symmetrization_mask), name=data.signal),
                            (data[data.axes[0]], data[data.axes[1]]))

        # Scale and skew counts
        skew_angle_adj = 90 - self.skew_angle
        counts_skew = ndimage.affine_transform(counts,
                                               t.inverted().get_matrix()[:2, :2],
                                               offset=[counts.shape[0] / 2 * np.sin(skew_angle_adj * np.pi / 180), 0],
                                               order=0,
                                               )
        scale1 = np.cos(skew_angle_adj * np.pi / 180)
        wedge = ndimage.affine_transform(counts_skew,
                                         Affine2D().scale(scale1, 1).get_matrix()[:2, :2],
                                         offset=[(1 - scale1) * counts.shape[0] / 2, 0],
                                         order=0,
                                         ) * symmetrization_mask

        scale2 = counts.shape[0] / counts.shape[1]
        wedge = ndimage.affine_transform(wedge,
                                         Affine2D().scale(scale2, 1).get_matrix()[:2, :2],
                                         offset=[(1 - scale2) * counts.shape[0] / 2, 0],
                                         order=0,
                                         )

        # Reconstruct full dataset from wedge
        reconstructed = np.zeros(counts.shape)
        for n in range(0, rotations):
            # The following are attempts to combine images with minimal overlapping pixels
            reconstructed += wedge
            # reconstructed = np.where(reconstructed == 0, reconstructed + wedge, reconstructed)

            wedge = ndimage.rotate(wedge, 360 / rotations, reshape=False, order=0)

        # self.rotated_only = NXdata(NXfield(reconstructed, name=data.signal),
        #                            (q1, q2))

        if mirror:
            # The following are attempts to combine images with minimal overlapping pixels
            reconstructed = np.where(reconstructed == 0,
                                     reconstructed + np.flip(reconstructed, axis=mirror_axis), reconstructed)
            # reconstructed += np.flip(reconstructed, axis=0)

        # self.rotated_and_mirrored = NXdata(NXfield(reconstructed, name=data.signal),
        #                                    (q1, q2))

        reconstructed = ndimage.affine_transform(reconstructed,
                                                 Affine2D().scale(scale2, 1).inverted().get_matrix()[:2, :2],
                                                 offset=[-(1 - scale2) * counts.shape[
                                                     0] / 2 / scale2, 0],
                                                 order=0,
                                                 )
        reconstructed = ndimage.affine_transform(reconstructed,
                                                 Affine2D().scale(scale1,
                                                                  1).inverted().get_matrix()[:2, :2],
                                                 offset=[-(1 - scale1) * counts.shape[
                                                     0] / 2 / scale1, 0],
                                                 order=0,
                                                 )
        reconstructed = ndimage.affine_transform(reconstructed,
                                                 t.get_matrix()[:2, :2],
                                                 offset=[(-counts.shape[0] / 2 * np.sin(skew_angle_adj * np.pi / 180)),
                                                         0],
                                                 order=0,
                                                 )

        reconstructed_unpadded = p.unpad(reconstructed)

        # Fix any overlapping pixels by truncating counts to max
        reconstructed_unpadded[reconstructed_unpadded > data[data.signal].nxdata.max()] = data[data.signal].nxdata.max()

        symmetrized = NXdata(NXfield(reconstructed_unpadded, name=data.signal),
                             (data[data.axes[0]],
                              data[data.axes[1]]))

        return symmetrized

    def test(self, data, **kwargs):
        """
        Performs a test visualization of the symmetrization process.

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
            The numpy array of Axes objects representing the subplots in the test visualization.

        Notes
        -----
        This method uses the `symmetrize_2d` method to perform the symmetrization on the input data and visualize
        the process.

        The test visualization plot includes the following subplots:
        - Subplot 1: The original dataset.
        - Subplot 2: The symmetrization mask.
        - Subplot 3: The wedge slice used for reconstruction of the full symmetrized dataset.
        - Subplot 4: The symmetrized dataset.

        Example usage:
        ```
        s = Scissors()
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


class Symmetrizer3D:
    """
    A class to symmetrize 3D datasets.
    """

    def __init__(self, data=None):
        """
        Initialize the Symmetrizer3D object.

        Parameters
        ----------
        data : NXdata, optional
            The input 3D dataset to be symmetrized.

        """

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
        Sets the data to be symmetrized.

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
        self.a, self.b, self.c, self.al, self.be, self.ga = lattice_params
        self.lattice_params = lattice_params
        self.reciprocal_lattice_params = reciprocal_lattice_params(lattice_params)
        self.a_star, self.b_star, self.c_star, self.al_star, self.be_star, self.ga_star = self.reciprocal_lattice_params

    def symmetrize(self):
        """
        Perform the symmetrization of the 3D dataset.

        Returns
        -------
        symmetrized : NXdata
            The symmetrized 3D dataset.

        """
        starttime = time.time()
        data = self.data
        q1, q2, q3 = self.q1, self.q2, self.q3
        out_array = np.zeros(data[data.signal].shape)

        print('Symmetrizing ' + self.plane1 + ' planes...')
        for k in range(0, len(q3)):
            print('Symmetrizing ' + q3.nxname + '=' + "{:.02f}".format(q3[k]) + "...", end='\r')
            data_symmetrized = self.plane1symmetrizer.symmetrize_2d(data[:, :, k])
            out_array[:, :, k] = data_symmetrized[data.signal].nxdata
        print('\nSymmetrized ' + self.plane1 + ' planes.')

        print('Symmetrizing ' + self.plane2 + ' planes...')
        for j in range(0, len(q2)):
            print('Symmetrizing ' + q2.nxname + '=' + "{:.02f}".format(q2[j]) + "...", end='\r')
            data_symmetrized = self.plane2symmetrizer.symmetrize_2d(
                NXdata(NXfield(out_array[:, j, :], name=data.signal),
                       (q1, q3)))
            out_array[:, j, :] = data_symmetrized[data.signal].nxdata
        print('\nSymmetrized ' + self.plane2 + ' planes.')

        print('Symmetrizing ' + self.plane3 + ' planes...')
        for i in range(0, len(q1)):
            print('Symmetrizing ' + q1.nxname + '=' + "{:.02f}".format(q1[i]) + "...", end='\r')
            data_symmetrized = self.plane3symmetrizer.symmetrize_2d(
                NXdata(NXfield(out_array[i, :, :], name=data.signal),
                       (q2, q3)))
            out_array[i, :, :] = data_symmetrized[data.signal].nxdata
        print('\nSymmetrized ' + self.plane3 + ' planes.')

        out_array[out_array < 0] = 0

        stoptime = time.time()
        print("\nSymmetriztaion finished in " + "{:.02f}".format((stoptime - starttime) / 60) + " minutes.")

        self.symmetrized = NXdata(NXfield(out_array, name=data.signal), tuple([data[axis] for axis in data.axes]))

        return self.symmetrized

    def save(self, fout_name=None):
        """
        Save the symmetrized dataset to a file.

        Parameters
        ----------
        fout_name : str, optional
            The name of the output file. If not provided, the default name 'symmetrized.nxs' will be used.

        """
        print("Saving file...")

        f = NXroot()
        f['entry'] = NXentry()
        f['entry']['data'] = self.symmetrized
        if fout_name is None:
            fout_name = 'symmetrized.nxs'
        nxsave(fout_name, f)
        print("Output file saved to: " + os.path.join(os.getcwd(), fout_name))


def generate_gaussian(H, K, L, amp, stddev, lattice_params, coeffs=None):
    """
    Generate a 3D Gaussian distribution.

    Parameters
    ----------
    H, K, L : ndarray
        Arrays specifying the values of H, K, and L coordinates.
    amp : float
        Amplitude of the Gaussian distribution.
    stddev : float
        Standard deviation of the Gaussian distribution.
    lattice_params : tuple
        Tuple of lattice parameters (a, b, c, alpha, beta, gamma).
    coeffs : list, optional
        Coefficients for the Gaussian expression, including cross-terms between axes. Default is [1, 0, 1, 0, 1, 0],
        corresponding to (1*H**2 + 0*H*K + 1*K**2 + 0*K*L + 1*L**2 + 0*L*H)

    Returns
    -------
    gaussian : ndarray
        3D Gaussian distribution.
    """
    if coeffs is None:
        coeffs = [1, 0, 1, 0, 1, 0]
    a, b, c, al, be, ga = lattice_params
    a_, b_, c_, al_, be_, ga_ = reciprocal_lattice_params((a, b, c, al, be, ga))
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
    def __init__(self):
        self.punched = None
        self.data = None
        self.H, self.K, self.L = [None] * 3
        self.mask = None
        self.reciprocal_lattice_params = None
        self.lattice_params = None
        self.a, self.b, self.c, self.al, self.be, self.ga = [None] * 6
        self.a_star, self.b_star, self.c_star, self.al_star, self.be_star, self.ga_star = [None] * 6

    def set_data(self, data):
        self.data = data
        if self.mask is None:
            self.mask = np.zeros(data[data.signal].nxdata.shape)
        self.H, self.K, self.L = np.meshgrid(data[data.axes[0]], data[data.axes[1]], data[data.axes[2]], indexing='ij')

    def set_lattice_params(self, lattice_params):
        self.a, self.b, self.c, self.al, self.be, self.ga = lattice_params
        self.lattice_params = lattice_params
        self.reciprocal_lattice_params = reciprocal_lattice_params(lattice_params)
        self.a_star, self.b_star, self.c_star, self.al_star, self.be_star, self.ga_star = self.reciprocal_lattice_params

    def add_mask(self, maskaddition):
        self.mask = np.logical_or(self.mask, maskaddition)

    def subtract_mask(self, masksubtraction):
        self.mask = np.logical_and(self.mask, np.logical_not(masksubtraction))

    def generate_bragg_mask(self, punch_radius, coeffs=None, thresh=None):
        if coeffs is None:
            coeffs = [1, 0, 1, 0, 1, 0]
        data = self.data
        H, K, L = self.H, self.K, self.L
        a_, b_, c_, al_, be_, ga_ = self.reciprocal_lattice_params

        mask = (coeffs[0] * (H - np.rint(H)) ** 2 +
                coeffs[1] * (b_ * a_ / (a_ ** 2)) * (H - np.rint(H)) * (K - np.rint(K)) +
                coeffs[2] * (b_ / a_) ** 2 * (K - np.rint(K)) ** 2 +
                coeffs[3] * (b_ * c_ / (a_ ** 2)) * (K - np.rint(K)) * (L - np.rint(L)) +
                coeffs[4] * (c_ / a_) ** 2 * (L - np.rint(L)) ** 2 +
                coeffs[5] * (c_ * a_ / (a_ ** 2)) * (L - np.rint(L)) * (H - np.rint(H))) < punch_radius ** 2

        if thresh:
            mask = np.logical_and(mask, data[data.signal] > thresh)

        return mask

    def generate_mask_at_coord(self, coordinate, punch_radius, coeffs=None, thresh=None):
        if coeffs is None:
            coeffs = [1, 0, 1, 0, 1, 0]
        data = self.data
        H, K, L = self.H, self.K, self.L
        a_, b_, c_, al_, be_, ga_ = self.reciprocal_lattice_params
        centerH, centerK, centerL = coordinate
        mask = (coeffs[0] * (H - centerH) ** 2 +
                coeffs[1] * (b_ * a_ / (a_ ** 2)) * (H - centerH) * (K - centerK) +
                coeffs[2] * (b_ / a_) ** 2 * (K - centerK) ** 2 +
                coeffs[3] * (b_ * c_ / (a_ ** 2)) * (K - centerK) * (L - centerL) +
                coeffs[4] * (c_ / a_) ** 2 * (L - centerL) ** 2 +
                coeffs[5] * (c_ * a_ / (a_ ** 2)) * (L - centerL) * (H - centerH)) < punch_radius ** 2

        if thresh:
            mask = np.logical_and(mask, data[data.signal] > thresh)

        return mask

    def punch(self):
        data= self.data
        self.punched = NXdata(NXfield(np.where(self.mask, np.nan, data[data.signal].nxdata), name=data.signal),
                              (data[data.axes[0]],data[data.axes[1]],data[data.axes[2]]))
        return self.punched


# class Reducer():
#     pass
#
#

class Interpolator():
    def __init__(self):
        self.data = None

    def set_data(self, data):
        self.data = data

    def set_kernel(self, kernel):
        self.kernel = kernel

    def generate_gaussian_kernel(self, amp, stddev, coeffs=None):
        pass

    def interpolate(self):
        pass
