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
from nxs_analysis_tools import plot_slice, reciprocal_lattice_params



class Padder():
    """
    A class to pad and unpad datasets with a symmetric region of zeros.
    """

    def __init__(self, data=None):
        """
        Initialize the Symmetrizer3D object.

        Parameters
        ----------
        data : NXdata, optional
            The input data to be symmetrized. If provided, the `set_data` method is called to set the data.

        """
        self.padded = None
        self.padding = None
        if data is not None:
            self.set_data(data)

    def set_data(self, data):
        """
        Set the input data for symmetrization.

        Parameters
        ----------
        data : NXdata
            The input data to be symmetrized.

        """
        self.data = data

        self.steps = tuple([(data[axis].nxdata[1] - data[axis].nxdata[0]) for axis in data.axes])

        # Absolute value of the maximum value; assumes the domain of the input is symmetric (eg, -H_min = H_max)
        self.maxes = tuple([data[axis].nxdata.max() for axis in data.axes])

    def pad(self, padding):
        """
        Symmetrically pads the data with zero values.

        Parameters
        ----------
        padding : tuple
            The number of zero-value pixels to add along each edge of the array.
        """
        data = self.data
        self.padding = padding

        padded_shape = tuple([data[data.signal].nxdata.shape[i] + self.padding[i] * 2 for i in range(data.ndim)])

        # Create padded dataset
        padded = np.zeros(padded_shape)

        slice_obj = [slice(None)] * data.ndim
        for i, _ in enumerate(slice_obj):
            slice_obj[i] = slice(self.padding[i], -self.padding[i], None)
        slice_obj = tuple(slice_obj)
        padded[slice_obj] = data[data.signal].nxdata

        padmaxes = tuple([self.maxes[i] + self.padding[i] * self.steps[i] for i in range(data.ndim)])

        padded = NXdata(NXfield(padded, name=data.signal),
                        tuple([NXfield(np.linspace(-padmaxes[i], padmaxes[i], padded_shape[i]),
                                       name=data.axes[i])
                               for i in range(data.ndim)]))

        self.padded = padded
        return padded

    def save(self, fout_name=None):
        """
        Saves the padded dataset to a .nxs file.

        Parameters
        ----------
        fout_name : str, optional
            The output file name. Default is padded_(Hpadding)_(Kpadding)_(Lpadding).nxs
        """
        padH, padK, padL = self.padding

        # Save padded dataset
        print("Saving padded dataset...")
        f = NXroot()
        f['entry'] = NXentry()
        f['entry']['data'] = self.padded
        if fout_name is None:
            fout_name = 'padded_' + str(padH) + '_' + str(padK) + '_' + str(padL) + '.nxs'
        nxsave(fout_name, f)
        print("Output file saved to: " + os.path.join(os.getcwd(), fout_name))

    def unpad(self, data):
        """
        Removes the padded region from the data.

        Parameters
        ----------
        data : ndarray or NXdata
            The padded data from which to remove the padding.

        Returns
        -------
        ndarray or NXdata
            The unpadded data, with the symmetric padding region removed.

        Notes
        -----
        This method removes the symmetric padding region that was added using the `pad` method. It returns the data
        without the padded region.


        """
        slice_obj = [slice(None)] * data.ndim
        for i in range(data.ndim):
            slice_obj[i] = slice(self.padding[i], -self.padding[i], None)
        slice_obj = tuple(slice_obj)
        return data[slice_obj]


class Symmetrizer2D:
    """
    A class for symmetrizing 2D datasets.
    """
    symmetrization_mask: NXdata

    def __init__(self, **kwargs):
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

    def set_parameters(self, theta_min, theta_max, skew_angle=90, mirror=True):
        """
        Sets the parameters for the symmetrization operation.

        Parameters
        ----------
        theta_min : float
            The minimum angle in degrees for symmetrization.
        theta_max : float
            The maximum angle in degrees for symmetrization.
        skew_angle : float, optional
            The angle in degrees to skew the data during symmetrization (default: 90).
        mirror : bool, optional
            If True, perform mirroring during symmetrization (default: True).
        """
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.skew_angle = skew_angle
        self.mirror = mirror

        # Define Transformation
        skew_angle_adj = 90 - skew_angle
        t = Affine2D()
        # Scale y-axis to preserve norm while shearing
        t += Affine2D().scale(1, np.cos(skew_angle_adj * np.pi / 180))
        # Shear along x-axis
        t += Affine2D().skew_deg(skew_angle_adj, 0)
        # Return to original y-axis scaling
        t += Affine2D().scale(1, np.cos(skew_angle_adj * np.pi / 180)).inverted()
        self.transform = t

        # Calculate number of rotations needed to reconstructed the dataset
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
            reconstructed = np.where(reconstructed == 0, reconstructed + np.flip(reconstructed, axis=0), reconstructed)
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

    def __init__(self, data):
        """
        Initialize the Symmetrizer3D object.

        Parameters
        ----------
        data : NXdata
            The input 3D dataset to be symmetrized.

        """
        self.symmetrized = None
        self.data = data
        self.q1 = data[data.axes[0]]
        self.q2 = data[data.axes[1]]
        self.q3 = data[data.axes[2]]
        self.plane1symmetrizer = Symmetrizer2D()
        self.plane2symmetrizer = Symmetrizer2D()
        self.plane3symmetrizer = Symmetrizer2D()
        self.plane1 = self.q1.nxname + self.q2.nxname
        self.plane2 = self.q1.nxname + self.q3.nxname
        self.plane3 = self.q2.nxname + self.q3.nxname

        print("Plane 1: " + self.plane1)
        print("Plane 2: " + self.plane2)
        print("Plane 3: " + self.plane3)

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
    H, K, L = np.meshgrid(H, K, L)
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
    return gaussian.transpose(1,0,2)


class Puncher:
    def __init__(self):
        self.mask = None
        self.a, self.b, self.c, self.al, self.be, self.ga = [None] * 6

    def set_data(self, data):
        self.data = data

    def set_lattice_params(self, lattice_params):
        self.a, self.b, self.c, self.al, self.be, self.ga = lattice_params
        self.lattice_params = lattice_params
        self.reciprocal_lattice_params = reciprocal_lattice_params(lattice_params)
        self.a_star, self.b_star, self.c_star, self.al_star, self.be_star, self.ga_star = self.reciprocal_lattice_params


    def set_gaussian_background(self, amp, stddev, coeffs=None):
        if coeffs is None:
            coeffs = [1, 0, 1, 0, 1, 0]
        data = self.data
        self.background = generate_gaussian(data[data.axes[0]], data[data.axes[1]], data[data.axes[2]],
                                            amp=amp, stddev=stddev, lattice_params=self.lattice_params,
                                            coeffs=coeffs)

    def plot_gaussian_background(self):
        data = self.data
        fig, axes = plt.subplots(1, 3)
        # Plot the background and subtracted
        plot_slice(self.background[:, :, len(data[data.axes[2]]) // 2], data[data.axes[0]], data[data.axes[1]],
                   ax=axes[0], skew_angle=self.ga_star)
        plot_slice(self.background[:, len(data[data.axes[1]]) // 2, :],data[data.axes[0]], data[data.axes[2]],
                   ax=axes[1], skew_angle=self.be_star)
        plot_slice(self.background[len(data[data.axes[0]]) // 2, :, :],data[data.axes[1]], data[data.axes[2]],
                   ax=axes[2], skew_angle=self.al_star)
        plt.show()
        return fig, axes

    def plot_background_subtraction(self):
        data = self.data
        fig, axes = plt.subplots(1, 3)
        # Plot the background and subtracted
        plot_slice(data[data.axes[0]], data[data.axes[1]],
                   data[data.signal][:, :, 0.0] - self.background[:, :, len(data[data.axes[2]]) // 2],
                   ax=axes[0], skew_angle=self.ga_star)
        plot_slice(data[data.axes[0]], data[data.axes[2]],
                   data[data.signal][:, 0.0, :] - self.background[:, len(data[data.axes[1]]) // 2, :],
                   ax=axes[1], skew_angle=self.be_star)
        plot_slice(data[data.axes[1]], data[data.axes[2]],
                   data[data.signal][0.0, :, :] - self.background[len(data[data.axes[0]]) // 2, :, :],
                   ax=axes[2], skew_angle=self.al_star)
        plt.show()
        return fig, axes

    def generate_mask(self, punch_radius, coeffs=None, thresh=None):
        if coeffs is None:
            coeffs = [1, 0, 1, 0, 1, 0]
        data = self.data
        lattice_params = self.lattice_params
        a, b, c, al, be, ga = lattice_params
        a_, b_, c_, al_, be_, ga_ = reciprocal_lattice_params((a, b, c, al, be, ga))

        H, K, L = np.meshgrid(data[data.axes[0]], data[data.axes[1]], data[data.axes[2]])

        mask = (coeffs[0] * (H - np.rint(H)) ** 2 +
                coeffs[1] * (b_ * a_ / (a_ ** 2)) * (H - np.rint(H)) * (K - np.rint(K)) +
                coeffs[2] * (b_ / a_) ** 2 * (K - np.rint(K)) ** 2 +
                coeffs[3] * (b_ * c_ / (a_ ** 2)) * (K - np.rint(K)) * (L - np.rint(L)) +
                coeffs[4] * (c_ / a_) ** 2 * (L - np.rint(L)) ** 2 +
                coeffs[5] * (c_ * a_ / (a_ ** 2)) * (L - np.rint(L)) * (H - np.rint(H))) < punch_radius ** 2

        if thresh:
            mask = np.logical_and(mask, data[data.signal] > thresh)

        self.mask = mask

        return mask



# class Reducer():
#     pass
#
#
# class Interpolator():
#     pass
#
#
# class FourierTransformer():
#     pass
