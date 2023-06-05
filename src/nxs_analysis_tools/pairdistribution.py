import time
import os
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from nexusformat.nexus import nxsave, NXroot, NXentry, NXdata, NXfield
import numpy as np
from nxs_analysis_tools import plot_slice



class Padder():
    def __init__(self, data):
        self.data = data

        self.steps = tuple([(data[axis].nxdata[1] - data[axis].nxdata[0]) for axis in data.axes])

        # Absolute value of the maximum value; assumes the domain of the input is symmetric (eg, -H_min = H_max)
        self.maxes = tuple([data[axis].nxdata.max() for axis in data.axes])

    def pad(self, padding):
        data = self.data
        self.padding = padding

        padded_shape = tuple([data[data.signal].nxdata.shape[i] + self.padding[i]*2 for i in range(data.ndim)])

        # Create padded dataset
        padded = np.zeros(padded_shape)

        slice_obj = [slice(None)]*data.ndim
        for i in range(len(slice_obj)):
            slice_obj[i] = slice(self.padding[i],-self.padding[i],None)
        slice_obj = tuple(slice_obj)
        padded[slice_obj] = data[data.signal].nxdata

        padmaxes = tuple([self.maxes[i] + self.padding[i] * self.steps[i] for i in range(data.ndim)])

        padded = NXdata(NXfield(padded, name=data.signal),
                        tuple([NXfield(np.linspace(-padmaxes[i],padmaxes[i],padded_shape[i]),
                                       name=data.axes[i])
                               for i in range(data.ndim)]))

        self.padded = padded
        return padded

    def save(self):
        padH, padK, padL = self.padding

        # Save padded dataset
        print("Saving padded dataset...")
        f = NXroot()
        f['entry'] = NXentry()
        f['entry']['data'] = self.padded
        fout_name = 'padded_' + str(padH) + '_' + str(padK) + '_' + str(padL) + '.nxs'
        nxsave(fout_name, f)
        print("Output file saved to: " + os.path.join(os.getcwd(),fout_name))

    def unpad(self,data):
        slice_obj = [slice(None)]*data.ndim
        for i in range(data.ndim):
            slice_obj[i] = slice(self.padding[i],-self.padding[i],None)
        slice_obj = tuple(slice_obj)
        return data[slice_obj]


class Symmetrizer2D():
    def __init__(self, theta_min, theta_max, skew_angle=90, mirror=True):
        """

        Args:
            data:
            theta_min:
            theta_max:
            skew_angle:
            mirror:
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

        scale2 = counts.shape[0]/counts.shape[1]
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
            reconstructed = np.where(reconstructed==0,reconstructed+np.flip(reconstructed, axis=0),reconstructed)
            # reconstructed += np.flip(reconstructed, axis=0)

        # self.rotated_and_mirrored = NXdata(NXfield(reconstructed, name=data.signal),
        #                                    (q1, q2))

        reconstructed = ndimage.affine_transform(reconstructed,
                                                 Affine2D().scale(scale2,1).inverted().get_matrix()[:2, :2],
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

        symmetrized = NXdata(NXfield(reconstructed_unpadded, name=data.signal),
                             (data[data.axes[0]],
                              data[data.axes[1]]))

        return symmetrized


class Symmetrizer3D():
    def __init__(self, data):
        self.data = data
        self.q1 = data[data.axes[0]]
        self.q2 = data[data.axes[1]]
        self.q3 = data[data.axes[2]]
        self.plane1symmetrizer = None
        self.plane2symmetrizer = None
        self.plane3symmetrizer = None
        self.plane1 = self.q1.nxname + self.q2.nxname
        self.plane2 = self.q1.nxname + self.q3.nxname
        self.plane3 = self.q2.nxname + self.q3.nxname

        print("\nPlane 1: " + self.plane1)
        print("Plane 2: " + self.plane2)
        print("Plane 3: " + self.plane3)

    def set_plane1symmetrizer(self, *args, **kwargs):
        s = Symmetrizer2D(*args, **kwargs)
        self.plane1symmetrizer = s

    def test_plane1(self, data):
        s = self.plane1symmetrizer
        symm_test = s.symmetrize_2d(data)
        fig, axes = plt.subplots(1, 4, figsize=(15, 3))
        plot_slice(data, skew_angle=s.skew_angle, ax=axes[0], title='data')
        plot_slice(s.symmetrization_mask, skew_angle=s.skew_angle, ax=axes[1], title='mask')
        plot_slice(s.wedge, ax=axes[2], title='wedge')
        plot_slice(symm_test, skew_angle=s.skew_angle, ax=axes[3], title='symmetrized')
        plt.subplots_adjust(wspace=0.2)
        plt.show()

    def set_plane2symmetrizer(self, *args, **kwargs):
        s = Symmetrizer2D(*args, **kwargs)
        self.plane2symmetrizer = s

    def test_plane2(self, data):
        s = self.plane2symmetrizer
        symm_test = s.symmetrize_2d(data)
        fig, axes = plt.subplots(1, 4, figsize=(15, 3))
        plot_slice(data, skew_angle=s.skew_angle, ax=axes[0], title='data')
        plot_slice(s.symmetrization_mask, skew_angle=s.skew_angle, ax=axes[1], title='mask')
        plot_slice(s.wedge, ax=axes[2], title='wedge')
        plot_slice(symm_test, skew_angle=s.skew_angle, ax=axes[3], title='symmetrized')
        plt.subplots_adjust(wspace=0.2)
        plt.show()

    def set_plane3symmetrizer(self, *args, **kwargs):
        s = Symmetrizer2D(*args, **kwargs)
        self.plane3symmetrizer = s

    def test_plane3(self, data):
        s = self.plane3symmetrizer
        symm_test = s.symmetrize_2d(data)
        fig, axes = plt.subplots(1, 4, figsize=(15, 3))
        plot_slice(data, skew_angle=s.skew_angle, ax=axes[0], title='data')
        plot_slice(s.symmetrization_mask, skew_angle=s.skew_angle, ax=axes[1], title='mask')
        plot_slice(s.wedge, ax=axes[2], title='wedge')
        plot_slice(symm_test, skew_angle=s.skew_angle, ax=axes[3], title='symmetrized')
        plt.subplots_adjust(wspace=0.2)
        plt.show()

    def symmetrize(self):
        starttime = time.time()
        data = self.data
        q1,q2,q3 = self.q1,self.q2,self.q3
        out_array = np.zeros(data[data.signal].shape)

        print('Symmetrizing ' + self.plane1 +' planes...')
        for k in range(0, len(q3)):
            print('Symmetrizing ' + q3.nxname + '=' + str(q3[k]) + "...", end='\r')
            data_symmetrized = self.plane1symmetrizer.symmetrize_2d(data[:,:,k])
            out_array[:, :, k] = data_symmetrized[data.signal].nxdata
        print('Symmetrized ' + self.plane1 +' planes.')

        print('Symmetrizing ' + self.plane2 + ' planes...')
        for j in range(0, len(q2)):
            print('Symmetrizing ' + q2.nxname + '=' + str(q2[j]) + "...", end='\r')
            data_symmetrized = self.plane2symmetrizer.symmetrize_2d(data[:, j, :])
            out_array[:, j, :] = data_symmetrized[data.signal].nxdata
        print('Symmetrized ' + self.plane2 + ' planes.')

        print('Symmetrizing ' + self.plane3 + ' planes...')
        for i in range(0, len(q1)):
            print('Symmetrizing ' + q1.nxname + '=' + str(q1[i]) + "...", end='\r')
            data_symmetrized = self.plane3symmetrizer.symmetrize_2d(data[i, :, :])
            out_array[i, :, :] = data_symmetrized[data.signal].nxdata
        print('Symmetrized ' + self.plane3 + ' planes.')


        out_array[out_array < 0] = 0

        stoptime = time.time()
        print("Symmetriztaion finished in " + str((stoptime - starttime)/60) + " minutes.")

        self.symmetrized = NXdata(NXfield(out_array, name=data.signal), tuple([data[axis] for axis in data.axes]))

    def save(self):
        print("Saving file...")

        f = NXroot()
        f['entry'] = NXentry()
        f['entry']['data'] = self.symmetrized
        nxsave('symmetrized.nxs', f)
        print("Output file saved to: " + os.path.join(os.getcwd(),'symmetrized.nxs'))


class Puncher():
    pass


class Reducer():
    pass


class Interpolator():
    pass


class FourierTransformer():
    pass
