import time
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from nexusformat.nexus import nxsave, NXroot, NXentry, NXdata, NXfield
import numpy as np
from nxs_analysis_tools import plot_slice

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

        self.wedge = None

        self.symmetrized = None

    def symmetrize_2d(self, data):
        theta_min = self.theta_min
        theta_max = self.theta_max
        mirror = self.mirror
        t = self.transform
        rotations = self.rotations

        # Define axes that span the plane to be transformed
        q1 = data[data.axes[0]]
        q2 = data[data.axes[1]]

        # Define counts
        counts = data[data.signal].nxdata

        # Calculate the angle for each data point
        theta = np.arctan2(q1.reshape((-1, 1)), q2.reshape((1, -1)))
        # Create a boolean array for the range of angles
        symmetrization_mask = np.logical_and(theta >= theta_min * np.pi / 180, theta <= theta_max * np.pi / 180)
        self.symmetrization_mask = NXdata(NXfield(symmetrization_mask, name='mask'),
                                          (q1,q2))

        # Scale and skew counts
        skew_angle_adj = 90 - self.skew_angle
        counts_skew = ndimage.affine_transform(counts,
                                               t.inverted().get_matrix()[:2, :2],
                                               offset=[counts.shape[0] / 2 * np.sin(skew_angle_adj * np.pi / 180), 0],
                                               order=0,
                                               )
        wedge = ndimage.affine_transform(counts_skew,
                                         Affine2D().scale(np.cos(skew_angle_adj * np.pi / 180), 1).get_matrix()[:2, :2],
                                         offset=[(1 - np.cos(skew_angle_adj * np.pi / 180)) * counts.shape[0] / 2, 0],
                                         order=0,
                                         ) * symmetrization_mask
        self.wedge = NXdata(NXfield(wedge, name=data.signal),
                            (q1,q2))

        # Reconstruct full dataset from wedge
        reconstructed = np.zeros(counts.shape)
        for n in range(0, rotations):
            reconstructed += wedge
            wedge = ndimage.rotate(wedge, 360 / rotations, reshape=False, order=0)

        if mirror:
            reconstructed += np.flip(reconstructed, axis=0)

        reconstructed = ndimage.affine_transform(reconstructed,
                                               Affine2D().scale(np.cos(skew_angle_adj * np.pi / 180),
                                                                1).inverted().get_matrix()[:2, :2],
                                               offset=[-(1 - np.cos(skew_angle_adj * np.pi / 180)) * counts.shape[
                                                   0] / 2 / np.cos(skew_angle_adj * np.pi / 180), 0],
                                               order=0,
                                               )
        reconstructed = ndimage.affine_transform(reconstructed,
                                               t.get_matrix()[:2, :2],
                                               offset=[(-counts.shape[0] / 2 * np.sin(skew_angle_adj * np.pi / 180)),
                                                       0],
                                               order=0,
                                               )
        symmetrized =  NXdata(NXfield(reconstructed, name=data.signal),
                              (data[data.axes[0]],
                               data[data.axes[1]]))

        return symmetrized

class Symmetrizer3D():
    def __init__(self, data):
        self.data = data
        self.q1 = data[data.axes[0]]
        self.q2 = data[data.axes[1]]
        self.q3 = data[data.axes[2]]
        self.s1 = None
        self.s2 = None
        self.s3 = None

        print("(Plane 1, Plane 2, Plane 3) = " + str((self.q1.nxname+self.q2.nxname,
                                     self.q1.nxname+self.q3.nxname,
                                     self.q2.nxname+self.q3.nxname)))

    def set_symmetry_operation_plane1(self, *args, **kwargs):
        s = Symmetrizer2D(*args, **kwargs)
        self.s1=s
    def test_symmetry_operation_plane1(self, data):
        s = self.s1
        symm_test = s.symmetrize_2d(data)
        fig, axes = plt.subplots(1,4,figsize=(15,3))
        plot_slice(data, skew_angle=s.skew_angle, ax=axes[0])
        plot_slice(s.symmetrization_mask, skew_angle=s.skew_angle, ax=axes[1])
        plot_slice(s.wedge, skew_angle=s.skew_angle, ax=axes[2])
        plot_slice(symm_test, skew_angle=s.skew_angle, ax=axes[3])
        plt.subplots_adjust(wspace=0.2)
        plt.show()

    def set_symmetry_operation_plane2(self, *args, **kwargs):
        s = Symmetrizer2D(*args, **kwargs)
        test = s.symmetrize_2d(self.data[:,len(self.q2) // 2,:])
        plot_slice(test)
        self.s2 = s

    def set_symmetry_operation_plane3(self, *args, **kwargs):
        s = Symmetrizer2D(*args, **kwargs)
        test = s.symmetrize_2d(self.data[len(self.q1) // 2,:,:])
        plot_slice(test)
        self.s3 = s

    # def symmetrize(self):
    #     data = self.data
    #     q1,q2,q3 = self.q1,self.q2,self.q3
    #     out_array = np.zeros(data[data.signal].shape)
    #
    #     for k in range(0, len(q3)):
    #         print('Symmetrizing L=' + str(q3[k]) + "...", end='\r')
    #         out_array[:, :, k] = s3(q1, q2, g[:, :, k].counts, theta_min=60, theta_max=90, mirror=True,
    #                                          mirror_axis=0, skew_angle=60)
    #
    #         if k % 2:
    #             plot_slice(g.H, g.K, out_arr[:, :, k],
    #                        vmin=0, vmax=50,
    #                        skew_angle=60,
    #                        title='None',
    #                        )
    #             plt.show()
    #             # time.sleep(1)
    #         else:
    #             clear_output()
    #     print('Symmetrized HK planes.')
    #
    #     for j in range(0, len(g.K)):
    #         print('Symmetrizing K=' + str(g.K[j]) + "...", end='\r')
    #         out_arr[:, j, :] = symmetrize_2d(g.K, g.L, out_arr[:, j, :], theta_min=-90, theta_max=90, mirror=True,
    #                                          mirror_axis=1, skew_angle=90)
    #
    #         if j % 2:
    #             clear_output()
    #             plot_slice(g.K, g.L, out_arr[:, j, :],
    #                        vmin=0, vmax=50,
    #                        skew_angle=90,
    #                        title='None',
    #                        )
    #             plt.gca().set(aspect=c_ / a_)
    #             plt.show()
    #             # time.sleep(1)
    #         else:
    #             clear_output()
    #     print('Symmetrized KL planes.')
    #
    #     end = time.time()
    #
    #     out_arr[out_arr < 0] = 0
    #
    #     print("Symmetriztaion finished. Time = " + str(end - start) + " seconds.")
    #
    #     print("Saving file...")
    #
    #     f = NXroot()
    #     f['entry'] = NXentry()
    #     f['entry']['data'] = NXdata(NXfield(out_arr, name='counts'),
    #                                 [g.H,
    #                                  g.K,
    #                                  g.L])
    #     fout_name = 'symmetrized.nxs'
    #     nxsave(outdir + fout_name, f)
    #     print("Output file saved to: " + fout_name)

class Puncher():
    pass

class Reducer():
    pass

class Interpolator():
    pass

class Padder():
    pass

class FourierTransformer():
    pass