import numpy as np
from scipy.ndimage import affine_transform
from matplotlib.transforms import Affine2D

def shear_transformation(angle):
    # Define shear transformation
    t = Affine2D()

    # Scale y-axis to preserve norm while shearing
    t += Affine2D().scale(1, np.cos(angle * np.pi / 180))

    # Shear along x-axis
    t += Affine2D().skew_deg(angle, 0)

    # Return to original y-axis scaling
    t += Affine2D().scale(1, np.cos(angle * np.pi / 180)).inverted()

    return t

class ShearTransformer():
    def __init__(self, angle):
        self.shear_angle = 90 - angle
        self.t = shear_transformation(self.shear_angle)
        self.scale = np.cos(self.shear_angle * np.pi / 180)

    def apply(self, image):
        # Perform shear operation
        image_skewed = affine_transform(image, self.t.inverted().get_matrix()[:2, :2], 
                                        offset=[image.shape[0] / 2 * np.sin(self.shear_angle * np.pi / 180), 0], 
                                        order=0
                                        )
        # Scale data based on skew angle
        image_scaled = affine_transform(image_skewed, Affine2D().scale(self.scale, 1).get_matrix()[:2, :2],
                                        offset=[(1 - self.scale) * image.shape[0] / 2, 0],
                                        order=0
                                        )
        return image_scaled

    def invert(self, image):

        # Undo scaling
        image_unscaled = affine_transform(image, Affine2D().scale(self.scale, 1).inverted().get_matrix()[:2, :2],
                                          offset=[-(1 - self.scale) * image.shape[0] / 2 / self.scale, 0],
                                          order=0
                                          )
        # Undo shear operation
        image_unskewed = affine_transform(image_unscaled, self.t.get_matrix()[:2, :2],
                                          offset=[(-image.shape[0] / 2 * np.sin(self.shear_angle * np.pi / 180)), 0],
                                          order=0
                                          )
        return image_unskewed


def rotate_plane_affine(
    image,
    lattice_angle,
    rotation_angle,
    dq1=1.0,
    dq2=1.0,
    origin=None,
    aspect=1.0,
    order=1,
    cval=0.0
):
    """
    Apply a single-pass affine rotation to a 2D image in reciprocal space.

    Parameters
    ----------
    image : numpy.ndarray
        2D array to rotate.
    lattice_angle : float
        Angle between the two in-plane lattice axes in degrees (gamma).
    rotation_angle : float
        Angle of rotation in degrees (counter-clockwise).
    dq1 : float, optional
        Step size along the first axis. Defaults to 1.0.
    dq2 : float, optional
        Step size along the second axis. Defaults to 1.0.
    origin : tuple of float or numpy.ndarray, optional
        Continuous pixel coordinates (c1, c2) of the rotation center (physical (0, 0)).
        If None, defaults to the array center ((shape[0]-1)/2, (shape[1]-1)/2).
    aspect : float, optional
        Aspect ratio |b*| / |a*| between lengths of basis vectors. Defaults to 1.0.
    order : int, optional
        Interpolation order for `scipy.ndimage.affine_transform`. Defaults to 1.
    cval : float, optional
        Value to fill points outside input boundaries. Defaults to 0.0.

    Returns
    -------
    numpy.ndarray
        Rotated 2D array of the same shape as `image`.
    """
    gamma = np.radians(lattice_angle)
    B = np.array([
        [dq1, aspect * dq2 * np.cos(gamma)],
        [0.0, aspect * dq2 * np.sin(gamma)]
    ], dtype=float)
    
    B_inv = np.linalg.inv(B)
    
    theta = np.radians(rotation_angle)
    R_neg = np.array([
        [np.cos(-theta), -np.sin(-theta)],
        [np.sin(-theta),  np.cos(-theta)]
    ], dtype=float)
    
    M = B_inv @ R_neg @ B
    
    if origin is None:
        c = np.array([(image.shape[0] - 1) / 2.0, (image.shape[1] - 1) / 2.0], dtype=float)
    else:
        c = np.asarray(origin, dtype=float)
        
    offset = c - M @ c
    return affine_transform(image, matrix=M, offset=offset, order=order, cval=cval)


def mirror_plane_affine(
    image,
    lattice_angle=90.0,
    mirror_angle=None,
    mirror_axis=None,
    dq1=1.0,
    dq2=1.0,
    origin=None,
    aspect=1.0,
    order=1,
    cval=0.0
):
    """
    Apply a single-pass affine reflection (mirror) to a 2D image in reciprocal space.

    Parameters
    ----------
    image : numpy.ndarray
        2D array to mirror.
    lattice_angle : float, optional
        Angle between the two in-plane lattice axes in degrees (gamma). Defaults to 90.0.
    mirror_angle : float, optional
        Cartesian angle in degrees of the reflection axis. For example, 30.0 for hexagonal
        diagonal reflection, 45.0 for square diagonal reflection.
    mirror_axis : int or str, optional
        Discrete mirror axis: 0 for q1 -> -q1, 1 for q2 -> -q2, or 'diagonal'/'transpose'
        for (q1, q2) -> (q2, q1).
    dq1 : float, optional
        Step size along the first axis. Defaults to 1.0.
    dq2 : float, optional
        Step size along the second axis. Defaults to 1.0.
    origin : tuple of float or numpy.ndarray, optional
        Continuous pixel coordinates (c1, c2) of the reflection center (physical (0, 0)).
        If None, defaults to the array center.
    aspect : float, optional
        Aspect ratio |b*| / |a*|. Defaults to 1.0.
    order : int, optional
        Interpolation order for `scipy.ndimage.affine_transform`. Defaults to 1.
    cval : float, optional
        Value to fill points outside input boundaries. Defaults to 0.0.

    Returns
    -------
    numpy.ndarray
        Mirrored 2D array of the same shape as `image`.
    """
    gamma = np.radians(lattice_angle)
    B = np.array([
        [dq1, aspect * dq2 * np.cos(gamma)],
        [0.0, aspect * dq2 * np.sin(gamma)]
    ], dtype=float)
    B_inv = np.linalg.inv(B)
    
    if mirror_angle is not None:
        psi = np.radians(mirror_angle)
        Ref = np.array([
            [np.cos(2 * psi), np.sin(2 * psi)],
            [np.sin(2 * psi), -np.cos(2 * psi)]
        ], dtype=float)
        M = B_inv @ Ref @ B
    elif mirror_axis == 0:
        M = np.array([[-1.0, 0.0], [0.0, 1.0]], dtype=float)
    elif mirror_axis == 1:
        M = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=float)
    elif mirror_axis in ('diagonal', 'transpose'):
        M = np.array([[0.0, dq2 / dq1], [dq1 / dq2, 0.0]], dtype=float)
    else:
        Ref = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=float)
        M = B_inv @ Ref @ B

    if origin is None:
        c = np.array([(image.shape[0] - 1) / 2.0, (image.shape[1] - 1) / 2.0], dtype=float)
    else:
        c = np.asarray(origin, dtype=float)

    offset = c - M @ c
    return affine_transform(image, matrix=M, offset=offset, order=order, cval=cval)