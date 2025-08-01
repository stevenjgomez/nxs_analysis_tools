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