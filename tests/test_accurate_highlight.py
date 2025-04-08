import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from matplotlib.markers import MarkerStyle
from matplotlib.ticker import MultipleLocator
from matplotlib import colors
from matplotlib import patches
from IPython.display import display, Markdown
from nexusformat.nexus import NXfield, NXdata, nxload, NeXusError, NXroot, NXentry, nxsave
from scipy import ndimage
from nxs_analysis_tools.datareduction import plot_slice, load_data


class Scissors:
    """
    Scissors class provides functionality for reducing data to a 1D linecut using an integration
    window.

    Attributes
    ----------
    data : :class:`nexusformat.nexus.NXdata` or None
        Input :class:`nexusformat.nexus.NXdata`.
    center : tuple or None
        Central coordinate around which to perform the linecut.
    window : tuple or None
        Extents of the window for integration along each axis.
    axis : int or None
        Axis along which to perform the integration.
    integration_volume : :class:`nexusformat.nexus.NXdata` or None
        Data array after applying the integration window.
    integrated_axes : tuple or None
        Indices of axes that were integrated.
    linecut : :class:`nexusformat.nexus.NXdata` or None
        1D linecut data after integration.
    integration_window : tuple or None
        Slice object representing the integration window in the data array.

    Methods
    -------
    set_data(data)
        Set the input :class:`nexusformat.nexus.NXdata`.
    get_data()
        Get the input :class:`nexusformat.nexus.NXdata`.
    set_center(center)
        Set the central coordinate for the linecut.
    set_window(window, axis=None, verbose=False)
        Set the extents of the integration window.
    get_window()
        Get the extents of the integration window.
    cut_data(center=None, window=None, axis=None, verbose=False)
        Reduce data to a 1D linecut using the integration window.
    highlight_integration_window(data=None, label=None, highlight_color='red', **kwargs)
        Plot the integration window highlighted on a 2D heatmap of the full dataset.
    plot_integration_window(**kwargs)
        Plot a 2D heatmap of the integration window data.
    """

    def __init__(self, data=None, center=None, window=None, axis=None):
        """
        Initializes a Scissors object.

        Parameters
        ----------
        data : :class:`nexusformat.nexus.NXdata` or None, optional
            Input NXdata. Default is None.
        center : tuple or None, optional
            Central coordinate around which to perform the linecut. Default is None.
        window : tuple or None, optional
            Extents of the window for integration along each axis. Default is None.
        axis : int or None, optional
            Axis along which to perform the integration. Default is None.
        """

        self.data = data
        self.center = tuple(float(i) for i in center) if center is not None else None
        self.window = tuple(float(i) for i in window) if window is not None else None
        self.axis = axis

        self.integration_volume = None
        self.integrated_axes = None
        self.linecut = None
        self.integration_window = None

    def set_data(self, data):
        """
        Set the input NXdata.

        Parameters
        ----------
        data : :class:`nexusformat.nexus.NXdata`
            Input data array.
        """
        self.data = data

    def get_data(self):
        """
        Get the input data array.

        Returns
        -------
        ndarray or None
            Input data array.
        """
        return self.data

    def set_center(self, center):
        """
        Set the central coordinate for the linecut.

        Parameters
        ----------
        center : tuple
            Central coordinate around which to perform the linecut.
        """
        self.center = tuple(float(i) for i in center) if center is not None else None

    def set_window(self, window, axis=None, verbose=False):
        """
        Set the extents of the integration window.

        Parameters
        ----------
        window : tuple
            Extents of the window for integration along each axis.
        axis : int or None, optional
            The axis along which to perform the linecut. If not specified, the value from the
            object's attribute will be used.
        verbose : bool, optional
            Enables printout of linecut axis and integrated axes. Default False.

        """
        self.window = tuple(float(i) for i in window) if window is not None else None

        # Determine the axis for integration
        self.axis = window.index(max(window)) if axis is None else axis

        # Determine the integrated axes (axes other than the integration axis)
        self.integrated_axes = tuple(i for i in range(self.data.ndim) if i != self.axis)

        if verbose:
            print("Linecut axis: " + str(self.data.axes[self.axis]))
            print("Integrated axes: " + str([self.data.axes[axis]
                                             for axis in self.integrated_axes]))

    def get_window(self):
        """
        Get the extents of the integration window.

        Returns
        -------
        tuple or None
            Extents of the integration window.
        """
        return self.window

    def cut_data(self, center=None, window=None, axis=None, verbose=False):
        """
        Reduces data to a 1D linecut with integration extents specified by the
        window about a central coordinate.

        Parameters
        ----------
        center : float or None, optional
            Central coordinate for the linecut. If not specified, the value from the object's
            attribute will be used.
        window : tuple or None, optional
            Integration window extents around the central coordinate. If not specified, the value
            from the object's attribute will be used.
        axis : int or None, optional
            The axis along which to perform the linecut. If not specified, the value from the
            object's attribute will be used.
        verbose : bool
            Enables printout of linecut axis and integrated axes. Default False.

        Returns
        -------
        integrated_data : :class:`nexusformat.nexus.NXdata`
            1D linecut data after integration.

        """

        # Extract necessary attributes from the object
        data = self.data
        center = center if center is not None else self.center
        self.set_center(center)
        window = window if window is not None else self.window
        self.set_window(window, axis, verbose)

        # Convert the center to a tuple of floats
        center = tuple(float(c) for c in center)

        # Calculate the start and stop indices for slicing the data
        start = np.subtract(center, window)
        stop = np.add(center, window)
        slice_obj = tuple(slice(s, e) for s, e in zip(start, stop))
        self.integration_window = slice_obj

        # Perform the data cut
        self.integration_volume = data[slice_obj]
        self.integration_volume.nxname = data.nxname

        # Perform integration along the integrated axes
        integrated_data = np.sum(self.integration_volume[self.integration_volume.signal].nxdata,
                                 axis=self.integrated_axes)

        # Create an NXdata object for the linecut data
        self.linecut = NXdata(NXfield(integrated_data, name=self.integration_volume.signal),
                              self.integration_volume[self.integration_volume.axes[self.axis]])
        self.linecut.nxname = self.integration_volume.nxname

        return self.linecut

    def highlight_integration_window(self, data=None, width=None, height=None, label=None, highlight_color='red', **kwargs):
        """
        Plots the integration window highlighted on the three principal 2D cross-sections of a 3D dataset.

        Parameters
        ----------
        data : array-like, optional
            The 3D dataset to visualize. If not provided, uses `self.data`.
        width : float, optional
            Width of the visible x-axis range in each subplot. Used to zoom in on the integration region.
        height : float, optional
            Height of the visible y-axis range in each subplot. Used to zoom in on the integration region.
        label : str, optional
            Label for the rectangle patch marking the integration window, used in the legend.
        highlight_color : str, optional
            Color of the rectangle edges highlighting the integration window. Default is 'red'.
        **kwargs : dict, optional
            Additional keyword arguments passed to `plot_slice` for customizing the plot (e.g., colormap, vmin, vmax).

        Returns
        -------
        p1, p2, p3 : matplotlib.collections.QuadMesh
            The plotted QuadMesh objects for the three cross-sections:
            XY at fixed Z, XZ at fixed Y, and YZ at fixed X.

        """
        data = self.data if data is None else data
        center = self.center
        window = self.window

        # Create a figure and subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Plot cross-section 1
        slice_obj = [slice(None)] * data.ndim
        slice_obj[2] = center[2]

        p1 = plot_slice(data[slice_obj],
                        X=data[data.axes[0]],
                        Y=data[data.axes[1]],
                        ax=axes[0],
                        **kwargs)
        ax = axes[0]
        rect_diffuse = patches.Rectangle(
            (center[0] - window[0],
             center[1] - window[1]),
            2 * window[0], 2 * window[1],
            linewidth=1, edgecolor=highlight_color,
            facecolor='none', transform=p1.get_transform(), label=label,
        )
        ax.add_patch(rect_diffuse)

        if 'xlim' not in kwargs and width is not None:
            ax.set(xlim=(center[0]-width/2,center[0]+width/2))
        if 'ylim' not in kwargs and height is not None:
            ax.set(ylim=(center[1]-height/2,center[1]+height/2))

        # Plot cross-section 2
        slice_obj = [slice(None)] * data.ndim
        slice_obj[1] = center[1]

        p2 = plot_slice(data[slice_obj],
                        X=data[data.axes[0]],
                        Y=data[data.axes[2]],
                        ax=axes[1],
                        **kwargs)
        ax = axes[1]
        rect_diffuse = patches.Rectangle(
            (center[0] - window[0],
             center[2] - window[2]),
            2 * window[0], 2 * window[2],
            linewidth=1, edgecolor=highlight_color,
            facecolor='none', transform=p2.get_transform(), label=label,
        )
        ax.add_patch(rect_diffuse)

        if 'xlim' not in kwargs and width is not None:
            ax.set(xlim=(center[0]-width/2,center[0]+width/2))
        if 'ylim' not in kwargs and height is not None:
            ax.set(ylim=(center[2]-height/2,center[2]+height/2))


        # Plot cross-section 3
        slice_obj = [slice(None)] * data.ndim
        slice_obj[0] = center[0]

        p3 = plot_slice(data[slice_obj],
                        X=data[data.axes[1]],
                        Y=data[data.axes[2]],
                        ax=axes[2],
                        **kwargs)
        ax = axes[2]
        rect_diffuse = patches.Rectangle(
            (center[1] - window[1],
             center[2] - window[2]),
            2 * window[1], 2 * window[2],
            linewidth=1, edgecolor=highlight_color,
            facecolor='none', transform=p3.get_transform(), label=label,
        )
        ax.add_patch(rect_diffuse)

        if 'xlim' not in kwargs and width is not None:
            ax.set(xlim=(center[1]-width/2,center[1]+width/2))
        if 'ylim' not in kwargs and height is not None:
            ax.set(ylim=(center[2]-height/2,center[2]+height/2))


        # Adjust subplot padding
        fig.subplots_adjust(wspace=0.5)

        if label is not None:
            [ax.legend() for ax in axes]

        plt.show()

        return p1, p2, p3

    def plot_integration_window(self, **kwargs):
        """
        Plots the three principal cross-sections of the integration volume on a single figure.

        Parameters
        ----------
        **kwargs : keyword arguments, optional
            Additional keyword arguments to customize the plot.
        """
        data = self.integration_volume
        center = self.center

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Plot cross section 1
        slice_obj = [slice(None)] * data.ndim
        slice_obj[2] = center[2]
        p1 = plot_slice(data[slice_obj],
                        X=data[data.axes[0]],
                        Y=data[data.axes[1]],
                        ax=axes[0],
                        **kwargs)
        axes[0].set_aspect(len(data[data.axes[0]].nxdata) / len(data[data.axes[1]].nxdata))

        # Plot cross section 2
        slice_obj = [slice(None)] * data.ndim
        slice_obj[1] = center[1]
        p3 = plot_slice(data[slice_obj],
                        X=data[data.axes[0]],
                        Y=data[data.axes[2]],
                        ax=axes[1],
                        **kwargs)
        axes[1].set_aspect(len(data[data.axes[0]].nxdata) / len(data[data.axes[2]].nxdata))

        # Plot cross section 3
        slice_obj = [slice(None)] * data.ndim
        slice_obj[0] = center[0]
        p2 = plot_slice(data[slice_obj],
                        X=data[data.axes[1]],
                        Y=data[data.axes[2]],
                        ax=axes[2],
                        **kwargs)
        axes[2].set_aspect(len(data[data.axes[1]].nxdata) / len(data[data.axes[2]].nxdata))

        # Adjust subplot padding
        fig.subplots_adjust(wspace=0.3)

        plt.show()

        return p1, p2, p3


data = load_data(r'C:\Users\steve\OneDrive\Documents\UCSB\Projects\RECd3P3\mini\15\3rot_hkli.nxs')
s = Scissors(data)
s.cut_data(center=(1,1,0), window=(0.5,0.2,0.2))
print(s.highlight_integration_window(vmin=0, vmax=100, width=10, height=10))

print(data.shape)