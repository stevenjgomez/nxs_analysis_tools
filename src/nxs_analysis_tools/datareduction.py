"""
Reduces scattering data into 2D and 1D datasets.
"""
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

# Specify items on which users are allowed to perform standalone imports
__all__ = ['load_data', 'plot_slice', 'Scissors', 'reciprocal_lattice_params', 'rotate_data', 'array_to_nxdata', 'Padder']


def load_data(path):
    """
    Load data from a specified path.

    Parameters
    ----------
    path : str
        The path to the data file.

    Returns
    -------
    data : nxdata object
        The loaded data stored in a nxdata object.

    """
    g = nxload(path)
    try:
        print(g.entry.data.tree)
    except NeXusError:
        pass

    return g.entry.data


def array_to_nxdata(array, data_template, signal_name='counts'):
    """
    Create an NXdata object from an input array and an NXdata template, with an optional signal name.

    Parameters
    ----------
    array : array-like
        The data array to be included in the NXdata object.

    data_template : NXdata
        An NXdata object serving as a template, which provides information about axes and other metadata.

    signal_name : str, optional
        The name of the signal within the NXdata object. If not provided,
        the default signal name 'counts' is used.

    Returns
    -------
    NXdata
        An NXdata object containing the input data array and associated axes based on the template.
    """
    d = data_template
    return NXdata(NXfield(array, name=signal_name), tuple([d[d.axes[i]] for i in range(len(d.axes))]))


def plot_slice(data, X=None, Y=None, transpose=False, vmin=None, vmax=None, skew_angle=90,
               ax=None, xlim=None, ylim=None, xticks=None, yticks=None, cbar=True, logscale=False,
               symlogscale=False, cmap='viridis', linthresh=1, title=None, mdheading=None, cbartitle=None,
               **kwargs):
    """
    Parameters
    ----------
    data : :class:`nexusformat.nexus.NXdata` object or ndarray
        The NXdata object containing the dataset to plot.

    X : NXfield, optional
        The X axis values. Default is first axis of `data`.

    Y : NXfield, optional
        The y axis values. Default is second axis of `data`.

    transpose : bool, optional
        If True, tranpose the dataset and its axes before plotting. Default is False.

    vmin : float, optional
        The minimum value to plot in the dataset.
        If not provided, the minimum of the dataset will be used.

    vmax : float, optional
        The maximum value to plot in the dataset.
        If not provided, the maximum of the dataset will be used.

    skew_angle : float, optional
        The angle to shear the plot in degrees. Defaults to 90 degrees (no skewing).

    ax : matplotlib.axes.Axes, optional
        An optional axis object to plot the heatmap onto.

    xlim : tuple, optional
        The limits of the x-axis. If not provided, the limits will be automatically set.

    ylim : tuple, optional
        The limits of the y-axis. If not provided, the limits will be automatically set.

    xticks : float, optional
        The major tick interval for the x-axis.
        If not provided, the function will use a default minor tick interval of 1.

    yticks : float, optional
        The major tick interval for the y-axis.
        If not provided, the function will use a default minor tick interval of 1.

    cbar : bool, optional
        Whether to include a colorbar in the plot. Defaults to True.

    logscale : bool, optional
        Whether to use a logarithmic color scale. Defaults to False.

    symlogscale : bool, optional
        Whether to use a symmetrical logarithmic color scale. Defaults to False.

    cmap : str or Colormap, optional
        The color map to use. Defaults to 'viridis'.

    linthresh : float, optional
        The linear threshold for the symmetrical logarithmic color scale. Defaults to 1.

    mdheading : str, optional
        A string containing the Markdown heading for the plot. Default `None`.

    Returns
    -------
    p : :class:`matplotlib.collections.QuadMesh`

        A :class:`matplotlib.collections.QuadMesh` object, to mimick behavior of
        :class:`matplotlib.pyplot.pcolormesh`.

    """
    if type(data) == np.ndarray:
        if X is None:
            X = NXfield(np.linspace(0, data.shape[1], data.shape[1]), name='x')
        if Y is None:
            Y = NXfield(np.linspace(0, data.shape[0], data.shape[0]), name='y')
        if transpose:
            X, Y = Y, X
            data = data.transpose()
        data = NXdata(NXfield(data, name='value'), (X, Y))
        data_arr = data
    elif type(data) == NXdata or type(data) == NXfield:
        if X is None:
            X = data[data.axes[0]]
        if Y is None:
            Y = data[data.axes[1]]
        if transpose:
            X, Y = Y, X
            data = data.transpose()
        data_arr = data[data.signal].nxdata.transpose()
    else:
        raise TypeError(f"Unexpected data type: {type(data)}. Supported types are np.ndarray and NXdata.")

    # Display Markdown heading
    if mdheading is None:
        pass
    elif mdheading == "None":
        display(Markdown('### Figure'))
    else:
        display(Markdown('### Figure - ' + mdheading))

    # Inherit axes if user provides some
    if ax is not None:
        ax = ax
        fig = ax.get_figure()
    # Otherwise set up some default axes
    else:
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])

    # If limits not provided, use extrema
    if vmin is None:
        vmin = data_arr.min()
    if vmax is None:
        vmax = data_arr.max()

    # Set norm (linear scale, logscale, or symlogscale)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)  # Default: linear scale

    if symlogscale:
        norm = colors.SymLogNorm(linthresh=linthresh, vmin=-1 * vmax, vmax=vmax)
    elif logscale:
        norm = colors.LogNorm(vmin=vmin, vmax=vmax)

    # Plot data
    p = ax.pcolormesh(X.nxdata, Y.nxdata, data_arr, shading='auto', norm=norm, cmap=cmap, **kwargs)

    ## Transform data to new coordinate system if necessary
    # Correct skew angle
    skew_angle_adj = 90 - skew_angle
    # Create blank 2D affine transformation
    t = Affine2D()
    # Scale y-axis to preserve norm while shearing
    t += Affine2D().scale(1, np.cos(skew_angle_adj * np.pi / 180))
    # Shear along x-axis
    t += Affine2D().skew_deg(skew_angle_adj, 0)
    # Return to original y-axis scaling
    t += Affine2D().scale(1, np.cos(skew_angle_adj * np.pi / 180)).inverted()
    ## Correct for x-displacement after shearing
    # If ylims provided, use those
    if ylim is not None:
        # Set ylims
        ax.set(ylim=ylim)
        ymin, ymax = ylim
    # Else, use current ylims
    else:
        ymin, ymax = ax.get_ylim()
    # Use ylims to calculate translation (necessary to display axes in correct position)
    p.set_transform(t + Affine2D().translate(-ymin * np.sin(skew_angle_adj * np.pi / 180), 0) + ax.transData)

    # Set x limits
    if xlim is not None:
        xmin, xmax = xlim
    else:
        xmin, xmax = ax.get_xlim()
    if skew_angle <= 90:
        ax.set(xlim=(xmin, xmax + (ymax - ymin) / np.tan((90 - skew_angle_adj) * np.pi / 180)))
    else:
        ax.set(xlim=(xmin - (ymax - ymin) / np.tan((skew_angle_adj - 90) * np.pi / 180), xmax))

    # Correct aspect ratio for the x/y axes after transformation
    ax.set(aspect=np.cos(skew_angle_adj * np.pi / 180))

    # Add tick marks all around
    ax.tick_params(direction='in', top=True, right=True, which='both')

    # Set tick locations
    if xticks is None:
        # Add default minor ticks
        ax.xaxis.set_minor_locator(MultipleLocator(1))
    else:
        # Otherwise use user provided values
        ax.xaxis.set_major_locator(MultipleLocator(xticks))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
    if yticks is None:
        # Add default minor ticks
        ax.yaxis.set_minor_locator(MultipleLocator(1))
    else:
        # Otherwise use user provided values
        ax.yaxis.set_major_locator(MultipleLocator(yticks))
        ax.yaxis.set_minor_locator(MultipleLocator(1))

    # Apply transform to tick marks
    for i in range(0, len(ax.xaxis.get_ticklines())):
        # Tick marker
        m = MarkerStyle(3)
        line = ax.xaxis.get_majorticklines()[i]
        if i % 2:
            # Top ticks (translation here makes their direction="in")
            m._transform.set(Affine2D().translate(0, -1) + Affine2D().skew_deg(skew_angle_adj, 0))
            # This first method shifts the top ticks horizontally to match the skew angle.
            # This does not look good in all cases.
            # line.set_transform(Affine2D().translate((ymax-ymin)*np.sin(skew_angle*np.pi/180),0) +
            #     line.get_transform())
            # This second method skews the tick marks in place and
            # can sometimes lead to them being misaligned.
            line.set_transform(line.get_transform())  # This does nothing
        else:
            # Bottom ticks
            m._transform.set(Affine2D().skew_deg(skew_angle_adj, 0))

        line.set_marker(m)

    for i in range(0, len(ax.xaxis.get_minorticklines())):
        m = MarkerStyle(2)
        line = ax.xaxis.get_minorticklines()[i]
        if i % 2:
            m._transform.set(Affine2D().translate(0, -1) + Affine2D().skew_deg(skew_angle_adj, 0))
        else:
            m._transform.set(Affine2D().skew_deg(skew_angle_adj, 0))

        line.set_marker(m)

    if cbar:
        colorbar = fig.colorbar(p)
        if cbartitle is None:
            colorbar.set_label(data.signal)

    ax.set(
        xlabel=X.nxname,
        ylabel=Y.nxname,
    )

    if title is not None:
        ax.set_title(title)

    # Return the quadmesh object
    return p


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
    data_cut : ndarray or None
        Data array after applying the integration window.
    integrated_axes : tuple or None
        Indices of axes that were integrated.
    linecut : :class:`nexusformat.nexus.NXdata` or None
        1D linecut data after integration.
    window_plane_slice_obj : list or None
        Slice object representing the integration window in the data array.

    Methods
    -------
    set_data(data)
        Set the input :class:`nexusformat.nexus.NXdata`
    get_data()
        Get the input :class:`nexusformat.nexus.NXdata`.
    set_center(center)
        Set the central coordinate for the linecut.
    set_window(window)
        Set the extents of the integration window.
    get_window()
        Get the extents of the integration window.
    cut_data(axis=None)
        Reduce data to a 1D linecut using the integration window.
    show_integration_window(label=None)
        Plot the integration window highlighted on a 2D heatmap of the full dataset.
    plot_window()
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
        self.center = tuple([float(i) for i in center])
        self.window = tuple([float(i) for i in window])
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
        self.center = tuple([float(i) for i in center])

    def set_window(self, window):
        """
        Set the extents of the integration window.

        Parameters
        ----------
        window : tuple
            Extents of the window for integration along each axis.
        """
        self.window = tuple([float(i) for i in window])

        # Determine the axis for integration
        self.axis = window.index(max(window))
        print("Linecut axis: " + str(self.data.axes[self.axis]))

        # Determine the integrated axes (axes other than the integration axis)
        self.integrated_axes = tuple(i for i in range(self.data.ndim) if i != self.axis)
        print("Integrated axes: " + str([self.data.axes[axis] for axis in self.integrated_axes]))

    def get_window(self):
        """
        Get the extents of the integration window.

        Returns
        -------
        tuple or None
            Extents of the integration window.
        """
        return self.window

    def cut_data(self, center=None, window=None, axis=None):
        """
        Reduces data to a 1D linecut with integration extents specified by the window about a central
        coordinate.

        Parameters
        -----------
        center : float or None, optional
            Central coordinate for the linecut. If not specified, the value from the object's
            attribute will be used.
        window : tuple or None, optional
            Integration window extents around the central coordinate. If not specified, the value
            from the object's attribute will be used.
        axis : int or None, optional
            The axis along which to perform the linecut. If not specified, the value from the
            object's attribute will be used.

        Returns
        --------
        integrated_data : :class:`nexusformat.nexus.NXdata`
            1D linecut data after integration.
        """

        # Extract necessary attributes from the object
        data = self.data
        center = center if center is not None else self.center
        self.set_center(center)
        window = window if window is not None else self.window
        self.set_window(window)
        axis = axis if axis is not None else self.axis

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
                              self.integration_volume[self.integration_volume.axes[axis]])
        self.linecut.nxname = self.integration_volume.nxname

        return self.linecut

    def highlight_integration_window(self, data=None, label=None, highlight_color='red', **kwargs):
        """
        Plots integration window highlighted on the three principal cross sections of the first
        temperature dataset.

        Parameters
        ----------
        data : array-like, optional
            The 2D heatmap dataset to plot. If not provided, the dataset stored in `self.data` will
            be used.
        label : str, optional
            The label for the integration window plot.
        highlight_color : str, optional
            The edge color used to highlight the integration window. Default is 'red'.
        **kwargs : keyword arguments, optional
            Additional keyword arguments to customize the plot.

        """
        data = self.data if data is None else data
        center = self.center
        window = self.window
        integrated_axes = self.integrated_axes

        # Create a figure and subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Plot cross section 1
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
            linewidth=1, edgecolor=highlight_color, facecolor='none', transform=p1.get_transform(), label=label,
        )
        ax.add_patch(rect_diffuse)

        # Plot cross section 2
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
            linewidth=1, edgecolor=highlight_color, facecolor='none', transform=p2.get_transform(), label=label,
        )
        ax.add_patch(rect_diffuse)

        # Plot cross section 3
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
            linewidth=1, edgecolor=highlight_color, facecolor='none', transform=p3.get_transform(), label=label,
        )
        ax.add_patch(rect_diffuse)

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
        axis = self.axis
        center = self.center
        window = self.window
        integrated_axes = self.integrated_axes

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


def reciprocal_lattice_params(lattice_params):
    a_mag, b_mag, c_mag, alpha, beta, gamma = lattice_params
    # Convert angles to radians
    alpha = np.deg2rad(alpha)
    beta = np.deg2rad(beta)
    gamma = np.deg2rad(gamma)

    # Calculate unit cell volume
    V = a_mag * b_mag * c_mag * np.sqrt(
        1 - np.cos(alpha) ** 2 - np.cos(beta) ** 2 - np.cos(gamma) ** 2 + 2 * np.cos(alpha) * np.cos(beta) * np.cos(
            gamma)
    )

    # Calculate reciprocal lattice parameters
    a_star = (b_mag * c_mag * np.sin(alpha)) / V
    b_star = (a_mag * c_mag * np.sin(beta)) / V
    c_star = (a_mag * b_mag * np.sin(gamma)) / V
    alpha_star = np.rad2deg(np.arccos((np.cos(beta) * np.cos(gamma) - np.cos(alpha)) / (np.sin(beta) * np.sin(gamma))))
    beta_star = np.rad2deg(np.arccos((np.cos(alpha) * np.cos(gamma) - np.cos(beta)) / (np.sin(alpha) * np.sin(gamma))))
    gamma_star = np.rad2deg(np.arccos((np.cos(alpha) * np.cos(beta) - np.cos(gamma)) / (np.sin(alpha) * np.sin(beta))))

    return a_star, b_star, c_star, alpha_star, beta_star, gamma_star


def rotate_data(data, lattice_angle, rotation_angle, rotation_axis, printout=False):
    """
    Rotates 3D data around a specified axis.

    Parameters
    ----------
    data : :class:`nexusformat.nexus.NXdata`
        Input data.
    lattice_angle : float
        Angle between the two in-plane lattice axes in degrees.
    rotation_angle : float
        Angle of rotation in degrees.
    rotation_axis : int
        Axis of rotation (0, 1, or 2).
    printout : bool, optional
        Enables printout of rotation progress. If set to True, information about each rotation slice will be printed
        to the console, indicating the axis being rotated and the corresponding
        coordinate value. Defaults to False.


    Returns
    -------
    rotated_data : :class:`nexusformat.nexus.NXdata`
        Rotated data as an NXdata object.
    """
    # Define output array
    output_array = np.zeros(data[data.signal].shape)

    # Define transformation
    skew_angle_adj = 90 - lattice_angle
    t = Affine2D()
    # Scale y-axis to preserve norm while shearing
    t += Affine2D().scale(1, np.cos(skew_angle_adj * np.pi / 180))
    # Shear along x-axis
    t += Affine2D().skew_deg(skew_angle_adj, 0)
    # Return to original y-axis scaling
    t += Affine2D().scale(1, np.cos(skew_angle_adj * np.pi / 180)).inverted()

    for i in range(len(data[data.axes[rotation_axis]])):
        if printout:
            print(f'\rRotating {data.axes[rotation_axis]}={data[data.axes[rotation_axis]][i]}...                      ',
                  end='', flush=True)
        # Identify current slice
        if rotation_axis == 0:
            sliced_data = data[i, :, :]
        elif rotation_axis == 1:
            sliced_data = data[:, i, :]
        elif rotation_axis == 2:
            sliced_data = data[:, :, i]
        else:
            sliced_data = None

        p = Padder(sliced_data)
        padding = tuple([len(sliced_data[axis]) for axis in sliced_data.axes])
        counts = p.pad(padding).counts

        counts_skewed = ndimage.affine_transform(counts,
                                                 t.inverted().get_matrix()[:2, :2],
                                                 offset=[counts.shape[0] / 2 * np.sin(skew_angle_adj * np.pi / 180), 0],
                                                 order=0,
                                                 )
        scale1 = np.cos(skew_angle_adj * np.pi / 180)
        counts_scaled1 = ndimage.affine_transform(counts_skewed,
                                                  Affine2D().scale(scale1, 1).get_matrix()[:2, :2],
                                                  offset=[(1 - scale1) * counts.shape[0] / 2, 0],
                                                  order=0,
                                                  )
        scale2 = counts.shape[0] / counts.shape[1]
        counts_scaled2 = ndimage.affine_transform(counts_scaled1,
                                                  Affine2D().scale(scale2, 1).get_matrix()[:2, :2],
                                                  offset=[(1 - scale2) * counts.shape[0] / 2, 0],
                                                  order=0,
                                                  )

        counts_rotated = ndimage.rotate(counts_scaled2, rotation_angle, reshape=False, order=0)

        counts_unscaled2 = ndimage.affine_transform(counts_rotated,
                                                    Affine2D().scale(scale2, 1).inverted().get_matrix()[:2, :2],
                                                    offset=[-(1 - scale2) * counts.shape[
                                                        0] / 2 / scale2, 0],
                                                    order=0,
                                                    )

        counts_unscaled1 = ndimage.affine_transform(counts_unscaled2,
                                                    Affine2D().scale(scale1,
                                                                     1).inverted().get_matrix()[:2, :2],
                                                    offset=[-(1 - scale1) * counts.shape[
                                                        0] / 2 / scale1, 0],
                                                    order=0,
                                                    )

        counts_unskewed = ndimage.affine_transform(counts_unscaled1,
                                                   t.get_matrix()[:2, :2],
                                                   offset=[
                                                       (-counts.shape[0] / 2 * np.sin(skew_angle_adj * np.pi / 180)),
                                                       0],
                                                   order=0,
                                                   )

        counts_unpadded = p.unpad(counts_unskewed)

        # Write current slice
        if rotation_axis == 0:
            output_array[i, :, :] = counts_unpadded
        elif rotation_axis == 1:
            output_array[:, i, :] = counts_unpadded
        elif rotation_axis == 2:
            output_array[:, :, i] = counts_unpadded
    print('\nDone.')
    return NXdata(NXfield(output_array, name='counts'),
                  (data[data.axes[0]], data[data.axes[1]], data[data.axes[2]]))


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
