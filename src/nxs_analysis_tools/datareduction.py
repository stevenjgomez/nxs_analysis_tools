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
__all__ = ['load_data', 'load_transform', 'plot_slice', 'Scissors',
           'reciprocal_lattice_params', 'rotate_data', 'rotate_data_2D',
           'convert_to_inverse_angstroms', 'array_to_nxdata', 'Padder',
           'rebin_nxdata', 'rebin_3d', 'rebin_1d']


def load_data(path, print_tree=True):
    """
    Load data from a NeXus file at a specified path. It is assumed that the data follows the CHESS
    file structure (i.e., root/entry/data/counts, etc.).

    Parameters
    ----------
    path : str
        The path to the NeXus data file.

    print_tree : bool, optional
        Whether to print the data tree upon loading. Default True.

    Returns
    -------
    data : nxdata object
        The loaded data stored in a nxdata object.

    """

    g = nxload(path)
    try:
        print(g.entry.data.tree) if print_tree else None
    except NeXusError:
        pass

    return g.entry.data


def load_transform(path, print_tree=True):
    """
    Load transform data from an nxrefine output file.

    Parameters
    ----------
    path : str
        The path to the transform data file.

    print_tree : bool, optional
        If True, prints the NeXus data tree upon loading. Default is True.

    Returns
    -------
    data : NXdata
        The loaded transform data as an NXdata object.
    """

    g = nxload(path)

    data = NXdata(NXfield(g.entry.transform.data.nxdata.transpose(2, 1, 0), name='counts'),
                  (g.entry.transform.Qh, g.entry.transform.Qk, g.entry.transform.Ql))

    print(data.tree) if print_tree else None

    return data


def array_to_nxdata(array, data_template, signal_name=None):
    """
    Create an NXdata object from an input array and an NXdata template,
    with an optional signal name.

    Parameters
    ----------
    array : array-like
        The data array to be included in the NXdata object.

    data_template : NXdata
        An NXdata object serving as a template, which provides information
        about axes and other metadata.

    signal_name : str, optional
        The name of the signal within the NXdata object. If not provided,
        the signal name is inherited from the data_template.

    Returns
    -------
    NXdata
        An NXdata object containing the input data array and associated axes
        based on the template.
    """
    d = data_template
    if signal_name is None:
        signal_name = d.signal
    return NXdata(NXfield(array, name=signal_name), d.nxaxes)


def rebin_3d(array):
    """
    Rebins a 3D NumPy array by a factor of 2 along each dimension.

    This function reduces the size of the input array by averaging over non-overlapping
    2x2x2 blocks. Each dimension of the input array must be divisible by 2.

    Parameters
    ----------
    array : np.ndarray
        A 3-dimensional NumPy array to be rebinned.

    Returns
    -------
    np.ndarray
        A rebinned array with shape (N//2, M//2, L//2) if the original shape was (N, M, L).
    """

    # Ensure the array shape is divisible by 2 in each dimension
    shape = array.shape
    if any(dim % 2 != 0 for dim in shape):
        raise ValueError("Each dimension of the array must be divisible by 2 to rebin.")

    # Reshape the array to group the data into 2x2x2 blocks
    reshaped = array.reshape(shape[0] // 2, 2, shape[1] // 2, 2, shape[2] // 2, 2)

    # Average over the 2x2x2 blocks
    rebinned = reshaped.mean(axis=(1, 3, 5))

    return rebinned


def rebin_1d(array):
    """
    Rebins a 1D NumPy array by a factor of 2.

    This function reduces the size of the input array by averaging over non-overlapping
    pairs of elements. The input array length must be divisible by 2.

    Parameters
    ----------
    array : np.ndarray
        A 1-dimensional NumPy array to be rebinned.

    Returns
    -------
    np.ndarray
        A rebinned array with length N//2 if the original length was N.
    """

    # Ensure the array length is divisible by 2
    if len(array) % 2 != 0:
        raise ValueError("The length of the array must be divisible by 2 to rebin.")

    # Reshape the array to group elements into pairs
    reshaped = array.reshape(len(array) // 2, 2)

    # Average over the pairs
    rebinned = reshaped.mean(axis=1)

    return rebinned


def rebin_nxdata(data):
    """
    Rebins the signal and axes of an NXdata object by a factor of 2 along each dimension.

    This function first checks each axis of the input `NXdata` object:
      - If the axis has an odd number of elements, the last element is excluded before rebinning.
      - Then, each axis is rebinned using `rebin_1d`.

    The signal array is similarly cropped to remove the last element along any dimension
    with an odd shape, and then the data is averaged over 2x2x... blocks using the same
    `rebin_1d` method (assumed to apply across 1D slices).

    Parameters
    ----------
    data : NXdata
        The NeXus data group containing the signal and axes to be rebinned.

    Returns
    -------
    NXdata
        A new NXdata object with signal and axes rebinned by a factor of 2 along each dimension.
    """
    # First, rebin axes
    new_axes = []
    for i in range(len(data.shape)):
        if data.shape[i] % 2 == 1:
            new_axes.append(
                NXfield(
                    rebin_1d(data.nxaxes[i].nxdata[:-1]),
                    name=data.axes[i]
                )
            )
        else:
            new_axes.append(
                NXfield(
                    rebin_1d(data.nxaxes[i].nxdata[:]),
                    name=data.axes[i]
                )
            )

    # Second, rebin signal
    data_arr = data.nxsignal.nxdata

    # Crop the array if the shape is odd in any direction
    slice_obj = []
    for i, dim in enumerate(data_arr.shape):
        if dim % 2 == 1:
            slice_obj.append(slice(0, dim - 1))
        else:
            slice_obj.append(slice(None))

    data_arr = data_arr[tuple(slice_obj)]

    # Perform actual rebinning
    data_arr = rebin_3d(data_arr)

    return NXdata(NXfield(data_arr, name=data.signal),
                  tuple([axis for axis in new_axes])
                  )


def plot_slice(data, X=None, Y=None, sum_axis=None, transpose=False, vmin=None, vmax=None,
               skew_angle=90, ax=None, xlim=None, ylim=None,
               xticks=None, yticks=None, cbar=True, logscale=False,
               symlogscale=False, cmap='viridis', linthresh=1,
               title=None, mdheading=None, cbartitle=None,
               **kwargs):
    """
    Plot a 2D slice of the provided dataset, with optional transformations
    and customizations.

    Parameters
    ----------
    data : :class:`nexusformat.nexus.NXdata` or ndarray
        The dataset to plot. Can be an `NXdata` object or a `numpy` array.

    X : NXfield, optional
        The X axis values. If None, a default range from 0 to the number of
         columns in `data` is used.

    Y : NXfield, optional
        The Y axis values. If None, a default range from 0 to the number of
         rows in `data` is used.

    sum_axis : int, optional
        If the input data is 3D, this specifies the axis to sum over in order
        to reduce the data to 2D for plotting. Required if `data` has three dimensions.

    transpose : bool, optional
        If True, transpose the dataset and its axes before plotting.
        Default is False.

    vmin : float, optional
        The minimum value for the color scale. If not provided, the minimum
         value of the dataset is used.

    vmax : float, optional
        The maximum value for the color scale. If not provided, the maximum
         value of the dataset is used.

    skew_angle : float, optional
        The angle in degrees to shear the plot. Default is 90 degrees (no skew).

    ax : matplotlib.axes.Axes, optional
        The `matplotlib` axis to plot on. If None, a new figure and axis will
         be created.

    xlim : tuple, optional
        The limits for the x-axis. If None, the limits are set automatically
         based on the data.

    ylim : tuple, optional
        The limits for the y-axis. If None, the limits are set automatically
         based on the data.

    xticks : float or list of float, optional
        The major tick interval or specific tick locations for the x-axis.
         Default is to use a minor tick interval of 1.

    yticks : float or list of float, optional
        The major tick interval or specific tick locations for the y-axis.
        Default is to use a minor tick interval of 1.

    cbar : bool, optional
        Whether to include a colorbar. Default is True.

    logscale : bool, optional
        Whether to use a logarithmic color scale. Default is False.

    symlogscale : bool, optional
        Whether to use a symmetrical logarithmic color scale. Default is False.

    cmap : str or Colormap, optional
        The colormap to use for the plot. Default is 'viridis'.

    linthresh : float, optional
        The linear threshold for symmetrical logarithmic scaling. Default is 1.

    title : str, optional
        The title for the plot. If None, no title is set.

    mdheading : str, optional
        A Markdown heading to display above the plot. If 'None' or not provided,
         no heading is displayed.

    cbartitle : str, optional
        The title for the colorbar. If None, the colorbar label will be set to
         the name of the signal.

    **kwargs
        Additional keyword arguments passed to `pcolormesh`.

    Returns
    -------
    p : :class:`matplotlib.collections.QuadMesh`
        The `matplotlib` QuadMesh object representing the plotted data.
    """
    is_array = False
    is_nxdata = False

    if isinstance(data, np.ndarray):
        is_array = True
    elif isinstance(data, (NXdata, NXfield)):
        is_nxdata = True
    else:
        raise TypeError(f"Unexpected data type: {type(data)}. "
                        f"Supported types are np.ndarray and NXdata.")

    # If three-dimensional, demand sum_axis to reduce to two dimensions.
    if is_array and len(data.shape) == 3:
        assert sum_axis is not None, "sum_axis must be specified when data is 3D."

        data = data.sum(axis=sum_axis)

    if is_nxdata and len(data.shape) == 3:
        assert sum_axis is not None, "sum_axis must be specified when data is 3D."

        arr = data.nxsignal.nxdata
        arr = arr.sum(axis=sum_axis)

        # Create a 2D template from the original nxdata
        slice_obj = [slice(None)] * len(data.shape)
        slice_obj[sum_axis] = 0

        # Use the 2D template to create a new nxdata
        data = array_to_nxdata(arr, data[slice_obj])

    if is_array:
        if X is None:
            X = NXfield(np.linspace(0, data.shape[0], data.shape[0]), name='x')
        if Y is None:
            Y = NXfield(np.linspace(0, data.shape[1], data.shape[1]), name='y')
        if transpose:
            X, Y = Y, X
            data = data.transpose()
        data = NXdata(NXfield(data, name='value'), (X, Y))
        data_arr = data[data.signal].nxdata.transpose()
    elif is_nxdata:
        if X is None:
            X = data[data.axes[0]]
        if Y is None:
            Y = data[data.axes[1]]
        if transpose:
            X, Y = Y, X
            data = data.transpose()
        data_arr = data[data.signal].nxdata.transpose()

    # Display Markdown heading
    if mdheading is None:
        pass
    elif mdheading == "None":
        display(Markdown('### Figure'))
    else:
        display(Markdown('### Figure - ' + mdheading))

    # Inherit axes if user provides some
    if ax is not None:
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
    p.set_transform(t
                    + Affine2D().translate(-ymin * np.sin(skew_angle_adj * np.pi / 180), 0)
                    + ax.transData)

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

    # Automatically set tick locations, only if NXdata or if X,Y axes are provided for an array
    if is_nxdata or (is_array and (X is not None and Y is not None)):
        # Add default minor ticks on x
        ax.xaxis.set_minor_locator(MultipleLocator(1))

        if xticks is not None:
            # Use user provided values
            ax.xaxis.set_major_locator(MultipleLocator(xticks))

        # Add default minor ticks on y
        ax.yaxis.set_minor_locator(MultipleLocator(1))

        if yticks is not None:
            # Use user provided values
            ax.yaxis.set_major_locator(MultipleLocator(yticks))

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

    def highlight_integration_window(self, data=None, width=None, height=None,
                                     label=None, highlight_color='red', **kwargs):
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
            Additional keyword arguments passed to `plot_slice` for customizing the plot (e.g., cmap, vmin, vmax).

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
            ax.set(xlim=(center[0] - width / 2, center[0] + width / 2))
        if 'ylim' not in kwargs and height is not None:
            ax.set(ylim=(center[1] - height / 2, center[1] + height / 2))

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
            ax.set(xlim=(center[0] - width / 2, center[0] + width / 2))
        if 'ylim' not in kwargs and height is not None:
            ax.set(ylim=(center[2] - height / 2, center[2] + height / 2))

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

        # If width and height are provided, center the view on the linecut area
        if 'xlim' not in kwargs and width is not None:
            ax.set(xlim=(center[1] - width / 2, center[1] + width / 2))
        if 'ylim' not in kwargs and height is not None:
            ax.set(ylim=(center[2] - height / 2, center[2] + height / 2))

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

        # Plot cross-section 1
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

        # Plot cross-section 3
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
    """
    Calculate the reciprocal lattice parameters from the given direct lattice parameters.

    Parameters
    ----------
    lattice_params : tuple
        A tuple containing the direct lattice parameters (a, b, c, alpha, beta, gamma), where
        a, b, and c are the magnitudes of the lattice vectors, and alpha, beta, and gamma are the
        angles between them in degrees.

    Returns
    -------
    tuple
        A tuple containing the reciprocal lattice parameters (a*, b*, c*, alpha*, beta*, gamma*),
        where a*, b*, and c* are the magnitudes of the reciprocal lattice vectors, and alpha*,
        beta*, and gamma* are the angles between them in degrees.
    """
    a_mag, b_mag, c_mag, alpha, beta, gamma = lattice_params
    # Convert angles to radians
    alpha = np.deg2rad(alpha)
    beta = np.deg2rad(beta)
    gamma = np.deg2rad(gamma)

    # Calculate unit cell volume
    V = a_mag * b_mag * c_mag * np.sqrt(
        1 - np.cos(alpha) ** 2 - np.cos(beta) ** 2 - np.cos(gamma) ** 2
        + 2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma)
    )

    # Calculate reciprocal lattice parameters
    a_star = (b_mag * c_mag * np.sin(alpha)) / V
    b_star = (a_mag * c_mag * np.sin(beta)) / V
    c_star = (a_mag * b_mag * np.sin(gamma)) / V
    alpha_star = np.rad2deg(np.arccos((np.cos(beta) * np.cos(gamma) - np.cos(alpha))
                                      / (np.sin(beta) * np.sin(gamma))))
    beta_star = np.rad2deg(np.arccos((np.cos(alpha) * np.cos(gamma) - np.cos(beta))
                                     / (np.sin(alpha) * np.sin(gamma))))
    gamma_star = np.rad2deg(np.arccos((np.cos(alpha) * np.cos(beta) - np.cos(gamma))
                                      / (np.sin(alpha) * np.sin(beta))))

    return a_star, b_star, c_star, alpha_star, beta_star, gamma_star


def convert_to_inverse_angstroms(data, lattice_params):
    """
    Convert the axes of a 3D NXdata object from reciprocal lattice units (r.l.u.)
    to inverse angstroms using provided lattice parameters.

    Parameters
    ----------
    data : :class:`nexusformat.nexus.NXdata`
        A 3D NXdata object with axes in reciprocal lattice units.

    lattice_params : tuple of float
        A tuple containing the real-space lattice parameters
        (a, b, c, alpha, beta, gamma) in angstroms and degrees.

    Returns
    -------
    NXdata
        A new NXdata object with axes scaled to inverse angstroms.
    """

    a_, b_, c_, al_, be_, ga_ = reciprocal_lattice_params(lattice_params)

    new_data = data.nxsignal
    a_star = NXfield(data.nxaxes[0].nxdata * a_, name='a_star')
    b_star = NXfield(data.nxaxes[1].nxdata * b_, name='b_star')
    c_star = NXfield(data.nxaxes[2].nxdata * c_, name='c_star')

    return NXdata(new_data, (a_star, b_star, c_star))


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
        Enables printout of rotation progress. If set to True, information
        about each rotation slice will be printed to the console, indicating
        the axis being rotated and the corresponding coordinate value.
        Defaults to False.


    Returns
    -------
    rotated_data : :class:`nexusformat.nexus.NXdata`
        Rotated data as an NXdata object.
    """
    # Define output array
    output_array = np.zeros(data[data.signal].shape)

    # Define shear transformation
    skew_angle_adj = 90 - lattice_angle
    t = Affine2D()
    # Scale y-axis to preserve norm while shearing
    t += Affine2D().scale(1, np.cos(skew_angle_adj * np.pi / 180))
    # Shear along x-axis
    t += Affine2D().skew_deg(skew_angle_adj, 0)
    # Return to original y-axis scaling
    t += Affine2D().scale(1, np.cos(skew_angle_adj * np.pi / 180)).inverted()

    # Iterate over all layers perpendicular to the rotation axis
    for i in range(len(data[data.axes[rotation_axis]])):
        # Print progress
        if printout:
            print(f'\rRotating {data.axes[rotation_axis]}'
                  f'={data[data.axes[rotation_axis]][i]}...                      ',
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

        # Add padding to avoid data cutoff during rotation
        p = Padder(sliced_data)
        padding = tuple(len(sliced_data[axis]) for axis in sliced_data.axes)
        counts = p.pad(padding)
        counts = p.padded[p.padded.signal]

        # Perform shear operation
        counts_skewed = ndimage.affine_transform(counts,
                                                 t.inverted().get_matrix()[:2, :2],
                                                 offset=[counts.shape[0] / 2
                                                         * np.sin(skew_angle_adj * np.pi / 180),
                                                         0],
                                                 order=0,
                                                 )
        # Scale data based on skew angle
        scale1 = np.cos(skew_angle_adj * np.pi / 180)
        counts_scaled1 = ndimage.affine_transform(counts_skewed,
                                                  Affine2D().scale(scale1, 1).get_matrix()[:2, :2],
                                                  offset=[(1 - scale1) * counts.shape[0] / 2, 0],
                                                  order=0,
                                                  )
        # Scale data based on ratio of array dimensions
        scale2 = counts.shape[0] / counts.shape[1]
        counts_scaled2 = ndimage.affine_transform(counts_scaled1,
                                                  Affine2D().scale(scale2, 1).get_matrix()[:2, :2],
                                                  offset=[(1 - scale2) * counts.shape[0] / 2, 0],
                                                  order=0,
                                                  )

        # Perform rotation
        counts_rotated = ndimage.rotate(counts_scaled2, rotation_angle, reshape=False, order=0)

        # Undo scaling 2
        counts_unscaled2 = ndimage.affine_transform(counts_rotated,
                                                    Affine2D().scale(
                                                        scale2, 1
                                                    ).inverted().get_matrix()[:2, :2],
                                                    offset=[-(1 - scale2) * counts.shape[
                                                        0] / 2 / scale2, 0],
                                                    order=0,
                                                    )
        # Undo scaling 1
        counts_unscaled1 = ndimage.affine_transform(counts_unscaled2,
                                                    Affine2D().scale(
                                                        scale1, 1
                                                    ).inverted().get_matrix()[:2, :2],
                                                    offset=[-(1 - scale1) * counts.shape[
                                                        0] / 2 / scale1, 0],
                                                    order=0,
                                                    )
        # Undo shear operation
        counts_unskewed = ndimage.affine_transform(counts_unscaled1,
                                                   t.get_matrix()[:2, :2],
                                                   offset=[
                                                       (-counts.shape[0] / 2
                                                        * np.sin(skew_angle_adj * np.pi / 180)),
                                                       0],
                                                   order=0,
                                                   )
        # Remove padding
        counts_unpadded = p.unpad(counts_unskewed)

        # Write current slice
        if rotation_axis == 0:
            output_array[i, :, :] = counts_unpadded
        elif rotation_axis == 1:
            output_array[:, i, :] = counts_unpadded
        elif rotation_axis == 2:
            output_array[:, :, i] = counts_unpadded
    print('\nDone.')
    return NXdata(NXfield(output_array, name=p.padded.signal),
                  (data[data.axes[0]], data[data.axes[1]], data[data.axes[2]]))


def rotate_data_2D(data, lattice_angle, rotation_angle):
    """
    Rotates 2D data.

    Parameters
    ----------
    data : :class:`nexusformat.nexus.NXdata`
        Input data.
    lattice_angle : float
        Angle between the two in-plane lattice axes in degrees.
    rotation_angle : float
        Angle of rotation in degrees.


    Returns
    -------
    rotated_data : :class:`nexusformat.nexus.NXdata`
        Rotated data as an NXdata object.
    """

    # Define transformation
    skew_angle_adj = 90 - lattice_angle
    t = Affine2D()
    # Scale y-axis to preserve norm while shearing
    t += Affine2D().scale(1, np.cos(skew_angle_adj * np.pi / 180))
    # Shear along x-axis
    t += Affine2D().skew_deg(skew_angle_adj, 0)
    # Return to original y-axis scaling
    t += Affine2D().scale(1, np.cos(skew_angle_adj * np.pi / 180)).inverted()

    # Add padding to avoid data cutoff during rotation
    p = Padder(data)
    padding = tuple(len(data[axis]) for axis in data.axes)
    counts = p.pad(padding)
    counts = p.padded[p.padded.signal]

    # Perform shear operation
    counts_skewed = ndimage.affine_transform(counts,
                                             t.inverted().get_matrix()[:2, :2],
                                             offset=[counts.shape[0] / 2
                                                     * np.sin(skew_angle_adj * np.pi / 180), 0],
                                             order=0,
                                             )
    # Scale data based on skew angle
    scale1 = np.cos(skew_angle_adj * np.pi / 180)
    counts_scaled1 = ndimage.affine_transform(counts_skewed,
                                              Affine2D().scale(scale1, 1).get_matrix()[:2, :2],
                                              offset=[(1 - scale1) * counts.shape[0] / 2, 0],
                                              order=0,
                                              )
    # Scale data based on ratio of array dimensions
    scale2 = counts.shape[0] / counts.shape[1]
    counts_scaled2 = ndimage.affine_transform(counts_scaled1,
                                              Affine2D().scale(scale2, 1).get_matrix()[:2, :2],
                                              offset=[(1 - scale2) * counts.shape[0] / 2, 0],
                                              order=0,
                                              )
    # Perform rotation
    counts_rotated = ndimage.rotate(counts_scaled2, rotation_angle, reshape=False, order=0)

    # Undo scaling 2
    counts_unscaled2 = ndimage.affine_transform(counts_rotated,
                                                Affine2D().scale(
                                                    scale2, 1
                                                ).inverted().get_matrix()[:2, :2],
                                                offset=[-(1 - scale2) * counts.shape[
                                                    0] / 2 / scale2, 0],
                                                order=0,
                                                )
    # Undo scaling 1
    counts_unscaled1 = ndimage.affine_transform(counts_unscaled2,
                                                Affine2D().scale(
                                                    scale1, 1
                                                ).inverted().get_matrix()[:2, :2],
                                                offset=[-(1 - scale1) * counts.shape[
                                                    0] / 2 / scale1, 0],
                                                order=0,
                                                )
    # Undo shear operation
    counts_unskewed = ndimage.affine_transform(counts_unscaled1,
                                               t.get_matrix()[:2, :2],
                                               offset=[
                                                   (-counts.shape[0] / 2
                                                    * np.sin(skew_angle_adj * np.pi / 180)),
                                                   0],
                                               order=0,
                                               )
    # Remove padding
    counts_unpadded = p.unpad(counts_unskewed)

    print('\nDone.')
    return NXdata(NXfield(counts_unpadded, name=p.padded.signal),
                  (data[data.axes[0]], data[data.axes[1]]))


class Padder:
    """
    A class to symmetrically pad and unpad datasets with a region of zeros.

    Attributes
    ----------
    data : NXdata or None
        The input data to be padded.
    padded : NXdata or None
        The padded data with symmetric zero padding.
    padding : tuple or None
        The number of zero-value pixels added along each edge of the array.
    steps : tuple or None
        The step sizes along each axis of the dataset.
    maxes : tuple or None
        The maximum values along each axis of the dataset.

    Methods
    -------
    set_data(data)
        Set the input data for padding.
    pad(padding)
        Symmetrically pads the data with zero values.
    save(fout_name=None)
        Saves the padded dataset to a .nxs file.
    unpad(data)
        Removes the padded region from the data.
    """

    def __init__(self, data=None):
        """
        Initialize the Padder object.

        Parameters
        ----------
        data : NXdata, optional
            The input data to be padded. If provided, the `set_data` method
            is called to set the data.
        """
        self.maxes = None
        self.steps = None
        self.data = None
        self.padded = None
        self.padding = None
        if data is not None:
            self.set_data(data)

    def set_data(self, data):
        """
        Set the input data for padding.

        Parameters
        ----------
        data : NXdata
            The input data to be padded.
        """
        self.data = data

        self.steps = tuple((data[axis].nxdata[1] - data[axis].nxdata[0])
                           for axis in data.axes)

        # Absolute value of the maximum value; assumes the domain of the input
        # is symmetric (eg, -H_min = H_max)
        self.maxes = tuple(data[axis].nxdata.max() for axis in data.axes)

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
        data = self.data
        self.padding = padding

        padded_shape = tuple(data[data.signal].nxdata.shape[i]
                             + self.padding[i] * 2 for i in range(data.ndim))

        # Create padded dataset
        padded = np.zeros(padded_shape)

        slice_obj = [slice(None)] * data.ndim
        for i, _ in enumerate(slice_obj):
            slice_obj[i] = slice(self.padding[i], -self.padding[i], None)
        slice_obj = tuple(slice_obj)
        padded[slice_obj] = data[data.signal].nxdata

        padmaxes = tuple(self.maxes[i] + self.padding[i] * self.steps[i]
                         for i in range(data.ndim))

        padded = NXdata(NXfield(padded, name=data.signal),
                        tuple(NXfield(np.linspace(-padmaxes[i], padmaxes[i], padded_shape[i]),
                                      name=data.axes[i])
                              for i in range(data.ndim)))

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
        """
        slice_obj = [slice(None)] * data.ndim
        for i in range(data.ndim):
            slice_obj[i] = slice(self.padding[i], -self.padding[i], None)
        slice_obj = tuple(slice_obj)
        return data[slice_obj]


def load_discus_nxs(path):
    """
    Load .nxs format data from the DISCUS program (by T. Proffen and R. Neder)
    and convert it to the CHESS format.

    Parameters
    ----------
    path : str
        The file path to the .nxs file generated by DISCUS.

    Returns
    -------
    NXdata
        The data converted to the CHESS format, with axes labeled 'H', 'K', and 'L',
        and the signal labeled 'counts'.

    """
    filename = path
    root = nxload(filename)
    hlim, klim, llim = root.lower_limits
    hstep, kstep, lstep = root.step_sizes
    h = NXfield(np.linspace(hlim, -hlim, int(np.abs(hlim * 2) / hstep) + 1), name='H')
    k = NXfield(np.linspace(klim, -klim, int(np.abs(klim * 2) / kstep) + 1), name='K')
    l = NXfield(np.linspace(llim, -llim, int(np.abs(llim * 2) / lstep) + 1), name='L')
    data = NXdata(NXfield(root.data[:, :, :], name='counts'), (h, k, l))

    return data
