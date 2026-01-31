"""
Tools for reducing data into 2D and 1D, and visualization functions for plotting and animating 
data.
"""
import os
import io
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from matplotlib.markers import MarkerStyle
from matplotlib.ticker import MultipleLocator
import matplotlib.animation as animation
from matplotlib import colors
from matplotlib import patches
from IPython.display import display, Markdown, HTML, Image
from nexusformat.nexus import NXfield, NXdata, nxload, NeXusError, NXroot, NXentry, nxsave
from scipy.ndimage import rotate, zoom

from .lineartransformations import ShearTransformer


# Specify items on which users are allowed to perform standalone imports
__all__ = ['load_data', 'load_transform', 'plot_slice', 'Scissors',
           'reciprocal_lattice_params', 'rotate_data',
           'convert_to_inverse_angstroms', 'array_to_nxdata', 'Padder',
           'rebin_nxdata', 'rebin_3d', 'rebin_1d', 'animate_slice_temp',
           'animate_slice_axis']


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


def load_transform(path, print_tree=True, use_nxlink=False):
    """
    Load transform data from an nxrefine output file.

    Parameters
    ----------
    path : str
        The path to the transform data file.

    print_tree : bool, optional
        If True, prints the NeXus data tree upon loading. Default is True.

    use_nxlink : bool, optional
        If True, maintains the NXlink defined in the data file, which references
        the raw data in the transform.nxs file. This saves memory when working with
        many datasets. In this case, the axes are in reverse order. Default is False.

    Returns
    -------
    data : NXdata
        The loaded transform data as an NXdata object.
    """

    root = nxload(path)

    if use_nxlink:
        data = root.entry.transform
    else:
        data = NXdata(NXfield(root.entry.transform.data.nxdata.transpose(2, 1, 0), name='counts'),
                      (root.entry.transform.Qh, root.entry.transform.Qk, root.entry.transform.Ql))

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
        signal_name = d.nxsignal.nxname
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

def rebin_2d(array):
    """
    Rebins a 2D NumPy array by a factor of 2 along each dimension.

    This function reduces the size of the input array by averaging over non-overlapping
    2x2 blocks. Each dimension of the input array must be divisible by 2.

    Parameters
    ----------
    array : np.ndarray
        A 2-dimensional NumPy array to be rebinned.

    Returns
    -------
    np.ndarray
        A rebinned array with shape (N//2, M//2) if the original shape was (N, M).
    """

    # Ensure the array shape is divisible by 2 in each dimension
    shape = array.shape
    if any(dim % 2 != 0 for dim in shape):
        raise ValueError("Each dimension of the array must be divisible by 2 to rebin.")

    # Reshape the array to group the data into 2x2 blocks
    reshaped = array.reshape(shape[0] // 2, 2, shape[1] // 2, 2)

    # Average over the 2x2 blocks
    rebinned = reshaped.mean(axis=(1, 3))

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
    with an odd shape, and then the data is averaged over 2x2x... blocks.

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
                    name=data.nxaxes[i].nxname
                )
            )
        else:
            new_axes.append(
                NXfield(
                    rebin_1d(data.nxaxes[i].nxdata[:]),
                    name=data.nxaxes[i].nxname
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
    if data.ndim == 3:
        data_arr = rebin_3d(data_arr)
    elif data.ndim == 2:
        data_arr = rebin_2d(data_arr)
    elif data.ndim == 1:
        data_arr = rebin_1d(data_arr)

    return NXdata(NXfield(data_arr, name=data.nxsignal.nxname),
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

    # Some logic to control the processing of the arrays
    is_array = False
    is_nxdata = False
    no_xy_provided = True

    # If X,Y not provided by user
    if X is not None and Y is not None:
        no_xy_provided = False

    # Examine data type to be plotted
    if isinstance(data, np.ndarray):
        is_array = True
    elif isinstance(data, (NXdata, NXfield)):
        is_nxdata = True
    else:
        raise TypeError(f"Unexpected data type: {type(data)}. "
                        f"Supported types are np.ndarray and NXdata.")

    # If three-dimensional, demand sum_axis to reduce to two dimensions.
    if data.ndim == 3:
        if sum_axis is None:
            raise ValueError("sum_axis must be specified when data.ndim == 3.")

        if is_array:
            data = data.sum(axis=sum_axis)
        elif is_nxdata:
            arr = data.nxsignal.nxdata
            arr = arr.sum(axis=sum_axis)

            # Create a 2D template from the original nxdata
            slice_obj = [slice(None)] * len(data.shape)
            slice_obj[sum_axis] = 0

            # Use the 2D template to create a new nxdata
            data = array_to_nxdata(arr, data[slice_obj])

    if data.ndim != 2:
        raise ValueError("Slice data must be 2D.")

    # If the data is of type ndarray, then convert to NXdata
    if is_array:
        # Convert X to NXfield if it is not already
        if X is None:
            X = NXfield(np.arange(data.shape[0]), name='x')
        elif isinstance(X, np.ndarray):
            X = NXfield(X, name='x')
        elif isinstance(X, NXfield):
            pass
        else:
            raise TypeError("X must be of type np.ndarray or NXdata")

        # Convert Y to NXfield if it is not already
        if Y is None:
            Y = NXfield(np.arange(data.shape[1]), name='y')
        elif isinstance(Y, np.ndarray):
            Y = NXfield(Y, name='y')
        elif isinstance(Y, NXfield):
            pass
        else:
            raise TypeError("Y must be of type np.ndarray or NXdata")

        if transpose:
            X, Y = Y, X
            data = data.transpose()

        data = NXdata(NXfield(data, name='value'), (X, Y))
        data_arr = data.nxsignal.nxdata.transpose()
    # Otherwise, if data is of type NXdata, then decide whether to create axes,
    # use provided axes, or inherit axes.
    elif is_nxdata:
        if X is None:
            X = data.nxaxes[0]
        elif isinstance(X, np.ndarray):
            X = NXfield(X, name='x')
        elif isinstance(X, NXdata):
            pass
        if Y is None:
            Y = data.nxaxes[1]
        elif isinstance(Y, np.ndarray):
            Y = NXfield(Y, name='y')
        elif isinstance(Y, NXdata):
            pass

        # Transpose axes and data if specified
        if transpose:
            X, Y = Y, X
            data = data.transpose()

        data_arr = data.nxsignal.nxdata.transpose()

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
    t = ShearTransformer(skew_angle)

    # If ylims provided, use those
    if ylim is not None:
        # Set ylims
        ax.set(ylim=ylim)
        ymin, ymax = ylim
    # Else, use current ylims
    else:
        ymin, ymax = ax.get_ylim()
    # Use ylims to calculate translation (necessary to display axes in correct position)
    p.set_transform(t.t
                    + Affine2D().translate(-ymin * np.sin(t.shear_angle * np.pi / 180), 0)
                    + ax.transData)

    # Set x limits
    if xlim is not None:
        xmin, xmax = xlim
    else:
        xmin, xmax = ax.get_xlim()
    if skew_angle <= 90:
        ax.set(xlim=(xmin, xmax + (ymax - ymin) / np.tan((90 - t.shear_angle) * np.pi / 180)))
    else:
        ax.set(xlim=(xmin - (ymax - ymin) / np.tan((t.shear_angle - 90) * np.pi / 180), xmax))

    # Correct aspect ratio for the x/y axes after transformation
    ax.set(aspect=np.cos(t.shear_angle * np.pi / 180))


    # Automatically set tick locations, only if NXdata or if X,Y axes are provided for an array
    if is_nxdata or (is_array and (no_xy_provided == False)):
        # Add default minor ticks on x
        ax.xaxis.set_minor_locator(MultipleLocator(1))

        # Add tick marks all around
        ax.tick_params(direction='in', top=True, right=True, which='both')

        if xticks is not None:
            # Use user provided values
            ax.xaxis.set_major_locator(MultipleLocator(xticks))

        # Add default minor ticks on y
        ax.yaxis.set_minor_locator(MultipleLocator(1))

        if yticks is not None:
            # Use user provided values
            ax.yaxis.set_major_locator(MultipleLocator(yticks))
    else:
        # Add tick marks all around
        ax.tick_params(direction='in', top=True, right=True, which='major')

    # Apply transform to tick marks
    for i in range(0, len(ax.xaxis.get_ticklines())):
        # Tick marker
        m = MarkerStyle(3)
        line = ax.xaxis.get_majorticklines()[i]
        if i % 2:
            # Top ticks (translation here makes their direction="in")
            m._transform.set(Affine2D().translate(0, -1) + Affine2D().skew_deg(t.shear_angle, 0))
            # This first method shifts the top ticks horizontally to match the skew angle.
            # This does not look good in all cases.
            # line.set_transform(Affine2D().translate((ymax-ymin)*np.sin(skew_angle*np.pi/180),0) +
            #     line.get_transform())
            # This second method skews the tick marks in place and
            # can sometimes lead to them being misaligned.
            line.set_transform(line.get_transform())  # This does nothing
        else:
            # Bottom ticks
            m._transform.set(Affine2D().skew_deg(t.shear_angle, 0))

        line.set_marker(m)

    for i in range(0, len(ax.xaxis.get_minorticklines())):
        m = MarkerStyle(2)
        line = ax.xaxis.get_minorticklines()[i]
        if i % 2:
            m._transform.set(Affine2D().translate(0, -1) + Affine2D().skew_deg(t.shear_angle, 0))
        else:
            m._transform.set(Affine2D().skew_deg(t.shear_angle, 0))

        line.set_marker(m)

    if cbar:
        colorbar = fig.colorbar(p)
        if cbartitle is None:
            colorbar.set_label(data.nxsignal.nxname)

    ax.set(
        xlabel=X.nxname,
        ylabel=Y.nxname,
    )

    if title is not None:
        ax.set_title(title)

    # Return the quadmesh object
    return p

def animate_slice_temp(temp_dependence, slice_obj, ax=None, reverse_temps=False, interval=500, 
                       save_gif=False, filename='animation',  title=True, title_fmt='d', 
                       plot_slice_kwargs=None, ax_kwargs=None):
    """
    Animate 2D slices from a temperature-dependent dataset.

    Creates a matplotlib animation by extracting 2D slices from each dataset 
    in a TempDependence object and animating them in sequence by temperature.
    Optionally displays the animation inline and/or saves it as a GIF.

    Parameters
    ----------
    temp_dependence : nxs_analysis_tools.chess.TempDependence
        Object holding datasets at various temperatures.
    slice_obj : list of slice or None
        Slice object to apply to each dataset; None entries are treated as ':'.
    ax : matplotlib.axes.Axes, optional
        The axes object to plot on. If None, a new figure and axes will be created.
    reverse_temps : bool, optional
        If True, animates datasets with increasing temperature. Default is False.
    interval : int, optional
        Delay between frames in milliseconds. Default is 500.
    save_gif : bool, optional
        If True, saves the animation to a .gif file. Default is False.
    filename : str, optional
        Filename (without extension) for saved .gif. Default is 'animation'.
    title : bool, optional
        If True, displays the temperature in the title of each frame. Default is True.
    title_fmt : str, optional
        Format string for temperature values (e.g., '.2f' for 2 decimals). Default is 'd' (integer).
    plot_slice_kwargs : dict, optional
        Additional keyword arguments passed to `plot_slice`.
    ax_kwargs : dict, optional
        Keyword arguments passed to `ax.set`.

    Returns
    -------
    ani : matplotlib.animation.FuncAnimation
        The resulting animation object.
    """
    if ax is None:
        fig,ax = plt.subplots() # Generate a new figure and axis
    else:
        fig = ax.figure  # Get the figure from the provided axis


    if plot_slice_kwargs is None:
        plot_slice_kwargs = {}
    if ax_kwargs is None:
        ax_kwargs = {}

    # Normalize the slice object
    normalized_slice = [slice(None) if s is None else s for s in slice_obj]

    # Warn if colorbar is requested
    if plot_slice_kwargs.get('cbar', False):
        warnings.warn("Colorbar is not supported in animation and will be ignored.", UserWarning)
        plot_slice_kwargs['cbar'] = False
    elif 'cbar' not in plot_slice_kwargs.keys():
        plot_slice_kwargs['cbar'] = False

    def update(temp):
        ax.clear()
        dataset = temp_dependence.datasets[temp]
        plot_slice(dataset[tuple(normalized_slice)], ax=ax, **plot_slice_kwargs)
        ax.set(**ax_kwargs)

        if title:
            try:
                formatted_temp = f"{int(temp):{title_fmt}}"
            except ValueError:
                raise ValueError(f"Invalid title_fmt '{title_fmt}' for temperature value '{temp}'")
            ax.set(title=f'$T$={formatted_temp}')

    # Animate frames upon warming
    if reverse_temps:
        frames = temp_dependence.temperatures.copy()
    # Animate frames upon cooling (default)
    else:
        frames = temp_dependence.temperatures.copy()
        frames.reverse()
        

    ani = animation.FuncAnimation(fig, update,
                                  frames=frames,
                                  interval=interval, repeat=False)

    display(HTML(ani.to_jshtml()))

    if save_gif:
        gif_file = f'{filename}.gif'
        writer = animation.PillowWriter(fps=1000 / interval)
        ani.save(gif_file, writer=writer)
        with open(gif_file, 'rb') as f:
            display(Image(f.read(), format='gif'))

    return ani

def animate_slice_axis(data, axis, axis_values, ax=None, interval=500, save_gif=False, filename='animation', title=True, title_fmt='.2f', plot_slice_kwargs={}, ax_kwargs={}):
    """
    Animate 2D slices of a 3D dataset along a given axis.

    Creates a matplotlib animation by sweeping through 2D slices of a 3D 
    dataset along the specified axis. Optionally displays the animation 
    inline (e.g., in Jupyter) and/or saves it as a GIF.

    Parameters
    ----------
    data : nexusformat.nexus.NXdata
        The 3D dataset to visualize.
    axis : int
        The axis along which to animate (must be 0, 1, or 2).
    axis_values : iterable
        The values along the animation axis to use as animation frames.
    ax : matplotlib.axes.Axes, optional
        The axes object to plot on. If None, a new figure and axes will be created.
    interval : int, optional
        Delay between frames in milliseconds. Default is 500.
    save_gif : bool, optional
        If True, saves the animation as a .gif file. Default is False.
    filename : str, optional
        Filename (without extension) to use for the saved .gif. Default is 'animation'.
    title : bool, optional
        If True, displays the axis value as a title for each frame. Default is True.
    title_fmt : str, optional
        Format string for axis value in the title (e.g., '.2f' for 2 decimals). Default is '.2f'.
    plot_slice_kwargs : dict, optional
        Additional keyword arguments passed to `plot_slice`.
    ax_kwargs : dict, optional
        Keyword arguments passed to `ax.set` to update axis settings.

    Returns
    -------
    ani : matplotlib.animation.FuncAnimation
        The animation object.
    """
    if ax is None:
        fig,ax = plt.subplots() # Generate a new figure and axis
    else:
        fig = ax.figure  # Get the figure from the provided axis

    if axis not in [0, 1, 2]:
        raise ValueError("axis must be either 0, 1, or 2.")

    if plot_slice_kwargs.get('cbar', False):
        warnings.warn("Colorbar is not supported in animation and will be ignored.", UserWarning)
        plot_slice_kwargs['cbar'] = False
    elif 'cbar' not in plot_slice_kwargs.keys():
        plot_slice_kwargs['cbar'] = False


    def update(parameter):
        ax.clear()

        # Construct slicing object for the selected axis
        slice_obj = [slice(None)] * 3
        slice_obj[axis] = parameter

        # Plot the 2D slice
        plot_slice(data[tuple(slice_obj)], ax=ax, **plot_slice_kwargs)
        ax.set(**ax_kwargs)

        if title:
            axis_label = data.nxaxes[axis].nxname
            ax.set(title=f'{axis_label}={parameter:{title_fmt}}')


    ani = animation.FuncAnimation(fig, update, frames=axis_values, interval=interval, repeat=False)

    display(HTML(ani.to_jshtml()))

    if save_gif:
        gif_file = f'{filename}.gif'
        writergif = animation.PillowWriter(fps=1000/interval)
        ani.save(gif_file, writer=writergif)
        display(HTML(ani.to_jshtml()))
        with open(gif_file, 'rb') as file:
            display(Image(file.read(), format='gif'))

    return ani


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
            print("Linecut axis: " + str(self.data.nxaxes[self.axis].nxname))
            print("Integrated axes: " + str([self.data.nxaxes[axis].nxname
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
        integrated_data = np.sum(self.integration_volume.nxsignal.nxdata,
                                 axis=self.integrated_axes)

        # Create an NXdata object for the linecut data
        self.linecut = NXdata(NXfield(integrated_data, name=self.integration_volume.nxsignal.nxname),
                              self.integration_volume[self.integration_volume.nxaxes[self.axis].nxname])
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
                        X=data.nxaxes[0],
                        Y=data.nxaxes[1],
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
                        X=data.nxaxes[0],
                        Y=data.nxaxes[2],
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
                        X=data.nxaxes[1],
                        Y=data.nxaxes[2],
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
                        X=data.nxaxes[0],
                        Y=data.nxaxes[1],
                        ax=axes[0],
                        **kwargs)
        axes[0].set_aspect(len(data.nxaxes[0].nxdata) / len(data.nxaxes[1].nxdata))

        # Plot cross section 2
        slice_obj = [slice(None)] * data.ndim
        slice_obj[1] = center[1]
        p3 = plot_slice(data[slice_obj],
                        X=data.nxaxes[0],
                        Y=data.nxaxes[2],
                        ax=axes[1],
                        **kwargs)
        axes[1].set_aspect(len(data.nxaxes[0].nxdata) / len(data.nxaxes[2].nxdata))

        # Plot cross-section 3
        slice_obj = [slice(None)] * data.ndim
        slice_obj[0] = center[0]
        p2 = plot_slice(data[slice_obj],
                        X=data.nxaxes[1],
                        Y=data.nxaxes[2],
                        ax=axes[2],
                        **kwargs)
        axes[2].set_aspect(len(data.nxaxes[1].nxdata) / len(data.nxaxes[2].nxdata))

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


def rotate_data(data, lattice_angle, rotation_angle, rotation_axis=None, rotation_order=None, aspect=None, aspect_order=None, printout=False):
    """
    Rotates slices of data around the normal axis.

    Parameters
    ----------
    data : :class:`nexusformat.nexus.NXdata`
        Input data.
    lattice_angle : float
        Angle between the two in-plane lattice axes in degrees.
    rotation_angle : float
        Angle of rotation in degrees.
    rotation_axis : int, optional
        Axis of rotation (0, 1, or 2). Only necessary when data is three-dimensional.
    rotation_order : int, optional
        Interpolation order passed to :func:`scipy.ndimage.rotate`. 
        Determines the spline interpolation used during rotation.
        Valid values are integers from 0 (nearest-neighbor) to 5 (higher-order splines).
        Defaults to 0 if not specified.
    aspect : float, optional
        True aspect ratio between the lengths of the basis vectors of the two principal axes of the plane to be rotated. Calculated as aspect = (length of y) / (length of x). Defaults to 1.
    aspect_order : int, optional
        Interpolation order passed to :func:`scipy.ndimage.zoom` when applying and undoing
        the coordinate aspect ratio correction. Determines the spline interpolation used
        during resampling. Valid values are integers from 0 (nearest-neighbor) to 5.
        Defaults to 0 if not specified.
    printout : bool, optional
        Enables printout of rotation progress for three-dimensional data. If set to True,
        information about each rotation slice will be printed to the console, indicating
        the axis being rotated and the corresponding coordinate value. Defaults to False.


    Returns
    -------
    rotated_data : :class:`nexusformat.nexus.NXdata`
        Rotated data as an NXdata object.
    """
    if aspect is None:
        aspect = 1
    if aspect_order is None:
        aspect_order = 0
    if rotation_order is None:
        rotation_order = 0

    if data.ndim == 3 and rotation_axis is None:
        raise ValueError('rotation_axis must be specified for three-dimensional datasets.')
    
    if not((data.ndim == 2) or (data.ndim == 3)):
        raise ValueError('Data must be 2 or 3 dimensional.')
    
    # Define output array
    output_array = np.zeros(data.nxsignal.shape)

    # Iterate over all layers perpendicular to the rotation axis
    if data.ndim == 3:
        num_slices = len(data.nxaxes[rotation_axis])
    elif data.ndim == 2:
        num_slices = 1
    
    for i in range(num_slices):
        
        if data.ndim == 3:
            # Print progress
            if printout:
                print(f'\rRotating {data.nxaxes[rotation_axis].nxname}'
                f'={data.nxaxes[rotation_axis][i]}...                      ',
                end='', flush=True)
            index = [slice(None)] * 3
            index[rotation_axis] = i
            sliced_data = data[tuple(index)]
            
        elif data.ndim == 2:
            sliced_data = data

        # Add padding to avoid data cutoff during rotation
        p = Padder(sliced_data)
        padding = tuple(len(axis) for axis in sliced_data.nxaxes)
        counts = p.pad(padding)
        counts = p.padded.nxsignal

        # Skew data to match lattice angle
        t = ShearTransformer(lattice_angle)
        counts = t.apply(counts)

        # Apply coordinate aspect ratio correction
        counts = zoom(counts, zoom=(1, aspect), order=aspect_order)

        # Perform rotation
        counts = rotate(counts, rotation_angle, reshape=False, order=rotation_order)

        # Undo aspect ratio correction
        counts = zoom(counts, zoom=(1, 1 / aspect), order=aspect_order)

        # Undo skew transformation
        counts = t.invert(counts)
        
        # Remove padding
        counts = p.unpad(counts)

        # Write slice
        if data.ndim == 3:
            index = [slice(None)] * 3
            index[rotation_axis] = i
            output_array[tuple(index)] = counts
        elif data.ndim == 2:
            output_array = counts

    print('\nRotation completed.')

    return NXdata(NXfield(output_array, name=p.padded.nxsignal.nxname),
                  ([axis for axis in data.nxaxes]))



def rotate_data_2D(data, lattice_angle, rotation_angle):
    """
    DEPRECATED: Use `rotate_data` instead.

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
    warnings.warn(
        "rotate_data_2D is deprecated and will be removed in a future release. "
        "Use rotate_data instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Call the new general function
    return rotate_data(data, lattice_angle=lattice_angle, rotation_angle=rotation_angle)


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

        self.steps = tuple((axis.nxdata[1] - axis.nxdata[0])
                           for axis in data.nxaxes)

        # Absolute value of the maximum value; assumes the domain of the input
        # is symmetric (eg, -H_min = H_max)
        self.maxes = tuple(axis.nxdata.max() for axis in data.nxaxes)

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

        padded_shape = tuple(data.nxsignal.nxdata.shape[i]
                             + self.padding[i] * 2 for i in range(data.ndim))

        # Create padded dataset
        padded = np.zeros(padded_shape)

        slice_obj = [slice(None)] * data.ndim
        for i, _ in enumerate(slice_obj):
            slice_obj[i] = slice(self.padding[i], -self.padding[i], None)
        slice_obj = tuple(slice_obj)
        padded[slice_obj] = data.nxsignal.nxdata

        padmaxes = tuple(self.maxes[i] + self.padding[i] * self.steps[i]
                         for i in range(data.ndim))

        padded = NXdata(NXfield(padded, name=data.nxsignal.nxname),
                        tuple(NXfield(np.linspace(-padmaxes[i], padmaxes[i], padded_shape[i]),
                                      name=data.nxaxes[i].nxname)
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
