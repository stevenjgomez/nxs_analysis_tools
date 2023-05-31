"""
Reduces scattering data into 2D and 1D datasets.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from matplotlib.markers import MarkerStyle
from matplotlib.ticker import MultipleLocator
from matplotlib import colors
from matplotlib import patches
from IPython.display import display, Markdown
from nexusformat.nexus import NXfield, NXdata, nxload

__all__=['load_data','plot_slice','Scissors']

def load_data(path):
    '''
    Load data from a specified path.

    Parameters
    ----------
    path : str
        The path to the data file.

    Returns
    -------
    data : nxdata object
        The loaded data stored in a nxdata object.

    '''
    g = nxload(path)
    try:
        print(g.entry.data.tree)
    except NeXusError:
        pass

    return g.entry.data


def plot_slice(data, X=None, Y=None, transpose=False, vmin=None, vmax=None, skew_angle=90,
    ax=None, xlim=None, ylim=None, xticks=None, yticks=None, cbar=True, logscale=False,
    symlogscale=False, cmap='viridis', linthresh = 1, title=None, mdheading=None, cbartitle=None):

    '''
    Parameters
    ----------
    data : :class:`nexusformat.nexus.NXdata` object
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

    '''

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
        ax=ax
        fig=ax.get_figure()
    # Otherwise set up some default axes
    else:
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])

    # If limits not provided, use extrema
    if vmin is None:
        vmin=data_arr.min()
    if vmax is None:
        vmax=data_arr.max()

    # Set norm (linear scale, logscale, or symlogscale)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)  # Default: linear scale

    if symlogscale:
        norm = colors.SymLogNorm(linthresh=linthresh, vmin=-1 * vmax, vmax=vmax)
    elif logscale:
        norm = colors.LogNorm(vmin=vmin, vmax=vmax)


    # Plot data
    p = ax.pcolormesh(X.nxdata, Y.nxdata, data_arr, shading='auto', norm=norm, cmap=cmap)

    ## Transform data to new coordinate system if necessary
    # Correct skew angle
    skew_angle = 90-skew_angle
    # Create blank 2D affine transformation
    t = Affine2D()
    # Scale y-axis to preserve norm while shearing
    t += Affine2D().scale(1,np.cos(skew_angle*np.pi/180))
    # Shear along x-axis
    t += Affine2D().skew_deg(skew_angle,0)
    # Return to original y-axis scaling
    t += Affine2D().scale(1,np.cos(skew_angle*np.pi/180)).inverted()
    ## Correct for x-displacement after shearing
    # If ylims provided, use those
    if ylim is not None:
        # Set ylims
        ax.set(ylim=ylim)
        ymin,ymax = ylim
    # Else, use current ylims
    else:
        ymin,ymax = ax.get_ylim()
    # Use ylims to calculate translation (necessary to display axes in correct position)
    p.set_transform(t + Affine2D().translate(-ymin*np.sin(skew_angle*np.pi/180),0) + ax.transData)

    # Set x limits
    if xlim is not None:
        xmin,xmax = xlim
    else:
        xmin,xmax = ax.get_xlim()
    ax.set(xlim=(xmin,xmax+(ymax-ymin)/np.tan((90-skew_angle)*np.pi/180)))

    # Correct aspect ratio for the x/y axes after transformation
    ax.set(aspect=np.cos(skew_angle*np.pi/180))

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

    ## Apply transform to tick marks
    for i in range(0,len(ax.xaxis.get_ticklines())):
        # Tick marker
        m = MarkerStyle(3)
        line =  ax.xaxis.get_majorticklines()[i]
        if i%2:
            # Top ticks (translation here makes their direction="in")
            m._transform.set(Affine2D().translate(0,-1) + Affine2D().skew_deg(skew_angle,0))
            # This first method shifts the top ticks horizontally to match the skew angle.
            # This does not look good in all cases.
            # line.set_transform(Affine2D().translate((ymax-ymin)*np.sin(skew_angle*np.pi/180),0) +
            #     line.get_transform())
            # This second method skews the tick marks in place and
            # can sometimes lead to them being misaligned.
            line.set_transform(line.get_transform()) # This does nothing
        else:
            # Bottom ticks
            m._transform.set(Affine2D().skew_deg(skew_angle,0))

        line.set_marker(m)

    for i in range(0,len(ax.xaxis.get_minorticklines())):
        m = MarkerStyle(2)
        line =  ax.xaxis.get_minorticklines()[i]
        if i%2:
            m._transform.set(Affine2D().translate(0,-1) + Affine2D().skew_deg(skew_angle,0))
        else:
            m._transform.set(Affine2D().skew_deg(skew_angle,0))

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

class Scissors():
    '''
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
    '''

    def __init__(self, data=None, center=None, window=None, axis=None):
        '''
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
        '''

        self.data = data
        self.center = center
        self.window = window
        self.axis = axis

        self.integration_volume = None
        self.integrated_axes = None
        self.linecut = None
        self.integration_window = None

    def set_data(self, data):
        '''
        Set the input NXdata.

        Parameters
        ----------
        data : :class:`nexusformat.nexus.NXdata`
            Input data array.
        '''
        self.data = data

    def get_data(self):
        '''
        Get the input data array.

        Returns
        -------
        ndarray or None
            Input data array.
        '''
        return self.data

    def set_center(self, center):
        '''
        Set the central coordinate for the linecut.

        Parameters
        ----------
        center : tuple
            Central coordinate around which to perform the linecut.
        '''
        self.center = center

    def set_window(self, window):
        '''
        Set the extents of the integration window.

        Parameters
        ----------
        window : tuple
            Extents of the window for integration along each axis.
        '''
        self.window = window

        # Determine the axis for integration
        self.axis = window.index(max(window))
        print("Linecut axis: "+str(self.data.axes[self.axis]))

        # Determine the integrated axes (axes other than the integration axis)
        self.integrated_axes = tuple(i for i in range(self.data.ndim) if i != self.axis)
        print("Integrated axes: "+str([self.data.axes[axis] for axis in self.integrated_axes]))

    def get_window(self):
        '''
        Get the extents of the integration window.

        Returns
        -------
        tuple or None
            Extents of the integration window.
        '''
        return self.window

    def cut_data(self, center=None, window=None, axis=None):
        '''
        Reduces data to a 1D linecut with integration extents specified by the window about a central
        coordinate.

        Parameters:
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

        Returns:
        --------
        integrated_data : :class:`nexusformat.nexus.NXdata`
            1D linecut data after integration.
        '''

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

    def show_integration_window(self, label=None, **kwargs):
        '''
        Plots integration window highlighted on 2D heatmap full dataset.
        '''
        data = self.data
        axis = self.axis
        center = self.center
        window = self.window
        integrated_axes = self.integrated_axes

        # Plot cross section 1
        slice_obj = [slice(None)]*data.ndim
        slice_obj[axis] = center[axis]

        p1 = plot_slice(data[slice_obj],
                       X=data[data.axes[integrated_axes[0]]],
                       Y=data[data.axes[integrated_axes[1]]],
                       **kwargs)
        ax = plt.gca()
        rect_diffuse = patches.Rectangle(
            (center[integrated_axes[0]]-window[integrated_axes[0]],
            center[integrated_axes[1]]-window[integrated_axes[1]]),
            2*window[integrated_axes[0]], 2*window[integrated_axes[1]],
            linewidth=1, edgecolor='r', facecolor='none', transform=p1.get_transform(), label=label,
            )
        ax.add_patch(rect_diffuse)
        plt.show()

        # Plot cross section 2
        slice_obj = [slice(None)]*data.ndim
        slice_obj[integrated_axes[1]] = center[integrated_axes[1]]

        p2 = plot_slice(data[slice_obj],
                       X=data[data.axes[integrated_axes[0]]],
                       Y=data[data.axes[axis]],
                       **kwargs)
        ax = plt.gca()
        rect_diffuse = patches.Rectangle(
            (center[integrated_axes[0]]-window[integrated_axes[0]],
            center[axis]-window[axis]),
            2*window[integrated_axes[0]], 2*window[axis],
            linewidth=1, edgecolor='r', facecolor='none', transform=p2.get_transform(), label=label,
            )
        ax.add_patch(rect_diffuse)
        plt.show()

        # Plot cross section 3
        slice_obj = [slice(None)]*data.ndim
        slice_obj[integrated_axes[0]] = center[integrated_axes[0]]

        p3 = plot_slice(data[slice_obj],
                       X=data[data.axes[integrated_axes[1]]],
                       Y=data[data.axes[axis]],
                       **kwargs)
        ax = plt.gca()
        rect_diffuse = patches.Rectangle(
            (center[integrated_axes[1]]-window[integrated_axes[1]],
            center[axis]-window[axis]),
           2*window[integrated_axes[1]], 2*window[axis],
            linewidth=1, edgecolor='r', facecolor='none', transform=p3.get_transform(), label=label,
            )
        ax.add_patch(rect_diffuse)
        plt.show()

        return (p1,p2,p3)

    # def plot_window(self):
    #     '''
    #     Plots 2D heatmap of integration window data on its own.
    #     '''
    #     data = self.integration_volume

    #     # TODO: Adjust code to plot 3 different cross sections, create slice_obj
    #     p = plot_slice(
    #         data[slice_obj],
    #         data[data.axes[self.integrated_axes[0]]],
    #         data[data.axes[self.integrated_axes[1]]],
    #         vmin=1, logscale=True,
    #         )
    #     plt.show()

    #     return
