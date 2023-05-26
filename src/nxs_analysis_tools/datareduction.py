"""
Reduces scattering data into 2D and 1D datasets.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from matplotlib.markers import MarkerStyle
from matplotlib.ticker import MultipleLocator
import matplotlib.colors as colors
import matplotlib.patches as patches
from IPython.display import display, Markdown
from nexusformat.nexus import NXfield, NXdata, nxload

__all__=['load_data','plot_slice','Scissors']

def load_data(path):
    g = nxload(path)
    print(g.entry.data.tree)
    return g.entry.data

def plot_slice(data, X=None, Y=None, transpose=False, vmin=None, vmax=None, skew_angle=90, ax=None, xlim=None, ylim=None,
    xticks=None, yticks=None, cbar=True, logscale=False, symlogscale=False, cmap='viridis', linthresh = 1, title=None, mdheading=None,
    cbartitle=None):

    '''
    Parameters
    ----------
    data : :class:`nexusformat.nexus.NXdata` object
        The NXdata object containing the dataset to plot.

    X : array_like, optional
        The X axis values. Default is first axis of `data`.

    Y : array_like, optional
        The y axis values. Default is second axis of `data`.

    vmin : float, optional
        The minimum value to plot in the dataset. If not provided, the minimum of the dataset will be used.

    vmax : float, optional
        The maximum value to plot in the dataset. If not provided, the maximum of the dataset will be used.

    skew_angle : float, optional
        The angle to shear the plot in degrees. Defaults to 90 degrees (no skewing).

    ax : matplotlib.axes.Axes, optional
        An optional axis object to plot the heatmap onto.

    xlim : tuple, optional
        The limits of the x-axis. If not provided, the limits will be automatically set.

    ylim : tuple, optional
        The limits of the y-axis. If not provided, the limits will be automatically set.

    xticks : float, optional
        The major tick interval for the x-axis. If not provided, the function will use a default minor tick interval of 1.

    yticks : float, optional
        The major tick interval for the y-axis. If not provided, the function will use a default minor tick interval of 1.

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

        A :class:`matplotlib.collections.QuadMesh` object, to mimick behavior of :class:`matplotlib.pyplot.pcolormesh`.

    '''

    if X is None:
        X = data[data.axes[0]].nxdata
    if Y is None:
        Y = data[data.axes[1]].nxdata
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
    norm = None
    if symlogscale:
        norm = colors.SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax)
    elif logscale:
        norm = colors.LogNorm(vmin=vmin, vmax=vmax)

    # Plot data
    p = ax.pcolormesh(X, Y, data_arr, vmin=vmin, vmax=vmax, shading='auto', norm=norm, cmap=cmap)

    
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
        xlabel=data.axes[0],
        ylabel=data.axes[1],
    )

    if title is not None:
        ax.set_title(title)

    # Return the quadmesh object
    return p

class Scissors():
    '''
    Scissors class
    '''
    def __init__(self):
        pass
    
    def set_data(self, data):
        self.data = data
    
    def get_data(self):
        return self.data
    
    def set_center(self, center):
        self.center = center
    
    def set_window(self, window):
        self.window = window
        
    def get_window(self):
        return self.window
    
    # def set(self, **kwargs):
    #     for attr_name, value in kwargs.items():
    #         setattr(self, attr_name, value)
    
    def cut_data(self, axis=None):
        '''
        Reduces data to 1D linecut with integration extents specified by window about a central coordinate.

        Parameters:
        -----------
        data : ndarray
            Input data array.
        center : tuple or list
            Central coordinate around which to perform the linecut.
        window : tuple or list
            Extents of the window for integration along each axis.
        axis : int, optional
            Axis along which to perform the integration. If not specified, the axis with the largest extent in the window will be used.

        Returns:
        --------
        integrated_data : ndarray
            1D linecut data after integration.

        '''
        data = self.data
        center = self.center
        window = self.window
        if axis is None:
            self.axis = window.index(max(window))
        else:
            self.axis = axis
        axis = self.axis

        center = tuple(float(c) for c in center)

        start = np.subtract(center,window)
        stop = np.add(center,window)
        slice_obj = tuple(slice(s,e) for s,e in zip(start,stop))

        self.data_cut = data[slice_obj]

        self.integrated_axes = tuple(i for i in range(data.ndim) if i != axis)
        integrated_data = np.sum(self.data_cut[self.data_cut.signal].nxdata, axis=self.integrated_axes)
        
        self.linecut = NXdata(NXfield(integrated_data, name=self.data_cut.signal), self.data_cut[self.data_cut.axes[axis]])
        
        return self.linecut
    
    def show_integration_window(self, label=None):
        '''
        Plots integration window highlighted on 2D heatmap full dataset.
        '''
        data = self.data
        axis = self.axis
        center = self.center
        window = self.window
        integrated_axes = self.integrated_axes
        
        # Plot cross section
        window_plane_slice_obj = [slice(None)]*data.ndim
        window_plane_slice_obj[axis] = center[axis]
        p = plot_slice(data[data.axes[integrated_axes[0]]], data[data.axes[integrated_axes[1]]],
            data[window_plane_slice_obj][data.signal], vmin=1, logscale=True)
        ax = plt.gca()
        rect_diffuse = patches.Rectangle(
            (center[integrated_axes[0]]-window[integrated_axes[0]],
            center[integrated_axes[1]]-window[integrated_axes[1]]),
            window[integrated_axes[0]], window[integrated_axes[1]],
            linewidth=1, edgecolor='r', facecolor='none', transform=p.get_transform(), label=label,
            )
        ax.add_patch(rect_diffuse)
        plt.show()

        p = plot_slice(data[data.axes[integrated_axes[0]]], data[data.axes[axis[1]]],
            data[window_plane_slice_obj][data.signal], vmin=1, logscale=True)

        p = plot_slice(data[data.axes[integrated_axes[1]]], data[data.axes[axis[1]]],
            data[window_plane_slice_obj][data.signal], vmin=1, logscale=True)
        
        
        self.window_plane_slice_obj = window_plane_slice_obj
        
    def plot_window(self):
        '''
        Plots 2D heatmap of integration window data on its own.
        '''
        data = self.data_cut
        p = plot_slice(
            data[data.axes[self.integrated_axes[0]]],
            data[data.axes[self.integrated_axes[1]]],
            data[self.window_plane_slice_obj][data.signal],
            vmin=1, logscale=True,
            )
        plt.show()