"""
Reduces scattering data into 2D and 1D datasets.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from matplotlib.markers import MarkerStyle
from matplotlib.ticker import MultipleLocator
import matplotlib.colors as colors
from IPython.display import display, Markdown
from nexusformat.nexus import NXfield, NXdata

__all__=['plot_slice','cut_data']

def plot_slice(X, Y, Z, vmin=None, vmax=None, skew_angle=90, ax=None, xlim=None, ylim=None,
    xticks=None, yticks=None, cbar=True, logscale=False, symlogscale=False, cmap='viridis', linthresh = 1, title=None):
    
    '''
    Parameters
    ----------

    X : array_like
        The x-coordinates of the data.

    Y : array_like
        The y-coordinates of the data.

    Z : array_like
        The 2D dataset to plot.

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

    title : str, optional
        A string containing the title for the plot. Default `None`.

    Returns
    -------
    p : :class:`matplotlib.collections.QuadMesh`

        A :class:`matplotlib.collections.QuadMesh` object, to mimick behavior of :class:`matplotlib.pyplot.pcolormesh`.


    '''
    
    Z = Z.transpose()

    # Display Markdown heading
    if title is None:
        pass
    elif title == "None":
        display(Markdown('### Figure'))
    else:
        display(Markdown('### Figure - '+title))
    
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
        vmin=Z.min()
    if vmax is None:
        vmax=Z.max()

    # Plot data
    if symlogscale:
        p = ax.pcolormesh(X, Y, Z,
                          # vmin=vmin, vmax=vmax,
                          shading='auto',
                          norm=colors.SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax),
                          cmap=cmap
                         )
    elif logscale:
        p = ax.pcolormesh(X, Y, Z,
                          # vmin=vmin, vmax=vmax,
                          shading='auto',
                          norm=colors.LogNorm(vmin=vmin, vmax=vmax),
                          cmap=cmap
                         )
    else:
        p = ax.pcolormesh(X, Y, Z,
                          vmin=vmin, vmax=vmax,
                          shading='auto',
                          cmap=cmap
                         )
    
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
        fig.colorbar(p)

    # Return the quadmesh object
    return p

def cut_data(data, center, window, axis=None):
    '''
    Reduces data to 1D linecut with integration extents specified by window about a central coordinate.
    '''
    start = np.subtract(center,window)
    stop = np.add(center,window)
    slice_obj = tuple(slice(s,e) for s,e in zip(start,stop))

    data_cut = data[slice_obj]

    if axis is None:
        axis = window.index(max(window))

    integrated_axes = tuple(i for i in range(g.ndim) if i != axis)
    integrated_data = np.sum(data_cut[data_cut.signal].nxdata, axis=integrated_axes)

    return NXdata(NXfield(integrated_data, name=data_cut.signal), data_cut[data_cut.axes[axis]])
