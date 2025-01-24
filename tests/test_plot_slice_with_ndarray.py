from nxs_analysis_tools import *
from nxs_analysis_tools.datareduction import load_transform

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


# def plot_slice(data, X=None, Y=None, transpose=False, vmin=None, vmax=None,
#                skew_angle=90, ax=None, xlim=None, ylim=None,
#                xticks=None, yticks=None, cbar=True, logscale=False,
#                symlogscale=False, cmap='viridis', linthresh=1,
#                title=None, mdheading=None, cbartitle=None,
#                **kwargs):
#     """
#     Plot a 2D slice of the provided dataset, with optional transformations
#     and customizations.
#
#     Parameters
#     ----------
#     data : :class:`nexusformat.nexus.NXdata` or ndarray
#         The dataset to plot. Can be an `NXdata` object or a `numpy` array.
#
#     X : NXfield, optional
#         The X axis values. If None, a default range from 0 to the number of
#          columns in `data` is used.
#
#     Y : NXfield, optional
#         The Y axis values. If None, a default range from 0 to the number of
#          rows in `data` is used.
#
#     transpose : bool, optional
#         If True, transpose the dataset and its axes before plotting.
#         Default is False.
#
#     vmin : float, optional
#         The minimum value for the color scale. If not provided, the minimum
#          value of the dataset is used.
#
#     vmax : float, optional
#         The maximum value for the color scale. If not provided, the maximum
#          value of the dataset is used.
#
#     skew_angle : float, optional
#         The angle in degrees to shear the plot. Default is 90 degrees (no skew).
#
#     ax : matplotlib.axes.Axes, optional
#         The `matplotlib` axis to plot on. If None, a new figure and axis will
#          be created.
#
#     xlim : tuple, optional
#         The limits for the x-axis. If None, the limits are set automatically
#          based on the data.
#
#     ylim : tuple, optional
#         The limits for the y-axis. If None, the limits are set automatically
#          based on the data.
#
#     xticks : float or list of float, optional
#         The major tick interval or specific tick locations for the x-axis.
#          Default is to use a minor tick interval of 1.
#
#     yticks : float or list of float, optional
#         The major tick interval or specific tick locations for the y-axis.
#         Default is to use a minor tick interval of 1.
#
#     cbar : bool, optional
#         Whether to include a colorbar. Default is True.
#
#     logscale : bool, optional
#         Whether to use a logarithmic color scale. Default is False.
#
#     symlogscale : bool, optional
#         Whether to use a symmetrical logarithmic color scale. Default is False.
#
#     cmap : str or Colormap, optional
#         The colormap to use for the plot. Default is 'viridis'.
#
#     linthresh : float, optional
#         The linear threshold for symmetrical logarithmic scaling. Default is 1.
#
#     title : str, optional
#         The title for the plot. If None, no title is set.
#
#     mdheading : str, optional
#         A Markdown heading to display above the plot. If 'None' or not provided,
#          no heading is displayed.
#
#     cbartitle : str, optional
#         The title for the colorbar. If None, the colorbar label will be set to
#          the name of the signal.
#
#     **kwargs
#         Additional keyword arguments passed to `pcolormesh`.
#
#     Returns
#     -------
#     p : :class:`matplotlib.collections.QuadMesh`
#         The `matplotlib` QuadMesh object representing the plotted data.
#     """
#     if isinstance(data, np.ndarray):
#         if X is None:
#             X = NXfield(np.linspace(0, data.shape[1], data.shape[1]), name='x')
#         if Y is None:
#             Y = NXfield(np.linspace(0, data.shape[0], data.shape[0]), name='y')
#         if transpose:
#             X, Y = Y, X
#             data = data.transpose()
#         data = NXdata(NXfield(data, name='value'), (X, Y))
#         data_arr = data
#     elif isinstance(data, (NXdata, NXfield)):
#         if X is None:
#             X = data[data.axes[0]]
#         if Y is None:
#             Y = data[data.axes[1]]
#         if transpose:
#             X, Y = Y, X
#             data = data.transpose()
#         data_arr = data[data.signal].nxdata.transpose()
#     else:
#         raise TypeError(f"Unexpected data type: {type(data)}. "
#                         f"Supported types are np.ndarray and NXdata.")
#
#     # Display Markdown heading
#     if mdheading is None:
#         pass
#     elif mdheading == "None":
#         display(Markdown('### Figure'))
#     else:
#         display(Markdown('### Figure - ' + mdheading))
#
#     # Inherit axes if user provides some
#     if ax is not None:
#         fig = ax.get_figure()
#     # Otherwise set up some default axes
#     else:
#         fig = plt.figure()
#         ax = fig.add_axes([0, 0, 1, 1])
#
#     # If limits not provided, use extrema
#     if vmin is None:
#         vmin = data_arr.min()
#     if vmax is None:
#         vmax = data_arr.max()
#
#     # Set norm (linear scale, logscale, or symlogscale)
#     norm = colors.Normalize(vmin=vmin, vmax=vmax)  # Default: linear scale
#
#     if symlogscale:
#         norm = colors.SymLogNorm(linthresh=linthresh, vmin=-1 * vmax, vmax=vmax)
#     elif logscale:
#         norm = colors.LogNorm(vmin=vmin, vmax=vmax)
#
#     # Plot data
#     p = ax.pcolormesh(X.nxdata, Y.nxdata, data_arr, shading='auto', norm=norm, cmap=cmap, **kwargs)
#
#     ## Transform data to new coordinate system if necessary
#     # Correct skew angle
#     skew_angle_adj = 90 - skew_angle
#     # Create blank 2D affine transformation
#     t = Affine2D()
#     # Scale y-axis to preserve norm while shearing
#     t += Affine2D().scale(1, np.cos(skew_angle_adj * np.pi / 180))
#     # Shear along x-axis
#     t += Affine2D().skew_deg(skew_angle_adj, 0)
#     # Return to original y-axis scaling
#     t += Affine2D().scale(1, np.cos(skew_angle_adj * np.pi / 180)).inverted()
#     ## Correct for x-displacement after shearing
#     # If ylims provided, use those
#     if ylim is not None:
#         # Set ylims
#         ax.set(ylim=ylim)
#         ymin, ymax = ylim
#     # Else, use current ylims
#     else:
#         ymin, ymax = ax.get_ylim()
#     # Use ylims to calculate translation (necessary to display axes in correct position)
#     p.set_transform(t
#                     + Affine2D().translate(-ymin * np.sin(skew_angle_adj * np.pi / 180), 0)
#                     + ax.transData)
#
#     # Set x limits
#     if xlim is not None:
#         xmin, xmax = xlim
#     else:
#         xmin, xmax = ax.get_xlim()
#     if skew_angle <= 90:
#         ax.set(xlim=(xmin, xmax + (ymax - ymin) / np.tan((90 - skew_angle_adj) * np.pi / 180)))
#     else:
#         ax.set(xlim=(xmin - (ymax - ymin) / np.tan((skew_angle_adj - 90) * np.pi / 180), xmax))
#
#     # Correct aspect ratio for the x/y axes after transformation
#     ax.set(aspect=np.cos(skew_angle_adj * np.pi / 180))
#
#     # Add tick marks all around
#     ax.tick_params(direction='in', top=True, right=True, which='both')
#
#     # Set tick locations
#     if xticks is None:
#         # Add default minor ticks
#         ax.xaxis.set_minor_locator(MultipleLocator(1))
#     else:
#         # Otherwise use user provided values
#         ax.xaxis.set_major_locator(MultipleLocator(xticks))
#         ax.xaxis.set_minor_locator(MultipleLocator(1))
#     if yticks is None:
#         # Add default minor ticks
#         ax.yaxis.set_minor_locator(MultipleLocator(1))
#     else:
#         # Otherwise use user provided values
#         ax.yaxis.set_major_locator(MultipleLocator(yticks))
#         ax.yaxis.set_minor_locator(MultipleLocator(1))
#
#     # Apply transform to tick marks
#     for i in range(0, len(ax.xaxis.get_ticklines())):
#         # Tick marker
#         m = MarkerStyle(3)
#         line = ax.xaxis.get_majorticklines()[i]
#         if i % 2:
#             # Top ticks (translation here makes their direction="in")
#             m._transform.set(Affine2D().translate(0, -1) + Affine2D().skew_deg(skew_angle_adj, 0))
#             # This first method shifts the top ticks horizontally to match the skew angle.
#             # This does not look good in all cases.
#             # line.set_transform(Affine2D().translate((ymax-ymin)*np.sin(skew_angle*np.pi/180),0) +
#             #     line.get_transform())
#             # This second method skews the tick marks in place and
#             # can sometimes lead to them being misaligned.
#             line.set_transform(line.get_transform())  # This does nothing
#         else:
#             # Bottom ticks
#             m._transform.set(Affine2D().skew_deg(skew_angle_adj, 0))
#
#         line.set_marker(m)
#
#     for i in range(0, len(ax.xaxis.get_minorticklines())):
#         m = MarkerStyle(2)
#         line = ax.xaxis.get_minorticklines()[i]
#         if i % 2:
#             m._transform.set(Affine2D().translate(0, -1) + Affine2D().skew_deg(skew_angle_adj, 0))
#         else:
#             m._transform.set(Affine2D().skew_deg(skew_angle_adj, 0))
#
#         line.set_marker(m)
#
#     if cbar:
#         colorbar = fig.colorbar(p)
#         if cbartitle is None:
#             colorbar.set_label(data.signal)
#
#     ax.set(
#         xlabel=X.nxname,
#         ylabel=Y.nxname,
#     )
#
#     if title is not None:
#         ax.set_title(title)
#
#     # Return the quadmesh object
#     return p


data = load_transform(r'K:\wilson-3947-a\nxrefine\LaCd3P3\MLA1\LaCd3P3_300.nxs')

# print(data.tree)
fig = plt.figure(figsize=(5,3))
ax = fig.add_axes([0,0,1,1])
plot_slice(data[-1.0:1.0, -0.5:0.5, 0.0].counts.nxdata, vmin=0, vmax=100, cbar=False, ax=ax)
fig.tight_layout()
plt.show()