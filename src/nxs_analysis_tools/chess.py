'''
This module provides classes and functions for analyzing scattering datasets collected at CHESS
(ID4B) with temperature dependence. It includes functions for loading data, cutting data, and
plotting linecuts.
'''
import os

import matplotlib.pyplot as plt
import matplotlib as mpl

from nxs_analysis_tools import load_data, Scissors


class TempDependence():
    '''
    Class for analyzing scattering datasets collected at CHESS (ID4B) with temperature dependence.
    '''

    def __init__(self):
        '''
        Initialize TempDependence class.
        '''
        self.datasets={}
        self.folder=None
        self.temperatures=None
        self.scissors= {}
        self.linecuts={}

    def get_folder(self):
        '''
        Get the folder path where the datasets are located.

        Returns
        -------
            str:
                The folder path.
        '''
        return self.folder

    def clear_datasets(self):
        '''
        Clear the datasets stored in the TempDependence instance.
        '''
        self.datasets={}

    def load_datasets(self, folder, file_ending='hkli.nxs', temperatures_list=None):
        '''
        Load scattering datasets from the specified folder.

        Parameters
        ----------
        folder : str
            The path to the folder where the datasets are located.
        file_ending : str, optional
            The file extension of the datasets to be loaded. The default is 'hkli.nxs'.
        temperatures_list : list of int or None, optional
            The list of specific temperatures to load. If None, all available temperatures are
            loaded. The default is None.
        '''
        self.folder=os.path.normpath(folder)
        temperature_folders=[] # Empty list to store temperature folder names
        for item in os.listdir(self.folder):
            try:
                temperature_folders.append(int(item)) # If folder name can be int, add it
            except ValueError:
                pass # Otherwise don't add it
        temperature_folders.sort() # Sort from low to high T
        temperature_folders = [str(i) for i in temperature_folders] # Convert to strings

        print('Found temperature folders:')
        [print('['+str(i)+'] '+folder) for i,folder in enumerate(temperature_folders)]

        self.temperatures = temperature_folders

        if temperatures_list is not None:
            self.temperatures = [str(t) for t in temperatures_list]

        # Load .nxs files
        for temperature in self.temperatures:
            for file in os.listdir(os.path.join(self.folder, temperature)):
                if file.endswith(file_ending):
                    filepath = os.path.join(self.folder, temperature, file)
                    print('-----------------------------------------------')
                    print('Loading ' + temperature + ' K indexed .nxs files...')
                    print('Found ' + filepath)

                    # Load dataset at each temperature
                    self.datasets[temperature] = load_data(filepath)

                    # Initialize scissors object at each temperature
                    self.scissors[temperature] = Scissors()
                    self.scissors[temperature].set_data(self.datasets[temperature])

    def set_window(self, window):
        """
        Set the extents of the integration window.

        Parameters
        ----------
        window : tuple
            Extents of the window for integration along each axis.
        """
        for T in self.temperatures:
            print("----------------------------------")
            print("T = " + T + " K")
            self.scissors[T].set_window(window)

    def set_center(self, center):
        """
        Set the central coordinate for the linecut.

        Parameters
        ----------
        center : tuple
            Central coordinate around which to perform the linecut.
        """
        for T in self.temperatures:
            self.scissors[T].set_center(center)

    def cut_data(self, center=None, window=None, axis=None):
        """
        Perform data cutting for each temperature dataset.

        Parameters
        ----------
        center : tuple
            The center point( for cutting the data.
        window : tuple
            The window size for cutting the data.
        axis : int or None, optional
            The axis along which to perform the cutting. If None, cutting is performed along the
            longest axis in `window`. The default is None.

        Returns
        -------
        list
            A list of linecuts obtained from the cutting operation.
        """

        center = center if center is not None else self.scissors[self.temperatures[0]].center
        window = window if window is not None else self.scissors[self.temperatures[0]].window

        for T in self.temperatures:
            print("-------------------------------")
            print("Cutting T = " + T + " K data...")
            self.scissors[T].cut_data(center, window, axis)
            self.linecuts[T] = self.scissors[T].linecut
        return self.linecuts

    def plot_linecuts(self, vertical_offset=0, **kwargs):
        """
        Plot the linecuts obtained from data cutting.

        Parameters
        ----------
        vertical_offset : float, optional
            The vertical offset between linecuts on the plot. The default is 0.
        **kwargs
            Additional keyword arguments to be passed to the plot function.
        """
        fig, ax = plt.subplots()

        # Get the Viridis colormap
        cmap = mpl.colormaps.get_cmap('viridis')

        for i, linecut in enumerate(self.linecuts.values()):
            x_data = linecut[linecut.axes[0]].nxdata
            y_data = linecut[linecut.signal].nxdata + i*vertical_offset
            ax.plot(x_data, y_data, color=cmap(i / len(self.linecuts)), label=self.temperatures[i],
            	**kwargs)

        xlabel_components = [self.linecuts[self.temperatures[0]].axes[0]
                             if i == self.scissors[self.temperatures[0]].axis \
                                 else str(c) for i,c in enumerate(self.scissors[self.temperatures[0]].center)]
        xlabel = ' '.join(xlabel_components)
        ax.set(xlabel=xlabel,
                ylabel=self.linecuts[self.temperatures[0]].signal)

        # Get the current legend handles and labels
        handles, labels = plt.gca().get_legend_handles_labels()

        # Reverse the order of handles and labels
        handles = handles[::-1]
        labels = labels[::-1]

        # Create a new legend with reversed order
        plt.legend(handles, labels)

        return fig,ax

    def highlight_integration_window(self, temperature=None, **kwargs):
        """
        Displays the integration window plot for a specific temperature, or for the first temperature if
        none is provided.

        Parameters
        ----------
        temperature : str, optional
            The temperature at which to display the integration window plot. If provided, the plot
            will be generated using the dataset corresponding to the specified temperature. If not
            provided, the integration window plots will be generated for the first temperature.
        **kwargs : keyword arguments, optional
            Additional keyword arguments to customize the plot.
        """

        if temperature is not None:
            p = self.scissors[self.temperatures[0]].highlight_integration_window(data=self.datasets[temperature],
                                                                                 **kwargs)
        else:
            p = self.scissors[self.temperatures[0]].highlight_integration_window(
                data=self.datasets[self.temperatures[0]], **kwargs
            )

        return p

    def plot_integration_window(self, temperature=None, **kwargs):
        """
        Plots the three principal cross-sections of the integration volume on a single figure for a specific
        temperature, or for the first temperature if none is provided.

        Parameters
        ----------
        temperature : str, optional
            The temperature at which to plot the integration volume. If provided, the plot
            will be generated using the dataset corresponding to the specified temperature. If not
            provided, the integration window plots will be generated for the first temperature.
        **kwargs : keyword arguments, optional
            Additional keyword arguments to customize the plot.
        """

        if temperature is not None:
            p = self.scissors[self.temperatures[0]].plot_integration_window(**kwargs)
        else:
            p = self.scissors[self.temperatures[0]].plot_integration_window(**kwargs)

        return p
