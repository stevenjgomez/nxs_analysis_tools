'''
This module provides classes and functions for analyzing scattering datasets collected at CHESS
(ID4B) with temperature dependence. It includes functions for loading data, cutting data, and
plotting linecuts.
'''
import os
from nexusformat.nexus import NXentry
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
        self.datasets=NXentry()
        self.folder=None
        self.temperatures=None
        self.scissors=None
        self.linecuts=None

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
        self.datasets=NXentry()

    def load_datasets(self, folder, file_ending='hkli.nxs', temperatures_list=None):
        '''
        Load scattering datasets from the specified folder.

        Parameters
        ----------
        folder : str
            The path to the folder where the datasets are located.
        file_ending : str, optional
            The file ending of the datasets to be loaded. The default is 'hkli.nxs'.
        temperatures_list : list of int or None, optional
            The list of specific temperatures to load. If None, all available temperatures are
            loaded. The default is None.
        '''
        self.folder = os.path.normpath(folder)

        temperature_folders=[] # Empty list to store temperature folder names
        for item in os.listdir(self.folder):
            try:
                temperature_folders.append(int(item)) # If folder name can be int, add it
            except ValueError:
                pass # Otherwise don't add it
        temperature_folders.sort() # Sort from low to high T
        temperature_folders = [str(i) for i in temperature_folders] # Convert to strings

        print('Found temperature folders:')
        [print('['+str(i)+'] '+folder for i,folder in temperature_folders)]

        self.temperatures = temperature_folders

        if temperatures_list is not None:
            temperature_folders = [str(t) for t in temperatures_list]
        else:
            temperature_folders = self.temperatures

        self.temperatures = temperature_folders

        # Load .nxs files
        for temperature in temperature_folders:
            for file in os.listdir(os.path.join(self.folder,temperature)):
                if file.endswith(file_ending):
                    filepath = os.path.join(self.folder,temperature, file)
                    print('-----------------------------------------------')
                    print('Loading ' + temperature + ' K indexed .nxs files...')
                    print('Found ' + filepath)
                    self.datasets[temperature] = load_data(filepath)

    def cut_data(self, center, window, axis=None):
        '''
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
        '''
        self.scissors = [Scissors() for _ in range(len(self.temperatures))]
        for i,T in enumerate(self.temperatures):
            print("-------------------------------")
            print("Cutting T = " + T + " K data...")
            self.scissors[i].set_data(self.datasets[T])
            self.scissors[i].cut_data(center, window, axis)

        self.linecuts = [scissors.linecut for scissors in self.scissors]
        return self.linecuts

    def plot_linecuts(self, vertical_offset=0, **kwargs):
        '''
        Plot the linecuts obtained from data cutting.

        Parameters
        ----------
        vertical_offset : float, optional
            The vertical offset between linecuts on the plot. The default is 0.
        **kwargs
            Additional keyword arguments to be passed to the plot function.
        '''
        fig, ax = plt.subplots()

        # Get the Viridis colormap
        cmap = mpl.colormaps.get_cmap('viridis')

        for i, linecut in enumerate(self.linecuts):
            x_data = linecut[linecut.axes[0]].nxdata
            y_data = linecut[linecut.signal].nxdata + i*vertical_offset
            ax.plot(x_data, y_data, color=cmap(i / len(self.linecuts)), label=linecut.nxname,
            	**kwargs)

        xlabel_components = [self.linecuts[0].axes[0] if i == self.scissors[0].axis \
        	else str(c) for i,c in enumerate(self.scissors[0].center)]
        xlabel = ' '.join(xlabel_components)
        ax.set(xlabel=xlabel,
                ylabel=self.linecuts[0].signal)

        # Get the current legend handles and labels
        handles, labels = plt.gca().get_legend_handles_labels()

        # Reverse the order of handles and labels
        handles = handles[::-1]
        labels = labels[::-1]

        # Create a new legend with reversed order
        plt.legend(handles, labels)
        plt.show()