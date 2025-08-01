"""
This module provides classes and functions for analyzing scattering datasets collected at CHESS
(ID4B) with temperature dependence. It includes functions for loading temperature series and 
performing operations on all datasets in the series at once (e.g., cutting, fitting). 
"""
import os
import re

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
from IPython.display import display, Markdown
from nxs_analysis_tools import load_data, Scissors
from nxs_analysis_tools.fitting import LinecutModel
from nxs_analysis_tools.datareduction import load_transform, reciprocal_lattice_params
from lmfit.models import PseudoVoigtModel, LinearModel


class TempDependence:
    """
    A class for analyzing temperature-dependent scattering datasets collected at CHESS (ID4B).

    The `TempDependence` class facilitates the loading, processing, and analysis of scattering
    data across different temperatures. It includes methods for handling datasets, setting
    lattice parameters, performing linecuts, modeling the data, and visualizing the results.

    Attributes
    ----------
    sample_directory : str
        Path to the directory containing the datasets.
    xlabel : str
        Label for the x-axis of plots, determined by the axis of the linecuts.
    datasets : dict
        Dictionary storing datasets keyed by temperature.
    temperatures : list of str
        List of temperatures for which data is available.
    scissors : dict
        Dictionary of Scissors objects, one for each temperature, used for data manipulation and
        linecut operations.
    linecuts : dict
        Dictionary storing the linecut data for each temperature.
    linecutmodels : dict
        Dictionary of LinecutModel objects, one for each temperature, used for fitting the linecuts.
    a, b, c, al, be, ga : float or None
        Lattice parameters (a, b, c, alpha, beta, gamma) of the crystal.
    a_star, b_star, c_star, al_star, be_star, ga_star : float or None
        Reciprocal lattice parameters (a*, b*, c*, alpha*, beta*, gamma*).

    Methods
    -------
    set_temperatures(temperatures):
        Set the list of temperatures for the datasets.
    find_temperatures():
        Set the list of temperatures by automatically scanning the sample directory.
    set_sample_directory(path):
        Set the directory path where the datasets are located.
    initialize():
        Initialize Scissors and LinecutModel objects for each temperature.
    set_data(temperature, data):
        Set the dataset for a specific temperature.
    load_transforms(temperatures_list=None, exclude_temperatures=None, print_tree=True):
        Load transform datasets (from nxrefine) based on temperature.
    load_datasets(file_ending='hkli.nxs', temperatures_list=None, exclude_temperatures=None, 
                  print_tree=True):
        Load datasets (legacy CHESS format) from the specified folder.
    get_sample_directory():
        Get the folder path where the datasets are located.
    clear_datasets():
        Clear the datasets stored in the TempDependence instance.
    set_Lattice_params(lattice_params):
        Set lattice parameters and calculate reciprocal lattice parameters.
    set_window(window, verbose=False):
        Set the extents of the integration window for each temperature.
    set_center(center):
        Set the central coordinate for the linecut for each temperature.
    cut_data(center=None, window=None, axis=None, verbose=False):
        Perform data cutting for each temperature dataset.
    plot_linecuts(vertical_offset=0, **kwargs):
        Plot the linecuts obtained from data cutting.
    plot_linecuts_heatmap(ax=None, **kwargs):
        Plot a heatmap of the linecuts obtained from data cutting.
    highlight_integration_window(temperature=None, **kwargs):
        Display the integration window plot for a specific temperature.
    plot_integration_window(temperature=None, **kwargs):
        Plot the integration window cross-sections for a specific temperature.
    set_model_components(model_components):
        Set the model components for all line cut models.
    set_param_hint(*args, **kwargs):
        Set parameter hints for all line cut models.
    make_params():
        Create parameters for all line cut models.
    guess():
        Make initial parameter guesses for all line cut models.
    print_initial_params():
        Print the initial parameter values for all line cut models.
    plot_initial_guess():
        Plot the initial guess for all line cut models.
    fit(verbose=False):
        Fit the line cut models for each temperature.
    plot_fit(mdheadings=False, **kwargs):
        Plot the fit results for each temperature.
    overlay_fits(numpoints=None, vertical_offset=0, cmap='viridis', ax=ax, 
                 data_kwargs=None, fit_kwargs=None):
        Plot raw data and fitted models for each temperature.
    fit_peak_simple():
        Perform a basic fit using a pseudo-Voigt peak shape, linear background, and no constraints.
    plot_order_parameter(ax, param_name='peakheight', **kwargs):
        Plot the temperature dependence of the peakheight parameter.
    print_fit_report():
        Print the fit report for each temperature.
    """

    def __init__(self, sample_directory=None):
        """
        Initialize the TempDependence class.

        Parameters
        ----------
        sample_directory : str, optional
            Path to the directory containing the temperature folders.
            If None, no directory is set initially.
        """

        if sample_directory is None:
            self.sample_directory = None
        else:
            self.set_sample_directory(sample_directory)

        self.xlabel = ''
        self.datasets = {}
        self.temperatures = []
        self.scissors = {}
        self.linecuts = {}
        self.linecutmodels = {}
        self.a, self.b, self.c, self.al, self.be, self.ga, \
            self.a_star, self.b_star, self.c_star, self.al_star, self.be_star, self.ga_star \
            = [None] * 12

    def set_temperatures(self, temperatures):
        """
        Set the list of temperatures for the datasets.

        Parameters
        ----------
        temperatures : list
            List of temperatures to set.
        """
        self.temperatures = temperatures

    def find_temperatures(self):
        """
        Set the list of temperatures by automatically scanning the sample directory for .nxs files from nxrefine.
        """

        # Assert that self.sample_directory must exist
        if self.sample_directory is None:
            raise ValueError("Sample directory is not set. Use set_sample_directory(path) first.")

        # Clear existing temperatures
        self.temperatures = []

        # Search for nxrefine .nxs files
        for item in os.listdir(self.sample_directory):
            pattern = r'_(\d+)\.nxs'
            match = re.search(pattern, item)
            if match:
                # Identify temperature
                temperature = match.group(1)
                self.temperatures.append(temperature)
        # Convert all temperatures to int temporarily to sort temperatures list
        self.temperatures = [int(t) for t in self.temperatures]
        self.temperatures.sort()
        self.temperatures = [str(t) for t in self.temperatures]

    def set_sample_directory(self, path):
        """
        Set the directory path where the datasets are located.

        Parameters
        ----------
        path : str
            Path to the sample directory.
        """
        self.sample_directory = os.path.normpath(path)

    def initialize(self):
        """
        Initialize Scissors and LinecutModel objects for each temperature.
        """
        for temperature in self.temperatures:
            self.scissors[temperature] = Scissors()
            if temperature in self.datasets.keys():
                self.scissors[temperature].set_data(self.datasets[temperature])
            self.linecutmodels[temperature] = LinecutModel()

    def set_data(self, temperature, data):
        """
        Set the dataset for a specific temperature.

        Parameters
        ----------
        temperature : str
            Temperature for which to set the data.
        data : object
            The dataset to be set.
        """
        self.datasets[temperature] = data

    def load_transforms(self, temperatures_list=None, exclude_temperatures=None, print_tree=True, use_nxlink=False):
        """
        Load transform datasets (from nxrefine) based on temperature.

        Parameters
        ----------
        temperatures_list : list of int or None, optional
            List of temperatures to load. If None, all available temperatures are loaded.

        exclude_temperatures : int, str, optional
            Temperatures to skip. Applied after filtering with `temperatures_list`, if provided.
        
        print_tree : bool, optional
            Whether to print the data tree upon loading. Default True.
        
        use_nxlink : bool, optional
            If True, maintains the NXlink defined in the data file, which references
            the raw data in the transform.nxs file. This saves memory when working with
            many datasets. In this case, the axes are in reverse order. Default is False.
        """
        # Convert all temperatures to strings
        if temperatures_list:
            temperatures_list = [str(t) for t in temperatures_list]
        if exclude_temperatures:
            if isinstance(exclude_temperatures, str):
                exclude_temperatures = [exclude_temperatures]
            exclude_temperatures = [str(t) for t in list(exclude_temperatures)]

        # Clear existing temperatures before loading files
        self.temperatures = []

        # Identify files to load
        items_to_load = []
        # Search for nxrefine .nxs files
        for item in os.listdir(self.sample_directory):
            pattern = r'_(\d+)\.nxs'
            match = re.search(pattern, item)
            if match:
                # Identify temperature
                temperature = match.group(1)
                # print(f'Temperature = {temperature}')
                if temperatures_list is not None:
                    incl_temp = temperature in temperatures_list
                else:
                    incl_temp = True
                if exclude_temperatures is not None:
                    not_excl_temp = temperature not in exclude_temperatures
                else:
                    not_excl_temp = True
                if incl_temp and not_excl_temp:
                    # Prepare file to be loaded
                    self.temperatures.append(temperature)
                    items_to_load.append(item)
                    # print(f'Preparing to load {temperature} K data: {item}')

        # Convert all temperatures to int temporarily to sort temperatures list before loading
        self.temperatures = [int(t) for t in self.temperatures]

        loading_template = pd.DataFrame({'temperature': self.temperatures,
                                         'filename': items_to_load})
        loading_template = loading_template.sort_values(by='temperature')
        self.temperatures = loading_template['temperature']
        self.temperatures = [str(t) for t in self.temperatures]
        items_to_load = loading_template['filename'].to_list()

        for i, item in enumerate(items_to_load):
            path = os.path.join(self.sample_directory, item)

            # Ensure path is a string before using it
            path = str(path)

            # Save dataset
            try:
                self.datasets[self.temperatures[i]] = load_transform(path, print_tree=print_tree, use_nxlink=use_nxlink)
            except Exception as e:
                # Report temperature that was unable to load, then raise exception.
                temp_failed = self.temperatures[i]
                print(f"Failed to load data for temperature {temp_failed} K from file {item}."
                      f" Error: {e}")
                raise  # Re-raise the exception

        self.initialize()

    def load_datasets(self, file_ending='hkli.nxs', temperatures_list=None, exclude_temperatures=None, print_tree=True):
        """
        Load datasets (CHESS format) from the specified folder.

        Parameters
        ----------
        file_ending : str, optional
            File extension of datasets to load. Default is 'hkli.nxs'.
        temperatures_list : list of int or str, optional
            Specific temperatures to load. If None, all temperatures are loaded.
        exclude_temperatures : list of int or str, optional
            Temperatures to skip. Applied after filtering with `temperatures_list`, if provided.
        print_tree : bool, optional
            If True, prints the NeXus tree structure for each file. Default is True.
        """

        if temperatures_list is not None:
            self.temperatures = [str(t) for t in temperatures_list]
        else:
            self.temperatures = []  # Empty list to store temperature folder names
            for item in os.listdir(self.sample_directory):
                try:
                    self.temperatures.append(int(item))  # If folder name can be int, add it
                except ValueError:
                    pass  # Otherwise don't add it
            self.temperatures.sort()  # Sort from low to high T
            self.temperatures = [str(i) for i in self.temperatures]  # Convert to strings
            
        if exclude_temperatures is not None:
            [self.temperatures.remove(str(t)) for t in exclude_temperatures]

        # Load .nxs files
        for T in self.temperatures:
            for file in os.listdir(os.path.join(self.sample_directory, T)):
                if file.endswith(file_ending):
                    filepath = os.path.join(self.sample_directory, T, file)

                    # Load dataset at each temperature
                    self.datasets[T] = load_data(filepath, print_tree)

        self.initialize()

    def get_sample_directory(self):
        """
        Get the folder path where the datasets are located.

        Returns
        -------
        str
            The folder path.
        """
        return self.sample_directory

    def clear_datasets(self):
        """
        Clear the datasets stored in the TempDependence instance.
        """
        self.datasets = {}

    def set_lattice_params(self, lattice_params):
        """
        Set lattice parameters and calculate reciprocal lattice parameters.

        Parameters
        ----------
        lattice_params : tuple
            Tuple containing lattice parameters (a, b, c, al, be, ga).
        """
        self.a, self.b, self.c, self.al, self.be, self.ga = lattice_params
        self.a_star, self.b_star, self.c_star, \
            self.al_star, self.be_star, self.ga_star = reciprocal_lattice_params(lattice_params)

    def set_window(self, window, verbose=False):
        """
        Set the extents of the integration window for each temperature.

        Parameters
        ----------
        window : tuple
            Extents of the window for integration along each axis.
        verbose : bool, optional
            Enables printout of linecut axis and integrated axes. Default is False.
        """
        for T in self.temperatures:
            if verbose:
                print("----------------------------------")
                print("T = " + T + " K")
            self.scissors[T].set_window(window, verbose)

    def set_center(self, center):
        """
        Set the central coordinate for the linecut for each temperature.

        Parameters
        ----------
        center : tuple
            Central coordinate around which to perform the linecut.
        """
        for T in self.temperatures:
            self.scissors[T].set_center(center)

    def cut_data(self, center=None, window=None, axis=None, verbose=False):
        """
        Perform data cutting for each temperature dataset.

        Parameters
        ----------
        center : tuple, optional
            The center point for cutting the data.
            Defaults to the first temperature's center if None.
        window : tuple, optional
            The window size for cutting the data.
            Defaults to the first temperature's window if None.
        axis : int or None, optional
            The axis along which to perform the cutting.
            Defaults to the longest axis in `window` if None.
        verbose : bool, optional
            Enables printout of linecut progress. Default is False.

        Returns
        -------
        dict
            A dictionary of linecuts obtained from the cutting operation.
        """

        for T in self.temperatures:
            if verbose:
                print("-------------------------------")
                print("Cutting T = " + T + " K data...")
            self.scissors[T].set_center(center)
            self.scissors[T].set_window(window)
            self.scissors[T].cut_data(axis=axis, verbose=verbose)
            self.linecuts[T] = self.scissors[T].linecut
            self.linecutmodels[T].set_data(self.linecuts[T])

        xlabel_components = [self.linecuts[self.temperatures[0]].axes
                             if i == self.scissors[self.temperatures[0]].axis
                             else str(c) for i, c in
                             enumerate(self.scissors[self.temperatures[0]].center)]
        self.xlabel = ' '.join(xlabel_components)

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

        # Reverse zorder
        zorder = 0

        for i, linecut in enumerate(self.linecuts.values()):
            
            x_data = linecut[linecut.axes].nxdata
            y_data = linecut[linecut.signal].nxdata + i * vertical_offset
            ax.plot(x_data, y_data, color=cmap(i / len(self.linecuts)), label=self.temperatures[i], 
                    zorder=zorder, **kwargs)
            zorder -= 1

        ax.set(xlabel=self.xlabel,
               ylabel=self.linecuts[self.temperatures[0]].signal)

        # Get the current legend handles and labels
        handles, labels = plt.gca().get_legend_handles_labels()

        # Reverse the order of handles and labels
        handles = handles[::-1]
        labels = labels[::-1]

        # Create a new legend with reversed order
        plt.legend(handles, labels)

        return fig, ax

    def plot_linecuts_heatmap(self, ax=None, **kwargs):
        """
        Plot the linecuts obtained from data cutting.

        Parameters
        ----------
        ax : :class:`matplotlib.axes.Axes`, optional
            The axes on which to plot the heatmap. If None, a new figure and axes
            are created. The default is None.
        **kwargs
            Additional keyword arguments to be passed to the `pcolormesh` function.

        Returns
        -------
        QuadMesh
            The plotted heatmap object.
        """

        # Retrieve linecut data for the first temperature and extract x-axis data
        cut = self.linecuts[self.temperatures[0]]
        x = cut[cut.axes].nxdata

        # Convert the list of temperatures to a NumPy array for the y-axis
        y = np.array([int(t) for t in self.temperatures])

        # Collect counts from each temperature and ensure they are numpy arrays
        v = [self.linecuts[T].nxsignal.nxdata for T in self.temperatures]

        # Convert list of arrays to a 2D array for the heatmap
        v_2d = np.array(v)

        # Create the grid for the heatmap
        X, Y = np.meshgrid(x, y)

        # Plot using pcolormesh
        if ax is None:
            _, ax = plt.subplots()
        p = ax.pcolormesh(X, Y, v_2d, **kwargs)
        plt.colorbar(p, label='counts')
        ax.set(xlabel=self.xlabel, ylabel=r'$T$ (K)')

        return p

    def highlight_integration_window(self, temperature=None, **kwargs):
        """
        Displays the integration window plot for a specific temperature,
        or for the first temperature if none is provided.

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
            p = self.scissors[
                self.temperatures[0]].highlight_integration_window(
                data=self.datasets[temperature], **kwargs
            )
        else:
            p = self.scissors[self.temperatures[0]].highlight_integration_window(
                data=self.datasets[self.temperatures[0]], **kwargs
            )

        return p

    def plot_integration_window(self, temperature=None, **kwargs):
        """
        Plots the three principal cross-sections of the integration volume on
        a single figure for a specific temperature, or for the first temperature
        if none is provided.

        Parameters
        ----------
        temperature : str, optional
            The temperature at which to plot the integration volume. If provided,
            the plot will be generated using the dataset corresponding to the
            specified temperature. If not provided, the integration window plots
            will be generated for the first temperature.

        **kwargs : keyword arguments, optional
            Additional keyword arguments to customize the plot.
        """

        if temperature is not None:
            p = self.scissors[self.temperatures[0]].plot_integration_window(**kwargs)
        else:
            p = self.scissors[self.temperatures[0]].plot_integration_window(**kwargs)

        return p

    def set_model_components(self, model_components):
        """
        Set the model components for all line cut models.

        This method sets the same model components for all line cut models in the
        analysis. It iterates over each line cut model and calls their respective
        `set_model_components` method with the provided `model_components`.

        Parameters
        ----------
        model_components : Model, CompositeModel, or iterable of Model
            The model components to set for all line cut models.

        """
        [linecutmodel.set_model_components(model_components) for
         linecutmodel in self.linecutmodels.values()]

    def set_param_hint(self, *args, **kwargs):
        """
        Set parameter hints for all line cut models.

        This method sets the parameter hints for all line cut models in the analysis.
        It iterates over each line cut model and calls their respective `set_param_hint` method
        with the provided arguments and keyword arguments. These are implemented when the
        .make_params() method is called.

        Parameters
        ----------
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.

        """
        [linecutmodel.set_param_hint(*args, **kwargs)
         for linecutmodel in self.linecutmodels.values()]
    
    def params_set(self, name, **kwargs):
        """
        Set constraints on a parameter for all line cut models.

        This method updates the specified parameter across all models in
        `self.linecutmodels` using the keyword arguments provided. These
        keyword arguments are passed to the `set()` method of the parameter,
        which comes from a `lmfit.Parameters` object.

        Parameters
        ----------
        name : str
            Name of the parameter to modify (must exist in each model).
        **kwargs
            Constraint arguments passed to `Parameter.set()`, such as `value`,
            `min`, `max`, `vary`, etc.

        Raises
        ------
        KeyError
            If the parameter `name` does not exist in one of the models.

        Example
        -------
        >>> sample.params_set('peakamplitude', value=5, min=0, vary=True)
        """

        for linecutmodel in self.linecutmodels.values():
            linecutmodel.params[name].set(**kwargs)

    def make_params(self):
        """
        Create and initialize the parameters for all models.

        This method creates the parameters for all line cut models in the analysis.
        It iterates over each line cut model and calls their respective `make_params` method.
        """
        [linecutmodel.make_params() for linecutmodel in self.linecutmodels.values()]

    def guess(self):
        """
        Make initial parameter guesses for all line cut models. This overwrites any prior initial
        values and constraints.

        This method generates initial parameter guesses for all line cut models in the analysis.
        It iterates over each line cut model and calls their respective `guess` method.

        """
        [linecutmodel.guess() for linecutmodel in self.linecutmodels.values()]

    def print_initial_params(self):
        """
        Print the initial parameter values for all line cut models.

        This method prints the initial parameter values for all line cut models
        in the analysis. It iterates over each line cut model and calls their
        respective `print_initial_params` method.

        """
        [linecutmodel.print_initial_params() for linecutmodel in self.linecutmodels.values()]

    def plot_initial_guess(self):
        """
        Plot the initial guess for all line cut models.

        This method plots the initial guess for all line cut models in the analysis.
        It iterates over each line cut model and calls their respective `plot_initial_guess` method.

        """
        for T, linecutmodel in self.linecutmodels.items():
            _, ax = plt.subplots()
            ax.set(title=T + ' K')
            linecutmodel.plot_initial_guess()

    def fit(self, verbose=False):
        """
        Fit the line cut models.

        This method fits the line cut models for each temperature in the analysis.
        It iterates over each line cut model, performs the fit, and prints the fitting progress.

        Parameters
        ----------
       verbose : bool, optional
            Enables printout of fitting progress. Default False.

        """
        for T, linecutmodel in self.linecutmodels.items():
            if verbose:
                print(f"Fitting {T} K  data...")
            linecutmodel.fit()
            if verbose:
                print("Done.")
        print("Fits completed.")

    def plot_fit(self, mdheadings=False, **kwargs):
        """
        Plot the fit results.

        This method plots the fit results for each temperature in the analysis.
        It iterates over each line cut model, calls their respective `plot_fit` method,
        and sets the xlabel, ylabel, and title for the plot.

        """
        for T, linecutmodel in self.linecutmodels.items():
            # Create a markdown heading for the plot
            if mdheadings:
                display(Markdown(f"### {T} K Fit Results"))
            # Plot fit
            linecutmodel.plot_fit(xlabel=self.xlabel,
                                  ylabel=self.datasets[self.temperatures[0]].signal,
                                  title=f"{T} K",
                                  **kwargs)

    def overlay_fits(self, numpoints=None, vertical_offset=0, cmap='viridis', ax=None,
                     data_kwargs=None, fit_kwargs=None):
        """
        Plot raw data and fitted models for each temperature with optional vertical offsets.

        Parameters:
        -----------
        numpoints : int or None, default=None
            Number of points to evaluate for the fitted model curves.
            If None, uses the number of raw data points for each linecut.
        vertical_offset : float, default=0
            Amount to vertically offset each linecut for clarity.
        cmap : str, default='viridis'
            Name of the matplotlib colormap used to distinguish different temperatures.
        ax : :class:`matplotlib.axes.Axes` or None, default=None
            Axis object to plot on. If None, a new figure and axis are created.
        data_kwargs : dict
            Keyword arguments to be passed to the data plot function.
        fit_kwargs : dict
            Keyword arguments to be passed to the fit plot function.


        The function:
        - Uses a colormap to assign unique colors to each temperature.
        - Plots raw data alongside evaluated fit models for each linecut.
        - Vertically offsets each trace by a constant value for visual separation.
        - Displays a legend in reverse order to match top-to-bottom visual stacking.
        - Automatically labels the x- and y-axes based on NeXus-style data metadata.
        """

        # Create a figure and axes if an axis is not already provided
        _, ax = plt.subplots() if ax is None else (None, ax)

        if data_kwargs is None:
            data_kwargs = {}
        if fit_kwargs is None:
            fit_kwargs = {}

        # Generate a color palette for the various temperatures
        cmap = plt.get_cmap(cmap)
        colors = [cmap(i / len(self.temperatures)) for i, _ in enumerate(self.temperatures)]

        for i, lm in enumerate(self.linecutmodels.values()):
            # Plot the raw data
            ax.plot(lm.x, lm.y + vertical_offset * i, '.', c=colors[i], **data_kwargs)

            # Evaluate the fit
            numpoints = len(lm.x) if numpoints is None else numpoints
            x_eval = np.linspace(lm.x.min(), lm.x.max(), numpoints)
            y_eval = lm.modelresult.eval(x=x_eval)
            ax.plot(x_eval, y_eval + vertical_offset * i, '-', c=colors[i], label=self.temperatures[i], **fit_kwargs)

        # Reverse legend entries to match top-to-bottom stacking
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1])

        # Add axis labels
        ax.set(xlabel=lm.data.nxaxes[0].nxname, ylabel=lm.data.nxsignal.nxname)
        
    def fit_peak_simple(self):
        """
        Fit all linecuts in the temperature series using a pseudo-Voigt peak shape and linear
        background, with no constraints.
        """

        for T in self.temperatures:
            linecutmodel = self.linecutmodels[T]
            linecutmodel.set_model_components([PseudoVoigtModel(prefix='peak'),
                                                LinearModel(prefix='background')])
            linecutmodel.make_params()
            linecutmodel.guess()
            linecutmodel.params['peakamplitude'].set(min=0)
            linecutmodel.fit()

    def plot_order_parameter(self, param_name='peakheight', ax=None, **kwargs):
        """
        Plot the temperature dependence of the peak height (order parameter).

        This method extracts the values of a chosen parameter from each temperature-dependent
        line cut fit stored in `linecutmodels` and plots it as a function of temperature.

        Parameters
        ----------
        ax : :class:`matplotlib.axes.Axes`, optional
            Axis object to plot on. If None, a new figure and axis are created.
        param_name : str, optional
            The name of the lmfit parameter to extract. Default is 'peakheight'.
        **kwargs
            Keyword arguments to be passed to the plot function.


        Returns
        -------
        Figure
            Matplotlib Figure object containing the peak height vs. temperature plot.
        Axes
            Matplotlib Axes object associated with the figure.

        Notes
        -----
        - Temperature values are converted to integers for plotting.
        - Peak heights are extracted from the 'peakheight' parameter in the model results.
        - The plot uses standard axes labels with temperature in Kelvin.
        """

        # Create an array of temperature values
        temperatures = [int(T) for T in self.temperatures]

        # Create an empty list for the peak heights
        peakheights = []

        # Extract the peakheight at every temperature
        for T in self.temperatures:
            
            # Verify that the fit has already been completed
            if self.linecutmodels[T].modelresult is None:
                raise AttributeError("Model result is empty. Have you fit the data to a model?")
            
            peakheights.append(self.linecutmodels[T].modelresult.params[param_name].value)

        # Plot the peakheights vs. temperature
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        ax.plot(temperatures, peakheights, **kwargs)
        ax.set(xlabel='$T$ (K)', ylabel=param_name)
        return fig, ax

    def print_fit_report(self):
        """
        Plot the fit results.

        This method plots the fit results for each temperature in the analysis.
        It iterates over each line cut model, calls their respective `plot_fit` method,
        and sets the xlabel, ylabel, and title for the plot.

        """
        for T, linecutmodel in self.linecutmodels.items():
            print(f"[[[{T} K Fit Report]]]")
            linecutmodel.print_fit_report()
