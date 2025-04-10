"""
Module for fitting of linecuts using the lmfit package.
"""

import operator
from lmfit.model import Model
from lmfit.model import CompositeModel
import matplotlib.pyplot as plt
import numpy as np


class LinecutModel:
    """
    A class representing a linecut model for data analysis and fitting.

    Attributes
    ----------
    y : array-like or None
        The dependent variable data.
    x : array-like or None
        The independent variable data.
    y_eval : array-like or None
        The evaluated y-values of the fitted model.
    x_eval : array-like or None
        The x-values used for evaluation.
    y_eval_components : dict or None
        The evaluated y-values of the model components.
    y_fit_components : dict or None
        The fitted y-values of the model components.
    y_fit : array-like or None
        The fitted y-values of the model.
    x_fit : array-like or None
        The x-values used for fitting.
    y_init_fit : array-like or None
        The initial guess of the y-values.
    params : Parameters or None
        The parameters of the model.
    model_components : Model or list of Models or None
        The model component(s) used for fitting.
    model : Model or None
        The composite model used for fitting.
    modelresult : ModelResult or None
        The result of the model fitting.
    data : NXdata or None
        The 1D linecut data used for analysis.

    Methods
    -------
    __init__(self, data=None)
        Initialize the LinecutModel.
    set_data(self, data)
        Set the data for analysis.
    set_model_components(self, model_components)
        Set the model components.
    set_param_hint(self, *args, **kwargs)
        Set parameter hints for the model.
    make_params(self)
        Create and initialize the parameters for the model.
    guess(self)
        Perform initial guesses for each model component.
    print_initial_params(self)
        Print out initial guesses for each parameter of the model.
    plot_initial_guess(self, numpoints=None)
        Plot the initial guess.
    fit(self)
        Fit the model to the data.
    plot_fit(self, numpoints=None, fit_report=True, **kwargs)
        Plot the fitted model.
    print_fit_report(self)
        Print the fit report.
    """
    def __init__(self, data=None):
        """
        Initialize the LinecutModel.
        """
        self.y = None
        self.x = None
        self.y_eval = None
        self.x_eval = None
        self.y_eval_components = None
        self.y_fit_components = None
        self.y_fit = None
        self.x_fit = None
        self.y_init_fit = None
        self.params = None
        self.model_components = None
        self.model = None
        self.modelresult = None
        self.data = data if data is not None else None
        if self.data is not None:
            self.x = data[data.axes].nxdata
            self.y = data[data.signal].nxdata

    def set_data(self, data):
        """
        Set the data for analysis.

        Parameters
        ----------
        data : NXdata
            The 1D linecut data to be used for analysis.
        """
        self.data = data
        self.x = data[data.axes].nxdata
        self.y = data[data.signal].nxdata

    def set_model_components(self, model_components):
        """
        Set the model components

        Parameters
        ----------
        model_components : Model or list of Models
            The model component(s) to be used for fitting,
             which will be combined into a CompositeModel.
        """

        # If the model only has one component, then use it as the model
        if isinstance(model_components, Model):
            self.model = model_components
        # Else, combine the components into a composite model and use that as the
        else:
            self.model_components = model_components
            self.model = model_components[0]

            # Combine remaining components into the composite model
            for component in model_components[1:]:
                self.model = CompositeModel(self.model, component, operator.add)

    def set_param_hint(self, *args, **kwargs):
        """
        Set parameter hints for the model.

        Parameters
        ----------
        *args : positional arguments
            Positional arguments passed to the `set_param_hint` method of the underlying model.

        **kwargs : keyword arguments
            Keyword arguments passed to the `set_param_hint` method of the underlying model.
        """

        self.model.set_param_hint(*args, **kwargs)

    def make_params(self):
        """
        Create and initialize the parameters for the model.

        Returns
        -------
        Parameters
            The initialized parameters for the model.
        """
        # Initialize empty parameters (in function)
        params = self.model.make_params()
        self.params = params

        return params

    def guess(self):
        """
        Perform initial guesses for each model component.
        """
        for model_component in list(self.model_components):
            self.params.update(model_component.guess(self.y, x=self.x))

    def print_initial_params(self):
        """
        Print out initial guesses for each parameter of the model.
        """
        model = self.model
        for param, hint in model.param_hints.items():
            print(f'{param}')
            for key, value in hint.items():
                print(f'\t{key}: {value}')

    def plot_initial_guess(self, numpoints=None):
        """
        Plot initial guess.
        """
        model = self.model
        params = self.params
        x = self.x
        y = self.y
        y_init_fit = model.eval(params=params, x=x)
        self.y_init_fit = y_init_fit
        plt.plot(x, y, 'o', label='data')
        plt.plot(x, y_init_fit, '--', label='guess')

        # Plot the components of the model
        if numpoints is None:
            numpoints = len(self.x)
        self.x_eval = np.linspace(self.x.min(), self.x.max(), numpoints)
        y_init_fit_components = model.eval_components(params=params, x=self.x_eval)
        ax = plt.gca()
        for model_component, value in y_init_fit_components.items():
            ax.fill_between(self.x_eval, value, alpha=0.3, label=model_component)
        plt.legend()
        plt.show()

    def fit(self):
        """
        Fit the model to the data.

        This method fits the model to the data using the specified parameters and the x-values.
        It updates the model result, fitted y-values, and the evaluated components.

        """
        self.modelresult = self.model.fit(self.y, self.params, x=self.x)
        self.y_fit = self.modelresult.eval(x=self.x)
        self.y_fit_components = self.modelresult.eval_components(x=self.x)

    def plot_fit(self, numpoints=None, fit_report=True, **kwargs):
        """
        Plot the fitted model.

        This method plots the fitted model along with the original data.
        It evaluates the model and its components at the specified number of points (numpoints)
        and plots the results.

        Parameters
        ----------
        numpoints : int, optional
            Number of points to evaluate the model and its components. If not provided,
            it defaults to the length of the x-values.

        fit_report : bool, optional
            Whether to print the fit report. Default is True.

        **kwargs : dict, optional
            Additional keyword arguments to be passed to the `plot` method.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The Axes object containing the plot.

        """
        if numpoints is None:
            numpoints = len(self.x)
        self.x_eval = np.linspace(self.x.min(), self.x.max(), numpoints)
        self.y_eval = self.modelresult.eval(x=self.x_eval)
        self.y_eval_components = self.modelresult.eval_components(x=self.x_eval)
        self.modelresult.plot(numpoints=numpoints, **kwargs)
        ax = plt.gca()
        for model_component, value in self.y_eval_components.items():
            ax.fill_between(self.x_eval, value, alpha=0.3, label=model_component)
            # ax.plot(self.x_eval, value, label=model_component)
        plt.legend()
        plt.show()
        if fit_report:
            print(self.modelresult.fit_report())
        return ax

    def print_fit_report(self):
        """
        Show fit report.
        """
        print(self.modelresult.fit_report())
