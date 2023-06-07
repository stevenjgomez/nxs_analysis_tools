"""
Module for fitting of linecuts using the lmfit package.
"""

from lmfit.model import Model
from lmfit.model import CompositeModel
import operator
import matplotlib.pyplot as plt
import numpy as np


class LinecutModel:
    def __init__(self):
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
        self.model_result = None

    def set_data(self, data):
        self.data = data
        self.x = data[data.axes[0]].nxdata
        self.y = data[data.signal].nxdata

    def set_model_components(self, model_components):
        """
        Create composite model
        """
        # If the model only has one component, then use it as the model
        if isinstance(model_components, Model):
            self.model = model_components
        # Else, combine the components into a composite model and use that as the
        else:
            self.model_components = model_components
            self.model = CompositeModel(*model_components, operator.add)

    def set_param_hint(self, *args, **kwargs):
        self.model.set_param_hint(*args, **kwargs)

    def make_params(self):
        # Intialize empty parameters (in function)
        params = self.model.make_params()
        self.params = params
        fwhm_params = {key: value for key, value in params.items() if 'fwhm' in key}
        for key, value in fwhm_params.items():
            pi_str = str(np.pi)
            params.add(key.replace('fwhm', 'corrlength'), expr='(2 * ' + pi_str + ') / ' + key)

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

    def plot_initial_guess(self):
        """
        Plot initial guess.
        """
        model = self.model
        params = self.params
        x = self.x
        y = self.y
        y_init_fit = model.eval(params=params, x=x)
        self.y_init_fit = y_init_fit
        plt.plot(x, y_init_fit, '--', label='guess')
        plt.plot(x, y, 'o', label='data')
        plt.legend()
        plt.show()

    def fit(self):
        self.model_result = self.model.fit(self.y, self.params, x=self.x)
        self.y_fit = self.model_result.eval(x=self.x)
        self.y_fit_components = self.model_result.eval_components(x=self.x)

    def plot_fit(self, numpoints=None, **kwargs):
        if numpoints is None:
            numpoints = len(self.x)
        self.x_eval = np.linspace(self.x.min(), self.x.max(), numpoints)
        self.x_eval = np.linspace(self.x.min(), self.x.max(), numpoints)
        self.y_eval = self.model_result.eval(x=self.x_eval)
        self.y_eval_components = self.model_result.eval_components(x=self.x_eval)
        self.model_result.plot(numpoints=numpoints, **kwargs)
        ax = plt.gca()
        for model_component, value in self.y_eval_components.items():
            ax.fill_between(self.x_eval, value, alpha=0.3, label=model_component)
            # ax.plot(self.x_eval, value, label=model_component)
        plt.legend()
        plt.show()
        return ax

    def print_fit_report(self):
        """
        Show fit report.
        """
        print(self.model_result.fit_report())
