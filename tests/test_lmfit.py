import sys

sys.path.append('../src/nxs_analysis_tools/')
from datareduction import load_data, Scissors
# from fitting import *


from test_lmfit.models import GaussianModel
from test_lmfit.models import PseudoVoigtModel
from test_lmfit.models import LorentzianModel
from test_lmfit.models import LinearModel
from test_lmfit.model import CompositeModel
import operator
import matplotlib.pyplot as plt
import numpy as np

data = load_data('../docs/source/examples/example_data/sample_name/15/example_hkli.nxs')
s = Scissors(data=data, center=(0, 0, 0), window=(0.1, 0.1, 0.5))
s.cut_data()

x = s.linecut[s.linecut.axes[0]].nxdata
y = s.linecut[s.linecut.signal].nxdata

# Create model components
model_components = [
    GaussianModel(prefix='peak'),
    LinearModel(prefix='background')
]

# Create composite model (in function)
model = CompositeModel(*model_components, operator.add)

# Intialize empty parameters (in function)
params = model.make_params()
fwhm_params = {key:value for key,value in params.items() if 'fwhm' in key}
for key,value in fwhm_params.items():
    pi_str = str(np.pi)
    params.add(key.replace('fwhm','corrlength'), expr='(2 * ' + pi_str + ') / ' + key)

# Perform initial guesses for each model component (in function)
for model_component in model_components:
    params.update(model_component.guess(y, x=x))

# Show initial guesses (in function)
for param, hint in model.param_hints.items():
    print(f'{param}')
    for key, value in hint.items():
        print(f'\t{key}: {value}')

# Plot initial guess (in function)
y_init_fit = model.eval(params=params, x=x)
plt.plot(x, y_init_fit, '--', label='guess')
plt.plot(x, y, 'o', label='data')
plt.legend()
plt.show()

# Perform fit (in function)
model_result = model.fit(y, params, x=x)
# # Determine correlation lengths after fitting
# fwhm_params = {key:value for key,value in model_result.params.items() if 'fwhm' in key}
# for key,value in fwhm_params.items():
#     corrlength = correlation_length = (2 * np.pi) / value
#     new_key = key.replace('fwhm','corrlength')
#     model_result.params.add(new_key, value=corrlength)

x_fit = x
# x_fit = np.linspace(x.min(), x.max(), num_points)
y_fit = model_result.eval(x=x_fit)
y_fit_components = model_result.eval_components(x=x_fit)

# Plot fit with **kwargs (in function)
model_result.plot(show_init=True)
plt.show()

# Show fit report
print(model_result.fit_report())

# # Calculate correlation length (can be done from expr in lmfit)
# fwhm_entries = {key: value for key, value in model_result.values.items() if 'fwhm' in key}
# correlation_lengths = {}
# for key, fwhm in fwhm_entries.items():
#     correlation_length = (2 * np.pi) / fwhm
#     new_key = key.replace('fwhm', 'correlationlength')
#     correlation_lengths[new_key] = correlation_length
#
# fwhm_errors = {key: param.stderr for key, param in model_result.params.items() if 'fwhm' in key}
# correlation_length_errors = {}
# for key, error in fwhm_errors.items():
#     new_key = key.replace('fwhm', 'correlationlength')
#     correlation_length_errors[new_key] = ((2 * np.pi) / fwhm_entries[key] ** 2) * error if error is not None else None
#
# print('[[Correlation Lengths]]')
# for key in correlation_lengths.keys():
#     print(f'\t{key}: {correlation_lengths[key]} +/- {correlation_length_errors[key]} ' +
#           f'({correlation_length_errors[key] / correlation_lengths[key] * 100:.2f}%)')