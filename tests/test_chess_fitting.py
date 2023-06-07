import sys
sys.path.append('../src/nxs_analysis_tools/')
from datareduction import load_data, Scissors
from chess import *
from fitting import *

from lmfit.models import GaussianModel, LinearModel

sample = TempDependence()
sample.load_datasets(folder='../docs/source/examples/example_data/sample_name')
sample.cut_data(center=(0,0,0), window=(0.1,0.75,0.1))

sample.set_model_components([GaussianModel(prefix='peak'), LinearModel(prefix='background')])
sample.set_param_hint('peakcenter', min=-0.1, max=0.1)
sample.make_params()
sample.guess()
sample.print_initial_params()
sample.plot_initial_guess()
sample.fit()
sample.plot_fit()
sample.print_fit_report()