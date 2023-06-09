import sys
sys.path.append('../src/nxs_analysis_tools/')
from fitting import LinecutModel

from nxs_analysis_tools.datareduction import load_data, Scissors
from lmfit.models import GaussianModel, LinearModel

data = load_data('../docs/source/examples/example_data/sample_name/15/example_hkli.nxs')

s = Scissors(data=data)

linecut = s.cut_data(center=(0,0,0), window=(0.1,0.5,0.1))

lm = LinecutModel(data=linecut)

lm.set_model_components([GaussianModel(prefix='peak'), LinearModel(prefix='background')])

lm.set_param_hint('peakcenter', min=-0.1, max=0.1)

lm.make_params()

lm.guess()

lm.print_initial_params()

lm.params

lm.plot_initial_guess()