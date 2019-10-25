__name__ = "optcom"
__version__ = "0.1.0"
# need Docstring doc

# from components
from optcom.components.cw import CW
from optcom.components.fiber import Fiber
from optcom.components.fiber_amplifier import FiberAmplifier
from optcom.components.fiber_coupler import FiberCoupler
from optcom.components.gaussian import Gaussian
from optcom.components.ideal_amplifier import IdealAmplifier
from optcom.components.ideal_combiner import IdealCombiner
from optcom.components.ideal_divider import IdealDivider
from optcom.components.ideal_fiber_coupler import IdealFiberCoupler
from optcom.components.ideal_phase_mod import IdealPhaseMod
from optcom.components.ideal_mz import IdealMZ
from optcom.components.sech import Sech
from optcom.components.soliton import Soliton

mod_comps = ['CW', 'Fiber', 'FiberAmplifier', 'FiberCoupler', 'Gaussian',
             'IdealAmplifier', 'IdealCombiner', 'IdealDivider',
             'IdealFiberCoupler', 'IdealPhaseMod', 'IdealMZ', 'Sech',
             'Soliton']

#from root
from optcom.domain import Domain
from optcom.layout import Layout

mod_root = ['Domain', 'Layout']


#from utils
from optcom.utils.utilities_user import temporal_power, spectral_power, phase,\
    CSVFit

mod_utils = ['CSVFit', 'temporal_power', 'spectral_power', 'phase']

#from staticmethod
from optcom.effects.coupling import Coupling
from optcom.effects.dispersion import Dispersion
calc_dispersion_length = Dispersion.calc_dispersion_length
calc_kappa = Coupling.calc_kappa

mod_static = ['calc_dispersion_length', 'calc_kappa']


#from plot
from optcom.utils.plot import plot2d, plot3d

mod_plot = ['plot2d', 'plot3d']



__all__ = mod_comps + mod_root + mod_utils + mod_static + mod_plot
