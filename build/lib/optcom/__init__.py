__name__ = "optcom"
__version__ = "1.0.0"
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


#from plot
from optcom.utils.plot import plot

mod_plot = ['plot']



__all__ = mod_comps + mod_root + mod_utils + mod_plot
