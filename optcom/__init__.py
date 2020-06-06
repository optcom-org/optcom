__name__ = "optcom"
__version__ = "0.2.0"
# need Docstring doc

# from components
from optcom.components.cw import CW
from optcom.components.fiber import Fiber
from optcom.components.fiber_yb import FiberYb
from optcom.components.fiber_coupler import FiberCoupler
from optcom.components.gaussian import Gaussian
from optcom.components.gaussian_filter import GaussianFilter
from optcom.components.ideal_amplifier import IdealAmplifier
from optcom.components.ideal_combiner import IdealCombiner
from optcom.components.ideal_divider import IdealDivider
from optcom.components.ideal_coupler import IdealCoupler
from optcom.components.ideal_isolator import IdealIsolator
from optcom.components.ideal_mzm import IdealMZM
from optcom.components.ideal_phase_mod import IdealPhaseMod
from optcom.components.load_field_from_file import LoadFieldFromFile
from optcom.components.load_field import LoadField
from optcom.components.save_field_to_file import SaveFieldToFile
from optcom.components.save_field import SaveField
from optcom.components.sech import Sech
from optcom.components.soliton import Soliton

mod_comps = ['CW', 'Fiber', 'FiberYb', 'FiberCoupler', 'Gaussian',
             'GaussianFilter', 'IdealAmplifier', 'IdealCombiner',
             'IdealDivider', 'IdealCoupler', 'IdealIsolator',
             'IdealPhaseMod', 'IdealMZM', 'LoadFieldFromFile', 'LoadField',
             'SaveFieldToFile', 'SaveField', 'Sech', 'Soliton']


# from root
from optcom.domain import Domain
from optcom.layout import Layout

mod_root = ['Domain', 'Layout']


# from utils
from optcom.utils.utilities_user import temporal_power, spectral_power,\
                                        temporal_phase, spectral_phase,\
                                        energy, average_power, CSVFit, fwhm

mod_utils = ['CSVFit', 'temporal_power', 'spectral_power', 'temporal_phase',
             'spectral_phase', 'energy', 'average_power', 'fwhm']


# from staticmethod
from optcom.parameters.fiber.coupling_coeff import CouplingCoeff
from optcom.parameters.dispersion.chromatic_disp import ChromaticDisp
from optcom.parameters.fiber.v_number import VNumber
from optcom.parameters.refractive_index.sellmeier import Sellmeier
from optcom.parameters.fiber.numerical_aperture import NumericalAperture
calc_dispersion_length = ChromaticDisp.calc_dispersion_length
calc_kappa = CouplingCoeff.calc_kappa
calc_v_number = VNumber.calc_v_number
calc_NA = NumericalAperture.calc_NA
calc_n_core = NumericalAperture.calc_n_core
calc_n_clad = NumericalAperture.calc_n_clad

mod_static = ['calc_dispersion_length', 'calc_kappa', 'calc_v_number',
              'Sellmeier', 'calc_NA', 'calc_n_core', 'calc_n_clad']


#from plot
from optcom.utils.plot import plot2d, plot3d, animation2d

mod_plot = ['plot2d', 'plot3d', 'animation2d']


# from parameters
from optcom.parameters.fiber.absorption_section import AbsorptionSection
from optcom.parameters.fiber.emission_section import EmissionSection

mod_para = ['AbsorptionSection', 'EmissionSection']


# from config
from optcom.config import set_separator_terminal, get_separator_terminal,\
                          set_rk4ip_opti_gnlse, get_rk4ip_opti_gnlse,\
                          set_file_extension, get_file_extension,\
                          set_save_leaf_fields, get_save_leaf_fields,\
                          set_max_nbr_pass, get_max_nbr_pass,\
                          set_field_op_matching_omega,\
                          get_field_op_matching_omega,\
                          set_field_op_matching_rep_freq,\
                          get_field_op_matching_rep_freq

mod_config = ['set_separator_terminal', 'get_separator_terminal',
              'set_rk4ip_opti_gnlse', 'get_rk4ip_opti_gnlse',
              'set_file_extension', 'get_file_extension',
              'set_save_leaf_fields', 'get_save_leaf_fields',
              'set_max_nbr_pass', 'get_max_nbr_pass',
              'set_field_op_matching_omega', 'get_field_op_matching_omega',
              'set_field_op_matching_rep_freq',
              'get_field_op_matching_rep_freq']


__all__ = (mod_comps + mod_root + mod_utils + mod_static + mod_plot
           + mod_para + mod_config)
