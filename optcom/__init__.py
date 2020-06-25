__name__ = "optcom"
__version__ = "0.3.2"
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
from optcom.components.ideal_coupler import IdealCoupler
from optcom.components.ideal_divider import IdealDivider
from optcom.components.ideal_filter import IdealFilter
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
             'IdealCoupler', 'IdealDivider', 'IdealFilter', 'IdealIsolator',
             'IdealPhaseMod', 'IdealMZM', 'LoadFieldFromFile', 'LoadField',
             'SaveFieldToFile', 'SaveField', 'Sech', 'Soliton']


# from root
from optcom.domain import Domain
from optcom.layout import Layout
from optcom.field import Field

mod_root = ['Domain', 'Layout', 'Field']


# from utils
import typing
import optcom.field as fld
temporal_power: typing.Callable = fld.Field.temporal_power
spectral_power: typing.Callable = fld.Field.spectral_power
temporal_phase: typing.Callable = fld.Field.temporal_phase
spectral_phase: typing.Callable = fld.Field.temporal_phase
temporal_peak_power: typing.Callable = fld.Field.temporal_peak_power
spectral_peak_power: typing.Callable = fld.Field.spectral_peak_power
energy: typing.Callable = fld.Field.energy
average_power: typing.Callable = fld.Field.average_power
fwhm: typing.Callable = fld.Field.fwhm
import optcom.domain as dmn
nu_to_omega: typing.Callable = dmn.Domain.nu_to_omega
nu_to_lambda: typing.Callable = dmn.Domain.nu_to_lambda
omega_to_nu: typing.Callable = dmn.Domain.omega_to_nu
omega_to_lambda: typing.Callable = dmn.Domain.omega_to_lambda
lambda_to_nu: typing.Callable = dmn.Domain.lambda_to_nu
lambda_to_omega: typing.Callable = dmn.Domain.lambda_to_omega
nu_bw_to_lambda_bw: typing.Callable = dmn.Domain.nu_bw_to_lambda_bw
lambda_bw_to_nu_bw: typing.Callable = dmn.Domain.lambda_bw_to_nu_bw
omega_bw_to_lambda_bw: typing.Callable = dmn.Domain.omega_bw_to_lambda_bw
lambda_bw_to_omega_bw: typing.Callable = dmn.Domain.lambda_bw_to_omega_bw
from optcom.utils.csv_fit import CSVFit
from optcom.utils.storage import Storage
from optcom.utils.utilities import db_to_linear, linear_to_db

mod_utils = ['CSVFit', 'temporal_power', 'spectral_power', 'temporal_phase',
             'spectral_phase', 'energy', 'average_power', 'fwhm',
             'temporal_peak_power', 'spectral_peak_power', 'db_to_linear',
             'linear_to_db', 'Storage', 'nu_to_omega', 'nu_to_lambda',
             'omega_to_nu', 'omega_to_lambda', 'lambda_to_nu',
             'lambda_to_omega', 'nu_bw_to_lambda_bw', 'lambda_bw_to_nu_bw',
             'lambda_bw_to_omega_bw', 'omega_bw_to_lambda_bw']


#from plot
from optcom.utils.plot import plot2d, plot3d, animation2d

mod_plot = ['plot2d', 'plot3d', 'animation2d']


# from parameters
from optcom.parameters.dispersion.chromatic_disp import ChromaticDisp
from optcom.parameters.fiber.absorption_section import AbsorptionSection
from optcom.parameters.fiber.asymmetry_coeff import AsymmetryCoeff
from optcom.parameters.fiber.coupling_coeff import CouplingCoeff
from optcom.parameters.fiber.doped_fiber_gain import DopedFiberGain
from optcom.parameters.fiber.effective_area import EffectiveArea
from optcom.parameters.fiber.emission_section import EmissionSection
from optcom.parameters.fiber.energy_saturation import EnergySaturation
from optcom.parameters.fiber.fiber_recovery_time import FiberRecoveryTime
from optcom.parameters.fiber.nl_coefficient import NLCoefficient
from optcom.parameters.fiber.nl_phase_shift import NLPhaseShift
from optcom.parameters.fiber.numerical_aperture import NumericalAperture
from optcom.parameters.fiber.overlap_factor import OverlapFactor
from optcom.parameters.fiber.raman_response import RamanResponse
from optcom.parameters.fiber.se_power import SEPower
from optcom.parameters.fiber.v_number import VNumber
from optcom.parameters.refractive_index.nl_index import NLIndex
from optcom.parameters.refractive_index.resonant_index import ResonantIndex
from optcom.parameters.refractive_index.sellmeier import Sellmeier


mod_para = ['AbsorptionSection', 'ChromaticDisp', 'AsymmetryCoeff',
            'CouplingCoeff', 'DopedFiberGain', 'EffectiveArea',
            'EmissionSection', 'EnergySaturation', 'FiberRecoveryTime',
            'NLCoefficient', 'NLPhaseShift', 'NumericalAperture',
            'OverlapFactor', 'RamanResponse', 'SEPower', 'VNumber', 'NLIndex',
            'ResonantIndex', 'Sellmeier']



# from constants
from optcom.utils.constant_values.physic_cst import C
from optcom.utils.constant_values.physic_cst import PI
from optcom.utils.constant_values.physic_cst import KB
from optcom.utils.constant_values.physic_cst import H
from optcom.utils.constant_values.physic_cst import HBAR
from optcom.utils.constant_values.physic_cst import M_E
from optcom.utils.constant_values.physic_cst import C_E
from optcom.utils.constant_values.physic_cst import EPS_0

mod_cst = ['C', 'PI', 'KB', 'H', 'HBAR', 'M_E', 'C_E', 'EPS_0']


# from config
from optcom.config import set_log_filename, get_log_filename,\
                          set_print_log, get_print_log,\
                          set_separator_terminal, get_separator_terminal,\
                          set_rk4ip_opti_gnlse, get_rk4ip_opti_gnlse,\
                          set_file_extension, get_file_extension,\
                          set_save_leaf_fields, get_save_leaf_fields,\
                          set_max_nbr_pass, get_max_nbr_pass,\
                          set_field_op_matching_omega,\
                          get_field_op_matching_omega,\
                          set_field_op_matching_rep_freq,\
                          get_field_op_matching_rep_freq,\
                          set_multiprocessing, get_multiprocessing

mod_config = ['set_separator_terminal', 'get_separator_terminal',
              'set_rk4ip_opti_gnlse', 'get_rk4ip_opti_gnlse',
              'set_file_extension', 'get_file_extension',
              'set_save_leaf_fields', 'get_save_leaf_fields',
              'set_max_nbr_pass', 'get_max_nbr_pass',
              'set_field_op_matching_omega', 'get_field_op_matching_omega',
              'set_field_op_matching_rep_freq',
              'get_field_op_matching_rep_freq',
              'set_multiprocessing', 'get_multiprocessing']


__all__ = (mod_comps + mod_root + mod_utils + mod_plot
           + mod_para + mod_config + mod_cst)
