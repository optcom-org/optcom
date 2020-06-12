# Copyright 2019 The Optcom Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""".. moduleauthor:: Sacha Medaer"""

from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.components.abstract_fiber_amp import AbstractFiberAmp
from optcom.components.abstract_fiber_amp import call_decorator
from optcom.domain import Domain
from optcom.equations.abstract_ampnlse import AbstractAmpNLSE
from optcom.equations.ampanlse import AmpANLSE
from optcom.equations.ampgnlse import AmpGNLSE
from optcom.equations.amphnlse import AmpHNLSE
from optcom.equations.boundary_conditions.boundary_conditions_ampnlse import\
    BoundaryConditionsAmpNLSE
from optcom.equations.convergence_checker.convergence_checker_consecutive \
    import ConvergenceCheckerConsecutive
from optcom.equations.re_fiber_yb import REFiberYb
from optcom.field import Field
from optcom.solvers.abstract_solver import AbstractSolver
from optcom.solvers.field_stepper import FieldStepper
from optcom.solvers.nlse_solver import NLSESolver
from optcom.solvers.ode_solver import ODESolver


TAYLOR_COEFF_TYPE_OPTIONAL = List[Union[List[float], Callable, None]]
FLOAT_COEFF_TYPE_OPTIONAL = List[Union[float, Callable, None]]


class AbstractFiberAmp2Levels(AbstractFiberAmp):
    r"""A non ideal Fiber Amplifier with 2-levels rate equations.

    Attributes
    ----------
    name : str
        The name of the component.
    ports_type : list of int
        Type of each port of the component, give also the number of
        ports in the component. For types, see
        :mod:`optcom/utils/constant_values/port_types`.
    save : bool
        If True, will save each field going through each port. The
        recorded fields can be accessed with the attribute
        :attr:`fields`.
    call_counter : int
        Count the number of times the function
        :func:`__call__` of the Component has been called.
    wait :
        If True, will wait for specified waiting port policy added
        with the function :func:`AbstractComponent.add_wait_policy`.
    pre_call_code :
        A string containing code which will be executed prior to
        the call to the function :func:`__call__`. The two parameters
        `input_ports` and `input_fields` are available.
    post_call_code :
        A string containing code which will be executed posterior to
        the call to the function :func:`__call__`. The two parameters
        `output_ports` and `output_fields` are available.
    REFL_SEED : bool
        If True, take into account the reflected seed waves for
        computation.
    REFL_PUMP : bool
        If True, take into account the reflected pump waves for
        computation.
    PROP_PUMP : bool
        If True, the pump is propagated forward in the layout.
    PROP_REFL : bool
        If True, the relfected fields are propagated in the layout
        as new fields.
    BISEED : bool
        If True, waiting policy waits for seed at both ends.
    BIPUMP : bool
        If True, waiting policy waits for pump at both ends.
    population_levels : numpy.ndarray of float
        The population density of each level. :math:`[nm^{-3}]`

    Notes
    -----
    Component diagram::

        [0] _____________________ [1]
            /                   \
           /                     \
        [2]                       [3]


    [0] and [1] : signal and [2] and [3] : pump

    """

    def __init__(self, name: str, default_name: str, length: float,
                 alpha: TAYLOR_COEFF_TYPE_OPTIONAL, alpha_order: int,
                 beta: TAYLOR_COEFF_TYPE_OPTIONAL, beta_order: int,
                 gain_order: int, gamma: FLOAT_COEFF_TYPE_OPTIONAL,
                 sigma: float, eta: float, T_R: float,
                 h_R: Optional[Union[float, Callable]], f_R: float,
                 n_core: FLOAT_COEFF_TYPE_OPTIONAL,
                 n_clad: FLOAT_COEFF_TYPE_OPTIONAL,
                 NA: FLOAT_COEFF_TYPE_OPTIONAL,
                 v_nbr: FLOAT_COEFF_TYPE_OPTIONAL,
                 eff_area: FLOAT_COEFF_TYPE_OPTIONAL,
                 nl_index: FLOAT_COEFF_TYPE_OPTIONAL,
                 overlap: FLOAT_COEFF_TYPE_OPTIONAL,
                 sigma_a: FLOAT_COEFF_TYPE_OPTIONAL,
                 sigma_e: FLOAT_COEFF_TYPE_OPTIONAL,
                 en_sat: FLOAT_COEFF_TYPE_OPTIONAL,
                 doped_area: Optional[float], tau: float, N_T: float,
                 core_radius: float, clad_radius: float,
                 R_0: Union[float, Callable], R_L: Union[float, Callable],
                 medium_core: str, medium_clad: str, temperature: float,
                 nl_approx: bool, RESO_INDEX: bool, CORE_PUMPED: bool,
                 CLAD_PUMPED: bool, ATT: List[bool], DISP: List[bool],
                 SPM: List[bool], XPM: List[bool], FWM: List[bool],
                 SS: List[bool], RS: List[bool], XNL: List[bool],
                 GAIN_SAT: bool, NOISE: bool, approx_type: int,
                 split_noise_option: str, noise_ode_method: str,
                 UNI_OMEGA: List[bool], STEP_UPDATE: bool,
                 INTRA_COMP_DELAY: bool, INTRA_PORT_DELAY: bool,
                 INTER_PORT_DELAY: bool, REFL_SEED: bool, REFL_PUMP: bool,
                 PRE_PUMP_PROP: bool, nlse_method: str, step_method: str,
                 steps: int, max_nbr_iter: int, error: float, PROP_PUMP: bool,
                 PROP_REFL: bool, BISEED: bool, BIPUMP: bool, save: bool,
                 save_all: bool, max_nbr_pass: Optional[List[int]],
                 pre_call_code: str, post_call_code: str) -> None:
        r"""
        Parameters
        ----------
        name :
            The name of the component.
        length :
            The length of the fiber. :math:`[km]`
        alpha :
            The derivatives of the attenuation coefficients.
            :math:`[km^{-1}, ps\cdot km^{-1}, ps^2\cdot km^{-1},
            ps^3\cdot km^{-1}, \ldots]` If a callable is provided,
            variable must be angular frequency. :math:`[ps^{-1}]`
        alpha_order :
            The order of alpha coefficients to take into account. (will
            be ignored if alpha values are provided - no file)
        beta :
            The derivatives of the propagation constant.
            :math:`[km^{-1}, ps\cdot km^{-1}, ps^2\cdot km^{-1},
            ps^3\cdot km^{-1}, \ldots]` If a callable is provided,
            variable must be angular frequency. :math:`[ps^{-1}]`
        beta_order :
            The order of beta coefficients to take into account. (will
            be ignored if beta values are provided - no file)
        gain_order :
            The order of the gain coefficients to take into account.
        gamma :
            The non linear coefficient.
            :math:`[rad\cdot W^{-1}\cdot km^{-1}]` If a callable is
            provided, variable must be angular frequency.
            :math:`[ps^{-1}]`
        sigma :
            Positive term multiplying the XPM terms of the NLSE.
        eta :
            Positive term multiplying the cross-non-linear terms of the
            NLSE.
        T_R :
            The raman coefficient. :math:`[]`
        h_R :
            The Raman response function values.  If a callable is
            provided, variable must be time. :math:`[ps]`
        f_R :
            The fractional contribution of the delayed Raman response.
            :math:`[]`
        n_core :
            The refractive index of the core.  If a callable
            is provided, variable must be angular frequency.
            :math:`[ps^{-1}]` (1<=len(n_core)<=2 for signal and pump)
        n_clad :
            The refractive index of the cladding.  If a callable
            is provided, variable must be angular frequency.
            :math:`[ps^{-1}]` (1<=len(n_clad)<=2 for signal and pump)
        NA :
            The numerical aperture. :math:`[]`  If a callable
            is provided, variable must be angular frequency.
            :math:`[ps^{-1}]` (1<=len(NA)<=2 for signal and pump)
        v_nbr :
            The V number. :math:`[]`  If a callable
            is provided, variable must be angular frequency.
            :math:`[ps^{-1}]` (1<=len(v_nbr)<=2 for signal and pump)
        eff_area :
            The effective area. :math:`[\u m^2]` If a callable
            is provided, variable must be angular frequency.
            :math:`[ps^{-1}]` (1<=len(eff_area)<=2 for signal and pump)
        nl_index :
            The non-linear coefficient. :math:`[m^2\cdot W^{-1}]`  If a
            callable is provided, variable must be angular frequency.
            :math:`[ps^{-1}]` (1<=len(nl_index)<=2 for signal and pump)
        overlap :
            The overlap factor. :math:`[]`  If a callable
            is provided, variable must be angular frequency.
            :math:`[ps^{-1}]` (1<=len(overlap)<=2 for signal and pump)
        sigma_a :
            The absorption cross sections. :math:`[nm^2]`  If a callable
            is provided, variable must be angular frequency.
            :math:`[ps^{-1}]` (1<=len(sigma_a)<=2 for signal and pump)
        sigma_e :
            The emission cross sections. :math:`[nm^2]`  If a callable
            is provided, variable must be angular frequency.
            :math:`[ps^{-1}]` (1<=len(sigma_e)<=2 for signal and pump)
        en_sat :
            The saturation energy. :math:`[J]`  If a callable
            is provided, variable must be angular frequency.
            :math:`[ps^{-1}]` (1<=len(en_sat)<=2 for signal and pump)
        doped_area :
            The doped area. :math:`[\mu m^2]`  If None, will be set to
            the core area.
        tau :
            The lifetime of the metastable level. :math:`[\mu s]`
        N_T :
            The total doping concentration. :math:`[nm^{-3}]`
        core_radius :
            The radius of the core. :math:`[\mu m]`
        clad_radius :
            The radius of the cladding. :math:`[\mu m]`
        R_0 :
            The reflectivity at the fiber start.  If a callable
            is provided, variable must be angular frequency.
            :math:`[ps^{-1}]`
        R_L :
            The reflectivity at the fiber end.  If a callable
            is provided, variable must be angular frequency.
            :math:`[ps^{-1}]`
        medium_core :
            The medium of the core.
        medium_clad :
            The medium of the cladding.
        temperature :
            The temperature of the fiber. :math:`[K]`
        nl_approx :
            If True, the approximation of the NLSE is used.
        RESO_INDEX :
            If True, trigger the resonant refractive index which will
            be added to the core refractive index. To see the effect of
            the resonant index, the flag STEP_UPDATE must be set to True
            in order to update the dispersion coefficient at each space
            step depending on the resonant index at each space step.
        CORE_PUMPED :
            If True, there is dopant in the core.
        CLAD_PUMPED :
            If True, there is dopant in the cladding.
        ATT :
            If True, trigger the attenuation. The first element is
            related to the seed and the second to the pump.
        DISP :
            If True, trigger the dispersion. The first element is
            related to the seed and the second to the pump.
        SPM :
            If True, trigger the self-phase modulation. The first
            element is related to the seed and the second to the pump.
        XPM :
            If True, trigger the cross-phase modulation. The first
            element is related to the seed and the second to the pump.
        FWM :
            If True, trigger the Four-Wave mixing. The first element is
            related to the seed and the second to the pump.
        SS :
            If True, trigger the self-steepening. The first element is
            related to the seed and the second to the pump.
        RS :
            If True, trigger the Raman scattering. The first element is
            related to the seed and the second to the pump.
        XNL :
            If True, trigger cross-non linear effects. The first element
            is related to the seed and the second to the pump.
        GAIN_SAT :
            If True, trigger the gain saturation.
        NOISE :
            If True, trigger the noise calculation.
        approx_type :
            The type of the NLSE approximation.
        split_noise_option :
            The way the spontaneous emission power is split among the
            fields.
        noise_ode_method :
            The ode solver method type for noise propagation
            computation.
        UNI_OMEGA :
            If True, consider only the center omega for computation.
            Otherwise, considered omega discretization.  The first
            element is related to the seed and the second to the pump.
        STEP_UPDATE :
            If True, update fiber parameters at each spatial sub-step.
        INTRA_COMP_DELAY :
            If True, take into account the relative time difference,
            between all waves, that is acquired while propagating
            in the component.
        INTRA_PORT_DELAY :
            If True, take into account the initial relative time
            difference between channels of all fields but for each port.
        INTER_PORT_DELAY :
            If True, take into account the initial relative time
            difference between channels of all fields of all ports.
        REFL_SEED :
            If True, take into account the reflected seed waves for
            computation.
        REFL_PUMP :
            If True, take into account the reflected pump waves for
            computation.
        PRE_PUMP_PROP :
            If True, propagate only the pump the first iteration and
            then add the seed at the second iteration.  Otherwise,
            consider both pump and seed simultaneously.  Results will be
            consistent but setting this flag to True can have a
            significant impact on the convergence speed.  Without
            knowledge of the algorithm, it is adviced to let it to
            False.
        nlse_method :
            The nlse solver method type.
        step_method :
            The method for spatial step size generation.
        steps :
            The number of steps for the solver
        max_nbr_iter :
            The maximum number of iterations if shooting method is
            employed.
        error :
            The error for convergence criterion of stepper resolution.
        PROP_PUMP :
            If True, the pump is propagated forward in the layout.
        PROP_REFL :
            If True, the relfected fields are propagated in the layout
            as new fields.
        BISEED :
            If True, waiting policy waits for seed at both ends.
        BIPUMP :
            If True, waiting policy waits for pump at both ends.
        save :
            If True, the last wave to enter/exit a port will be saved.
        save_all :
            If True, save the wave at each spatial step in the
            component.
        max_nbr_pass :
            No fields will be propagated if the number of
            fields which passed through a specific port exceed the
            specified maximum number of pass for this port.
        pre_call_code :
            A string containing code which will be executed prior to
            the call to the function :func:`__call__`. The two
            parameters `input_ports` and `input_fields` are available.
        post_call_code :
            A string containing code which will be executed posterior to
            the call to the function :func:`__call__`. The two
            parameters `output_ports` and `output_fields` are available.

        """
        # Parent constructor -------------------------------------------
        super().__init__(name, default_name, save, max_nbr_pass, pre_call_code,
                         post_call_code, REFL_SEED, REFL_PUMP, PROP_PUMP,
                         PROP_REFL, BISEED, BIPUMP)
        # Attr types check ---------------------------------------------
        util.check_attr_type(length, 'length', float)
        util.check_attr_type(alpha, 'alpha', None, Callable, float, List)
        util.check_attr_type(alpha_order, 'alpha_order', int)
        util.check_attr_type(beta, 'beta', None, Callable, float, list)
        util.check_attr_type(beta_order, 'beta_order', int)
        util.check_attr_type(gamma, 'gamma', None, float, Callable, list)
        util.check_attr_type(gain_order, 'gain_order', int)
        util.check_attr_type(sigma, 'sigma', float)
        util.check_attr_type(eta, 'eta', float)
        util.check_attr_type(T_R, 'T_R', float)
        util.check_attr_type(f_R, 'f_R', float)
        util.check_attr_type(nl_approx, 'nl_approx', bool)
        util.check_attr_type(n_core, 'n_core', None, float, list)
        util.check_attr_type(n_clad, 'n_clad', None, float, list)
        util.check_attr_type(NA, 'NA', None, float, Callable, list)
        util.check_attr_type(v_nbr, 'v_nbr', None, float, Callable, list)
        util.check_attr_type(eff_area, 'eff_area', None, float, Callable, list)
        util.check_attr_type(nl_index, 'nl_index', None, float, Callable, list)
        util.check_attr_type(overlap, 'overlap', None, float, Callable, list)
        util.check_attr_type(sigma_a, 'sigma_a', None, float, Callable, list)
        util.check_attr_type(sigma_e, 'sigma_e', None, float, Callable, list)
        util.check_attr_type(en_sat, 'en_sat', None, float, Callable, list)
        util.check_attr_type(tau, 'tau', float)
        util.check_attr_type(N_T, 'N_T', float)
        util.check_attr_type(core_radius, 'core_radius', float)
        util.check_attr_type(clad_radius, 'clad_radius', float)
        util.check_attr_type(R_0, 'R_0', float, Callable)
        util.check_attr_type(R_L, 'R_L', float, Callable)
        util.check_attr_type(medium_core, 'medium_core', str)
        util.check_attr_type(medium_clad, 'medium_clad', str)
        util.check_attr_type(temperature, 'temperature', float)
        util.check_attr_type(RESO_INDEX, 'RESO_INDEX', bool)
        util.check_attr_type(CORE_PUMPED, 'CORE_PUMPED', bool)
        util.check_attr_type(CLAD_PUMPED, 'CLAD_PUMPED', bool)
        util.check_attr_type(ATT, 'ATT', bool, list)
        util.check_attr_type(DISP, 'DISP', bool, list)
        util.check_attr_type(SPM, 'SPM', bool, list)
        util.check_attr_type(XPM, 'XPM', bool, list)
        util.check_attr_type(FWM, 'FWM', bool, list)
        util.check_attr_type(SS, 'SS', bool, list)
        util.check_attr_type(RS, 'RS', bool, list)
        util.check_attr_type(XNL, 'XNL', bool, list)
        util.check_attr_type(GAIN_SAT, 'GAIN_SAT', bool)
        util.check_attr_type(NOISE, 'NOISE', bool)
        util.check_attr_type(approx_type, 'approx_type', int)
        util.check_attr_type(noise_ode_method, 'noise_ode_method', str)
        util.check_attr_type(UNI_OMEGA, 'UNI_OMEGA', bool, list)
        util.check_attr_type(STEP_UPDATE, 'STEP_UPDATE', bool)
        util.check_attr_type(INTRA_COMP_DELAY, 'INTRA_COMP_DELAY', bool)
        util.check_attr_type(INTRA_PORT_DELAY, 'INTRA_PORT_DELAY', bool)
        util.check_attr_type(INTER_PORT_DELAY, 'INTER_PORT_DELAY', bool)
        util.check_attr_type(PRE_PUMP_PROP, 'PRE_PUMP_PROP', bool)
        util.check_attr_type(nlse_method, 'nlse_method', str)
        util.check_attr_type(step_method, 'step_method', str)
        util.check_attr_type(steps, 'steps', int)
        util.check_attr_type(max_nbr_iter, 'max_nbr_iter', int)
        util.check_attr_type(error, 'error', float)
        # Attr ---------------------------------------------------------
        # Component equations ------------------------------------------
        re_fiber: REFiberYb
        re_fiber = REFiberYb(N_T, tau, doped_area, n_core, n_clad, NA, v_nbr,
                             eff_area, overlap, sigma_a, sigma_e, core_radius,
                             clad_radius, medium_core, medium_clad,
                             temperature, RESO_INDEX, CORE_PUMPED, CLAD_PUMPED,
                             NOISE, UNI_OMEGA, STEP_UPDATE)
        nlse: AbstractAmpNLSE
        if (nl_approx or (not sum(RS) and not sum(SS))):
            nlse = AmpANLSE(re_fiber, alpha, alpha_order, beta, beta_order,
                            gain_order, gamma, sigma, eta, T_R, n_core, n_clad,
                            NA, v_nbr, eff_area, nl_index, overlap, sigma_a,
                            sigma_e, en_sat, R_0, R_L, core_radius,
                            clad_radius, medium_core, medium_clad, temperature,
                            ATT, DISP, SPM, XPM, FWM, SS, RS, XNL, GAIN_SAT,
                            NOISE, approx_type, split_noise_option, UNI_OMEGA,
                            STEP_UPDATE, INTRA_COMP_DELAY, INTRA_PORT_DELAY,
                            INTER_PORT_DELAY)
        else:
            if (sum(SS) == 2):
                nlse = AmpGNLSE(re_fiber, alpha, alpha_order, beta, beta_order,
                                gain_order, gamma, sigma, eta, h_R, f_R,
                                n_core, n_clad, NA, v_nbr, eff_area, nl_index,
                                overlap, sigma_a, sigma_e, en_sat, R_0, R_L,
                                core_radius, clad_radius, medium_core,
                                medium_clad, temperature, ATT, DISP, SPM, XPM,
                                FWM, XNL, GAIN_SAT, NOISE, split_noise_option,
                                UNI_OMEGA, STEP_UPDATE, INTRA_COMP_DELAY,
                                INTRA_PORT_DELAY, INTER_PORT_DELAY)
            else:
                nlse = AmpHNLSE(re_fiber, alpha, alpha_order, beta, beta_order,
                                gain_order, gamma, sigma, eta, h_R, f_R,
                                n_core, n_clad, NA, v_nbr, eff_area, nl_index,
                                overlap, sigma_a, sigma_e, en_sat, R_0, R_L,
                                core_radius, clad_radius, medium_core,
                                medium_clad, temperature, ATT, DISP, SPM, XPM,
                                FWM, SS, XNL, GAIN_SAT, NOISE,
                                split_noise_option, UNI_OMEGA, STEP_UPDATE,
                                INTRA_COMP_DELAY, INTRA_PORT_DELAY,
                                INTER_PORT_DELAY)
        # Component stepper --------------------------------------------
        stepper_method: List[str] = ['shooting', 'shooting']
        solver_order: str = 'alternating'
        # if method is None, will directly call the equation object
        solvers: List[AbstractSolver]
        solvers = [NLSESolver(re_fiber, None), NLSESolver(nlse, nlse_method)]
        noise_solvers: List[Optional[AbstractSolver]]
        noise_solvers = [ODESolver(re_fiber.calc_noise, None),
                         ODESolver(nlse.calc_noise, noise_ode_method)]
        conv_checker: ConvergenceCheckerConsecutive
        conv_checker = ConvergenceCheckerConsecutive(error, max_nbr_iter,
                                                     False)
        boundary_cond: BoundaryConditionsAmpNLSE
        boundary_cond = BoundaryConditionsAmpNLSE(nlse, REFL_SEED, REFL_PUMP,
                                                  PRE_PUMP_PROP)
        self._stepper: FieldStepper
        self._stepper = FieldStepper(solvers, noise_solvers, length, [steps],
                                     [step_method], solver_order,
                                     stepper_method,
                                     boundary_cond=boundary_cond,
                                     conv_checker=conv_checker,
                                     save_all=save_all)
        self.population_levels: np.ndarray = np.array([])
        self._population_levels_getter = re_fiber.get_population_levels
    # ==================================================================
    @call_decorator
    def __call__(self, domain: Domain, ports: List[int], fields: List[Field]
                 ) -> Tuple[List[int], List[Field]]:
        output_ports: List[int] = []
        output_fields: List[Field] = []
        fields_per_eq: List[List[Field]] = [[] for i in range(4)]
        # Sort fields and ports ----------------------------------------
        for i in range(len(ports)):
            fields_per_eq[ports[i]].append(fields[i])
        output_ports = sorted(ports)
        # Compute ------------------------------------------------------
        # Insert a copy of each field list as the reflected fields
        len_fields_per_eq: int = len(fields_per_eq)
        for i in range(len_fields_per_eq):
            fields_to_add: List[Field] = []
            for field in fields_per_eq[i]:
                new_name = ('reflection_of_' + field.name + '_from_'
                            + self.name)
                fields_to_add.append(field.get_copy(new_name, True, True,
                                                    False))
            fields_per_eq.append(fields_to_add)
        output_fields = self._stepper(domain, *fields_per_eq)
        # Save to storage ----------------------------------------------
        if (self._stepper.save_all):
            self.storages.append(self._stepper.storage)
        # Record population densities ----------------------------------
        self.population_levels = self._population_levels_getter()

        return self.output_ports(output_ports), output_fields
