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

import math

import numpy as np
from typing import Callable, List, Optional, Sequence, Tuple, Union

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.components.abstract_pass_comp import AbstractPassComp
from optcom.components.abstract_pass_comp import call_decorator
from optcom.domain import Domain
from optcom.equations.abstract_cnlse import AbstractCNLSE
from optcom.equations.canlse import CANLSE
from optcom.equations.cgnlse import CGNLSE
from optcom.equations.cnlse import CNLSE
from optcom.field import Field
from optcom.solvers.field_stepper import FieldStepper
from optcom.solvers.abstract_solver import AbstractSolver
from optcom.solvers.nlse_solver import NLSESolver
from optcom.solvers.ode_solver import ODESolver


default_name = 'Fiber Coupler'
TAYLOR_COEFF_TYPE_OPTIONAL = List[Union[List[float], Callable, None]]
FLOAT_COEFF_TYPE_OPTIONAL = List[Union[float, Callable, None]]
TAYLOR_COUP_COEFF_OPTIONAL = List[List[Union[List[float], Callable, None]]]


class FiberCoupler(AbstractPassComp):
    r"""A non ideal fiber coupler.

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

    Notes
    -----
    Component diagram::

        [0] _______        ______ [2]
                   \______/
        [1] _______/      \______ [3]

    """

    _nbr_instances: int = 0
    _nbr_instances_with_default_name: int = 0

    def __init__(self, name: str = default_name, length: float = 1.0,
                 nbr_fibers: int = 2,
                 alpha: TAYLOR_COEFF_TYPE_OPTIONAL = [None],
                 alpha_order: int = 0,
                 beta: TAYLOR_COEFF_TYPE_OPTIONAL = [None],
                 beta_order: int = 2,
                 gamma: FLOAT_COEFF_TYPE_OPTIONAL = [None],
                 kappa: TAYLOR_COUP_COEFF_OPTIONAL = [[None]],
                 sigma: List[float] = [cst.XPM_COEFF],
                 sigma_cross: List[List[float]] = [[cst.XPM_COEFF_CROSS]],
                 eta: List[float] = [cst.XNL_COEFF],
                 eta_cross: List[List[float]] = [[cst.XNL_COEFF_CROSS]],
                 T_R: List[float] = [cst.RAMAN_COEFF],
                 h_R: FLOAT_COEFF_TYPE_OPTIONAL = [None], f_R: float = cst.F_R,
                 core_radius: List[float] = [cst.CORE_RADIUS],
                 clad_radius: float = cst.CLAD_RADIUS_COUP,
                 c2c_spacing: List[List[float]] = [[cst.C2C_SPACING]],
                 n_core: FLOAT_COEFF_TYPE_OPTIONAL = [None],
                 n_clad: Optional[Union[float, Callable]] = None,
                 NA: FLOAT_COEFF_TYPE_OPTIONAL = [None],
                 v_nbr: FLOAT_COEFF_TYPE_OPTIONAL = [None],
                 eff_area: FLOAT_COEFF_TYPE_OPTIONAL = [None],
                 nl_index: FLOAT_COEFF_TYPE_OPTIONAL = [None],
                 nl_approx: bool = True,
                 medium_core: List[str] = [cst.FIBER_MEDIUM_CORE],
                 medium_clad: str = cst.FIBER_MEDIUM_CLAD,
                 temperature: float = cst.TEMPERATURE, ATT: bool = True,
                 DISP: bool = True, SPM: bool = True, XPM: bool = False,
                 FWM: bool = False, SS: bool = False, RS: bool = False,
                 XNL: bool = False, ASYM: bool = True, COUP: bool = True,
                 NOISE: bool = True,
                 approx_type: int = cst.DEFAULT_APPROX_TYPE,
                 noise_ode_method: str = 'rk4', UNI_OMEGA: bool = True,
                 STEP_UPDATE: bool = False, INTRA_COMP_DELAY: bool = True,
                 INTRA_PORT_DELAY: bool = True, INTER_PORT_DELAY: bool = False,
                 nlse_method: str = "rk4ip", ode_method: str = "euler",
                 step_method: str = "fixed", steps: int = 100,
                 save: bool = False, save_all: bool = False,
                 wait: bool = False, max_nbr_pass: Optional[List[int]] = None,
                 pre_call_code: str = '', post_call_code: str = '') -> None:
        r"""
        Parameters
        ----------
        name :
            The name of the component.
        length :
            The length of the coupler. :math:`[km]`
        nbr_fibers :
            The number of fibers in the coupler.
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
        gamma :
            The non linear coefficient.
            :math:`[rad\cdot W^{-1}\cdot km^{-1}]` If a callable is
            provided, variable must be angular frequency.
            :math:`[ps^{-1}]`
        kappa :
            The coupling coefficients. :math:`[km^{-1}]`
        sigma :
            Positive term multiplying the XPM terms of the NLSE.
        sigma_cross :
            Positive term multiplying the XPM terms of the NLSE
            inbetween the fibers.
        eta :
            Positive term multiplying the cross-non-linear terms of the
            NLSE.
        eta_cross :
            Positive term multiplying the cross-non-linear terms of the
            NLSE inbetween the fibers.
        T_R :
            The raman coefficient. :math:`[]`
        h_R :
            The Raman response function values.  If a callable is
            provided, variable must be time. :math:`[ps]`
        f_R :
            The fractional contribution of the delayed Raman response.
            :math:`[]`
        core_radius :
            The core radius. :math:`[\mu m]`
        clad_radius :
            The radius of the cladding. :math:`[\mu m]`
        c2c_spacing :
            The center to center distance between two cores.
            :math:`[\mu m]`
        n_core :
            The refractive index of the core.  If a callable is
            provided, variable must be angular frequency.
            :math:`[ps^{-1}]`
        n_clad :
            The refractive index of the clading.  If a callable is
            provided, variable must be angular frequency.
            :math:`[ps^{-1}]`
        NA :
            The numerical aperture.  If a callable is provided, variable
            must be angular frequency. :math:`[ps^{-1}]`
        v_nbr :
            The V number.  If a callable is provided, variable must be
            angular frequency. :math:`[ps^{-1}]`
        eff_area :
            The effective area.  If a callable is provided, variable
            must be angular frequency. :math:`[ps^{-1}]`
        nl_index :
            The non-linear coefficient.  If a callable is provided,
            variable must be angular frequency. :math:`[ps^{-1}]`
        nl_approx :
            If True, the approximation of the NLSE is used.
        medium_core :
            The medium of the fiber core.
        medium_clad :
            The medium of the fiber cladding.
        temperature :
            The temperature of the fiber. :math:`[K]`
        ATT :
            If True, trigger the attenuation.
        DISP :
            If True, trigger the dispersion.
        SPM :
            If True, trigger the self-phase modulation.
        XPM :
            If True, trigger the cross-phase modulation.
        FWM :
            If True, trigger the Four-Wave mixing.
        SS : bool
            If True, trigger the self-steepening.
        RS :
            If True, trigger the Raman scattering.
        XNL :
            If True, trigger cross-non linear effects.
        ASYM :
            If True, trigger the asymmetry effects between cores.
        COUP :
            If True, trigger the coupling effects between cores.
        NOISE :
            If True, trigger the noise calculation.
        approx_type :
            The type of the NLSE approximation.
        noise_ode_method :
            The ode solver method type for noise propagation
            computation.
        UNI_OMEGA :
            If True, consider only the center omega for computation.
            Otherwise, considered omega discretization.
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
        nlse_method :
            The nlse solver method type.
        ode_method :
            The ode solver method type.
        step_method :
            The method for spatial step size generation.
        steps :
            The number of steps for the solver
        save :
            If True, the last wave to enter/exit a port will be saved.
        save_all :
            If True, save the wave at each spatial step in the
            component.
        wait :
            If True, wait for another pulse in the anolog port
            [0 <-> 1, and 2 <-> 3] to launch the simulation.
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
        ports_type = [cst.OPTI_ALL for i in range(4)]
        super().__init__(name, default_name, ports_type, save, wait=wait,
                         max_nbr_pass=max_nbr_pass,
                         pre_call_code=pre_call_code,
                         post_call_code=post_call_code)
        # Attr types check ---------------------------------------------
        util.check_attr_type(length, 'length', float)
        util.check_attr_type(nbr_fibers, 'nbr_fibers', int)
        util.check_attr_type(alpha, 'alpha', None, Callable, float, List)
        util.check_attr_type(alpha_order, 'alpha_order', int)
        util.check_attr_type(beta, 'beta', None, Callable, float, list)
        util.check_attr_type(beta_order, 'beta_order', int)
        util.check_attr_type(gamma, 'gamma', None, float, list, Callable)
        util.check_attr_type(kappa, 'kappa', None, float, list)
        util.check_attr_type(sigma, 'sigma', float, list)
        util.check_attr_type(sigma_cross, 'sigma_cross', float, list)
        util.check_attr_type(eta, 'eta', float, list)
        util.check_attr_type(eta_cross, 'eta_cross', float, list)
        util.check_attr_type(T_R, 'T_R', float, list)
        util.check_attr_type(f_R, 'f_R', float)
        util.check_attr_type(core_radius, 'core_radius', float, list)
        util.check_attr_type(clad_radius, 'clad_radius', float)
        util.check_attr_type(c2c_spacing, 'c2c_spacing', float, list)
        util.check_attr_type(n_core, 'n_core', None, float, Callable, list)
        util.check_attr_type(n_clad, 'n_clad', None, float, Callable, list)
        util.check_attr_type(NA, 'NA', None, float, Callable, list)
        util.check_attr_type(v_nbr, 'v_nbr', None, float, Callable, list)
        util.check_attr_type(eff_area, 'eff_area', None, float, Callable, list)
        util.check_attr_type(nl_index, 'nl_index', None, float, Callable, list)
        util.check_attr_type(medium_core, 'medium_core', str, list)
        util.check_attr_type(medium_clad, 'medium_clad', str)
        util.check_attr_type(temperature, 'temperature', float)
        util.check_attr_type(nl_approx, 'nl_approx', bool)
        util.check_attr_type(ATT, 'ATT', bool)
        util.check_attr_type(DISP, 'DISP', bool)
        util.check_attr_type(SPM, 'SPM', bool)
        util.check_attr_type(XPM, 'XPM', bool)
        util.check_attr_type(FWM, 'FWM', bool)
        util.check_attr_type(SS, 'SS', bool)
        util.check_attr_type(RS, 'RS', bool)
        util.check_attr_type(XNL, 'XNL', bool)
        util.check_attr_type(ASYM, 'ASYM', bool)
        util.check_attr_type(COUP, 'COUP', bool)
        util.check_attr_type(NOISE, 'NOISE', bool)
        util.check_attr_type(approx_type, 'approx_type', int)
        util.check_attr_type(noise_ode_method, 'noise_ode_method', str)
        util.check_attr_type(UNI_OMEGA, 'UNI_OMEGA', bool)
        util.check_attr_type(STEP_UPDATE, 'STEP_UPDATE', bool)
        util.check_attr_type(INTRA_COMP_DELAY, 'INTRA_COMP_DELAY', bool)
        util.check_attr_type(INTRA_PORT_DELAY, 'INTRA_PORT_DELAY', bool)
        util.check_attr_type(INTER_PORT_DELAY, 'INTER_PORT_DELAY', bool)
        util.check_attr_type(nlse_method, 'nlse_method', str)
        util.check_attr_type(ode_method, 'ode_method', str)
        util.check_attr_type(step_method, 'step_method', str)
        util.check_attr_type(steps, 'steps', int)
        # Attr ---------------------------------------------------------
        self._NOISE = NOISE
        cnlse: AbstractCNLSE
        if (nl_approx or (not RS and not SS)):
            cnlse = CANLSE(nbr_fibers, alpha, alpha_order, beta, beta_order,
                           gamma, kappa, sigma, sigma_cross, eta, eta_cross,
                           T_R, core_radius, clad_radius, c2c_spacing, n_core,
                           n_clad, NA, v_nbr, eff_area, nl_index, medium_core,
                           medium_clad, temperature, ATT, DISP, SPM, XPM, FWM,
                           SS, RS, XNL, ASYM, COUP, NOISE, approx_type,
                           UNI_OMEGA, STEP_UPDATE, INTRA_COMP_DELAY,
                           INTRA_PORT_DELAY, INTER_PORT_DELAY)
        else:
            if (SS):
                cnlse = CGNLSE(nbr_fibers, alpha, alpha_order, beta,
                               beta_order, gamma, kappa, sigma, sigma_cross,
                               eta, eta_cross, h_R, f_R, core_radius,
                               clad_radius, c2c_spacing, n_core, n_clad, NA,
                               v_nbr, eff_area, nl_index, medium_core,
                               medium_clad, temperature, ATT, DISP, SPM, XPM,
                               FWM, XNL, ASYM, COUP, NOISE, UNI_OMEGA,
                               STEP_UPDATE, INTRA_COMP_DELAY, INTRA_PORT_DELAY,
                               INTER_PORT_DELAY)
            else:
                cnlse = CNLSE(nbr_fibers, alpha, alpha_order, beta,
                              beta_order, gamma, kappa, sigma, sigma_cross,
                              eta, eta_cross, h_R, f_R, core_radius,
                              clad_radius, c2c_spacing, n_core, n_clad, NA,
                              v_nbr, eff_area, nl_index, medium_core,
                              medium_clad, temperature, ATT, DISP, SPM, XPM,
                              FWM, XNL, ASYM, COUP, NOISE, UNI_OMEGA,
                              STEP_UPDATE, INTRA_COMP_DELAY, INTRA_PORT_DELAY,
                              INTER_PORT_DELAY)
        solvers: List[AbstractSolver]
        solvers = [NLSESolver(cnlse, nlse_method),
                   ODESolver(cnlse, ode_method)]
        noise_solvers: List[Optional[AbstractSolver]]
        noise_solvers = [ODESolver(cnlse.calc_noise), None]
        solver_order: str = "alternating"
        self._stepper = FieldStepper(solvers, noise_solvers, length,
                                     [steps, steps], [step_method],
                                     solver_order, save_all=save_all)
        # Policy -------------------------------------------------------
        self.add_port_policy(([0, 1], [2, 3], True))
        self.add_wait_policy([0, 1], [2, 3])
        # temp
        self._get_kappa_for_noise = cnlse.get_kappa_for_noise
    # ==================================================================
    def output_ports(self, input_ports: List[int]) -> List[int]:
        # 1 input ------------------------------------------------------
        uni_ports = util.unique(input_ports)
        if (len(uni_ports) == 1):
            # 0 -> 1 /\ 1 -> 0 /\ 2 -> 3 /\ 3 -> 2
            analog_port = uni_ports[0] ^ 1
            input_ports.append(analog_port)

        return super().output_ports(input_ports)
    # ==================================================================
    @call_decorator
    def __call__(self, domain: Domain, ports: List[int], fields: List[Field]
                 ) -> Tuple[List[int], List[Field]]:

        output_fields: List[Field] = []
        null_fields: List[Field] = []
        uni_ports = util.unique(ports)
        one_port_input = (len(uni_ports) == 1)
        # 1 input port - len(uni_ports) == 1 ---------------------------
        if (one_port_input):
            new_name: str = fields[0].name + "_copy_from_" + self.name
            null_field = fields[0].get_copy(new_name, True, True, False)
            null_fields.append(null_field)
        # 2 input ports - len(uni_ports) == 2 --------------------------
        fields_port_0: List[Field] = []
        fields_port_1: List[Field] = []
        if (one_port_input):    # Add corresponding null_fields in corr. port
            fields_port_0 = fields
            fields_port_1 = null_fields
        else:                   # Distribute each field depending on port
            for i in range(len(ports)):
                if (ports[i] == uni_ports[0]):
                    fields_port_0.append(fields[i])
                else:
                    fields_port_1.append(fields[i])
        output_fields = self._stepper(domain, fields_port_0, fields_port_1)
        # Noise management
        if (self._NOISE):
            # 1. splitting assumption
            #noise = np.zeros(domain.noise_samples)
            #for i in range(len(ports)):
            #    noise += fields[i].noise
            #noise /= len(ports)
            #for i in range(len(ports)):
            #    fields[i].noise = noise
            # 2. Anal sol, only bi-fiber coupler config supported for now
            # and make assumptions of a symmetric coupler
            # k_{12} = k_{21}.  See Agrawal applications top page 61 for
            # analytical solution
            length = self._stepper._length
            kappa = self._get_kappa_for_noise()
            factor = np.power(np.cos(kappa*length), 2)
            factor_ = np.power(np.sin(kappa*length), 2)
            accu_noise_port_0 = np.zeros(domain.noise_samples)
            accu_noise_port_1 = np.zeros(domain.noise_samples)
            # Solution of for ecah considered alone
            for i, field in enumerate(fields_port_0):   # len == 2
                accu_noise_port_0 += factor_ * field.noise
                field.noise *= factor
            for i, field in enumerate(fields_port_1):   # len == 2
                accu_noise_port_1 += factor_ * field.noise
                field.noise *= factor
            # Redistributing the noise that switched in the other fiber
            for field in fields_port_0:
                field.noise += accu_noise_port_1 / len(fields_port_0)
            for field in fields_port_1:
                field.noise += accu_noise_port_0 / len(fields_port_1)

        # Get storage
        if (self._stepper.save_all):
            self.storages.append(self._stepper.storage)

        return self.output_ports(ports), output_fields


if __name__ == "__main__":
    """Give an example of FiberCoupler usage.
    This piece of code is standalone, i.e. can be used in a separate
    file as an example.
    """

    import math
    from typing import Callable, List, Optional, Union

    import numpy as np

    import optcom as oc

    noise_samples: int = 100
    lt: oc.Layout = oc.Layout(oc.Domain(bit_width=20.0, samples_per_bit=1024,
                                        noise_samples=noise_samples))

    Lambda: float = 1030.0
    pulse_1: oc.Gaussian = oc.Gaussian(channels=1, peak_power=[38.5, 0.5],
                                       fwhm=[1.], center_lambda=[Lambda],
                                       noise=np.ones(noise_samples)*12)
    pulse_2: oc.Gaussian = oc.Gaussian(channels=1, peak_power=[23.5, 0.3],
                                       fwhm=[1.], center_lambda=[1050.0],
                                       noise=np.ones(noise_samples)*5)

    steps: int = int(100)
    alpha: List[Union[List[float], Callable, None]] = [[0.046], [0.046]]
    beta_01: float = 1e5
    beta_02: float = 1e5
    beta: List[Union[List[float], Callable, None]] =\
        [[beta_01,10.0,-0.0],[beta_02,10.0,-0.0]]
    gamma: List[Union[float, Callable, None]] = [4.3, 4.3]
    fitting_kappa: bool = False
    v_nbr_value = 2.0
    v_nbr: List[Union[float, Callable, None]] = [v_nbr_value]
    core_radius: List[float] = [5.0]
    c2c_spacing: List[List[float]] = [[15.0]]
    n_clad: float = 1.02
    omega: float
    kappa_: Union[float, Callable]
    kappa: List[List[Union[List[float], Callable, None]]]
    if (fitting_kappa):
        omega = oc.Domain.lambda_to_omega(Lambda)
        kappa_ = oc.CouplingCoeff.calc_kappa(omega, v_nbr_value,
                                             core_radius[0], c2c_spacing[0][0],
                                             n_clad)
        kappa = [[None]]
    else:
        # k = 1cm^-1
        kappa_ = 1.0 * 1e5 # cm^-1 -> km^-1
        kappa = [[[kappa_]], [[kappa_]]]
    # coupling length is the shortest length where most of power is
    # transmitted to the second core (see p.59 Agrawal applications)
    # coupling length: Lc = pi/(2*k_e) with k_e = sqrt(k^2 + \delta_a^2)
    # if L = Lc -> most power transfers to second core
    # if L = 2*Lc -> most of power stays in primary core
    # if L = Lc / 2 -> 50:50 coupler
    delta_a: float = 0.5*(beta_01 - beta_02)
    length_c: float = oc.PI/(2*math.sqrt(delta_a**2 + kappa_**2))
    length: float = length_c / 2.0

    coupler: oc.FiberCoupler
    coupler = oc.FiberCoupler(length=length, alpha=alpha,
                              kappa=kappa, v_nbr=v_nbr, n_clad=n_clad,
                              c2c_spacing=c2c_spacing, gamma = gamma,
                              ATT=True, DISP=True,
                              nl_approx=False, SPM=True, SS=True, RS=True,
                              XPM=True, XNL=True, ASYM=True, COUP=True,
                              nlse_method = 'rk4ip', steps=steps,
                              ode_method = 'rk4', save=True, wait=True,
                              UNI_OMEGA=True, STEP_UPDATE=False,
                              INTRA_COMP_DELAY=True, INTRA_PORT_DELAY=False,
                              INTER_PORT_DELAY=False, noise_ode_method='rk4',
                              NOISE=True)
    lt.add_link(pulse_1[0], coupler[0])
    lt.add_link(pulse_2[0], coupler[1])

    lt.run(pulse_1, pulse_2)

    y_datas: List[np.ndarray] = [oc.temporal_power(pulse_1[0][0].channels),
                                 oc.temporal_power(pulse_2[0][0].channels),
                                 oc.temporal_power(coupler[2][0].channels),
                                 oc.temporal_power(coupler[3][0].channels),
                                 pulse_1[0][0].noise, coupler[2][0].noise,
                                 pulse_2[0][0].noise, coupler[3][0].noise]
    x_datas: List[np.ndarray] = [pulse_1[0][0].time, pulse_2[0][0].time,
                                 coupler[2][0].time, coupler[3][0].time,
                                 pulse_1[0][0].domain.noise_omega,
                                 coupler[2][0].domain.noise_omega,
                                 pulse_2[0][0].domain.noise_omega,
                                 coupler[3][0].domain.noise_omega]
    plot_groups: List[int] = [0, 1, 2, 3, 4, 4, 5, 5]
    plot_titles: List[str] = ["Original pulse", "Original pulse",
                              "Pulse coming out of the "
                              "coupler with Lk = {}"
                              .format(str(round(length*kappa_,2)))]
    plot_titles.append(plot_titles[-1])

    line_labels: List[Optional[str]] = ["port 0", "port 1", "port 2", "port 3",
                                        "init noise port 0",
                                        "final noise port 2",
                                        "init noise port 1",
                                        "final noise port 3"]

    oc.plot2d(x_datas, y_datas, plot_groups=plot_groups,
              plot_titles=plot_titles, x_labels=['t'], y_labels=['P_t'],
              line_labels=line_labels, line_opacities=[0.3])
