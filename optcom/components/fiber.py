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

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.components.abstract_pass_comp import AbstractPassComp
from optcom.components.abstract_pass_comp import call_decorator
from optcom.domain import Domain
from optcom.equations.abstract_nlse import AbstractNLSE
from optcom.equations.anlse import ANLSE
from optcom.equations.gnlse import GNLSE
from optcom.equations.nlse import NLSE
from optcom.field import Field
from optcom.solvers.field_stepper import FieldStepper
from optcom.solvers.nlse_solver import NLSESolver
from optcom.solvers.ode_solver import ODESolver


default_name = 'Fiber'


class Fiber(AbstractPassComp):
    """A non ideal Fiber.

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

        [0] __________________ [1]

    """

    _nbr_instances: int = 0
    _nbr_instances_with_default_name: int = 0

    def __init__(self, name: str = default_name, length: float = 1.0,
                 alpha: Optional[Union[List[float], Callable]] = None,
                 alpha_order: int = 0,
                 beta: Optional[Union[List[float], Callable]] = None,
                 beta_order: int = 2,
                 gamma: Optional[Union[float, Callable]] = None,
                 sigma: float = cst.XPM_COEFF, eta: float = cst.XNL_COEFF,
                 T_R: float = cst.RAMAN_COEFF,
                 h_R: Optional[Union[float, Callable]] = None,
                 f_R: float = cst.F_R, core_radius: float = cst.CORE_RADIUS,
                 clad_radius: float = cst.CLAD_RADIUS,
                 n_core: Optional[Union[float, Callable]] = None,
                 n_clad: Optional[Union[float, Callable]] = None,
                 NA: Optional[Union[float, Callable]] = None,
                 v_nbr: Optional[Union[float, Callable]] = None,
                 eff_area: Optional[Union[float, Callable]] = None,
                 nl_index: Optional[Union[float, Callable]] = None,
                 nl_approx: bool = True,
                 medium_core: str = cst.FIBER_MEDIUM_CORE,
                 medium_clad: str = cst.FIBER_MEDIUM_CLAD,
                 temperature: float = cst.TEMPERATURE, ATT: bool = True,
                 DISP: bool = True, SPM: bool = True, XPM: bool = False,
                 FWM: bool = False, SS: bool = False, RS: bool = False,
                 XNL: bool = False, NOISE: bool = True,
                 approx_type: int = cst.DEFAULT_APPROX_TYPE,
                 noise_ode_method: str = 'rk4', UNI_OMEGA: bool = True,
                 STEP_UPDATE: bool = False, INTRA_COMP_DELAY: bool = True,
                 INTRA_PORT_DELAY: bool = True, INTER_PORT_DELAY: bool = False,
                 nlse_method: str = "rk4ip", step_method: str = "fixed",
                 steps: int = 100, save: bool = False, save_all: bool = False,
                 max_nbr_pass: Optional[List[int]] = None,
                 pre_call_code: str = '', post_call_code: str = '') -> None:
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
        core_radius :
            The radius of the core. :math:`[\mu m]`
        clad_radius :
            The radius of the cladding. :math:`[\mu m]`
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
        approx_type :
            The type of the NLSE approximation.
        NOISE :
            If True, trigger the noise calculation.
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
        step_method :
            The method for spatial step size generation.
        steps :
            The number of steps for the solver
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
        ports_type = [cst.OPTI_ALL, cst.OPTI_ALL]
        super().__init__(name, default_name, ports_type, save,
                         max_nbr_pass=max_nbr_pass,
                         pre_call_code=pre_call_code,
                         post_call_code=post_call_code)
        # Attr types check ---------------------------------------------
        util.check_attr_type(length, 'length', float)
        util.check_attr_type(alpha, 'alpha', None, Callable, float, List)
        util.check_attr_type(alpha_order, 'alpha_order', int)
        util.check_attr_type(beta, 'beta', None, Callable, float, list)
        util.check_attr_type(beta_order, 'beta_order', int)
        util.check_attr_type(gamma, 'gamma', None, float, Callable)
        util.check_attr_type(sigma, 'sigma', float)
        util.check_attr_type(eta, 'eta', float)
        util.check_attr_type(T_R, 'T_R', float)
        util.check_attr_type(h_R, 'h_R', None, float, callable)
        util.check_attr_type(f_R, 'f_R', float)
        util.check_attr_type(core_radius, 'core_radius', float)
        util.check_attr_type(clad_radius, 'clad_radius', float)
        util.check_attr_type(n_core, 'n_core', None, float, Callable)
        util.check_attr_type(n_clad, 'n_clad', None, float, Callable)
        util.check_attr_type(NA, 'NA', None, float, Callable)
        util.check_attr_type(v_nbr, 'v_nbr', None, float, Callable)
        util.check_attr_type(eff_area, 'eff_area', None, float, Callable)
        util.check_attr_type(nl_index, 'nl_index', None, float, Callable)
        util.check_attr_type(nl_approx, 'nl_approx', bool)
        util.check_attr_type(medium_core, 'medium_core', str)
        util.check_attr_type(medium_clad, 'medium_clad', str)
        util.check_attr_type(temperature, 'temperature', float)
        util.check_attr_type(ATT, 'ATT', bool)
        util.check_attr_type(DISP, 'DISP', bool)
        util.check_attr_type(SPM, 'SPM', bool)
        util.check_attr_type(XPM, 'XPM', bool)
        util.check_attr_type(FWM, 'FWM', bool)
        util.check_attr_type(SS, 'SS', bool)
        util.check_attr_type(RS, 'RS', bool)
        util.check_attr_type(XNL, 'XNL', bool)
        util.check_attr_type(NOISE, 'NOISE', bool)
        util.check_attr_type(approx_type, 'approx_type', int)
        util.check_attr_type(noise_ode_method, 'noise_ode_method', str)
        util.check_attr_type(UNI_OMEGA, 'UNI_OMEGA', bool)
        util.check_attr_type(STEP_UPDATE, 'STEP_UPDATE', bool)
        util.check_attr_type(INTRA_COMP_DELAY, 'INTRA_COMP_DELAY', bool)
        util.check_attr_type(INTRA_PORT_DELAY, 'INTRA_PORT_DELAY', bool)
        util.check_attr_type(INTER_PORT_DELAY, 'INTER_PORT_DELAY', bool)
        util.check_attr_type(nlse_method, 'nlse_method', str)
        util.check_attr_type(step_method, 'step_method', str)
        util.check_attr_type(steps, 'steps', int)
        # Attr ---------------------------------------------------------
        nlse: AbstractNLSE
        if (nl_approx or (not RS and not SS)):
            nlse = ANLSE(alpha, alpha_order, beta, beta_order, gamma, sigma,
                         eta, T_R, core_radius, clad_radius, n_core, n_clad,
                         NA, v_nbr, eff_area, nl_index, medium_core,
                         medium_clad, temperature, ATT, DISP, SPM, XPM, FWM,
                         SS, RS, XNL, NOISE, approx_type, UNI_OMEGA,
                         STEP_UPDATE, INTRA_COMP_DELAY, INTRA_PORT_DELAY,
                         INTER_PORT_DELAY)
        else:
            if (SS):
                nlse = GNLSE(alpha, alpha_order, beta, beta_order, gamma,
                             sigma, eta, h_R, f_R, core_radius, clad_radius,
                             n_core, n_clad, NA, v_nbr, eff_area, nl_index,
                             medium_core, medium_clad, temperature, ATT, DISP,
                             SPM, XPM, FWM, XNL, NOISE, UNI_OMEGA, STEP_UPDATE,
                             INTRA_COMP_DELAY, INTRA_PORT_DELAY,
                             INTER_PORT_DELAY)
            else:
                nlse = NLSE(alpha, alpha_order, beta, beta_order, gamma,
                            sigma, eta, h_R, f_R, core_radius, clad_radius,
                            n_core, n_clad, NA, v_nbr, eff_area, nl_index,
                            medium_core, medium_clad, temperature, ATT, DISP,
                            SPM, XPM, FWM, XNL, NOISE, UNI_OMEGA, STEP_UPDATE,
                            INTRA_COMP_DELAY, INTRA_PORT_DELAY,
                            INTER_PORT_DELAY)
        solver: NLSESolver = NLSESolver(nlse, nlse_method)
        noise_solver: ODESolver = ODESolver(nlse.calc_noise, 'rk4')
        self._stepper: FieldStepper = FieldStepper([solver], [noise_solver],
                                                   length, [steps],
                                                   [step_method],
                                                   save_all=save_all)
        # Policy -------------------------------------------------------
        self.add_port_policy(([0], [1], True))
    # ==================================================================
    @call_decorator
    def __call__(self, domain: Domain, ports: List[int], fields: List[Field]
                 ) -> Tuple[List[int], List[Field]]:

        output_fields: List[Field] = []
        output_fields = self._stepper(domain, fields)
        if (self._stepper.save_all):
            self.storages.append(self._stepper.storage)

        return self.output_ports(ports), output_fields


if __name__ == "__main__":
    """Give an example of Fiber usage.
    This piece of code is standalone, i.e. can be used in a separate
    file as an example.
    """

    from typing import Callable, List, Optional

    import numpy as np

    import optcom as oc

    noise_samples = 200
    domain: oc.Domain = oc.Domain(samples_per_bit=2048,bit_width=70.0,
                                  noise_samples=noise_samples)
    lt: oc.Layout = oc.Layout(domain)

    pulse: oc.Gaussian
    pulse = oc.Gaussian(channels=4, peak_power=[10.0, 10e-1, 5.0, 7.0],
                        width=[0.1, 5.0, 3.0, 4.0],
                        center_lambda=[1050.0, 1048.0, 1049.0, 1051.0],
                        noise=np.ones(noise_samples)*4)
    fiber = oc.Fiber(length=0.05, nlse_method="ssfm", alpha=[.46],
                     nl_approx=False, ATT=True, DISP=True, gamma=10.,
                     SPM=True, XPM=False, SS=True, RS=True, XNL=False,
                     approx_type=1, steps=1000, medium_core='sio2',
                     UNI_OMEGA=True, STEP_UPDATE=False, save_all=True,
                     INTRA_COMP_DELAY=True, INTRA_PORT_DELAY=False,
                     INTER_PORT_DELAY=False, noise_ode_method='rk1',
                     NOISE=True)
    lt.add_link(pulse[0], fiber[0])
    lt.run(pulse)

    x_datas: List[np.ndarray] = [pulse[0][0].nu, fiber[1][0].nu,
                                 pulse[0][0].time, fiber[1][0].time]

    y_datas: List[np.ndarray] = [oc.spectral_power(pulse[0][0].channels),
                                 oc.spectral_power(fiber[1][0].channels),
                                 oc.temporal_power(pulse[0][0].channels),
                                 oc.temporal_power(fiber[1][0].channels)]

    x_labels: List[str] = ['nu', 'nu', 't', 't']
    y_labels: List[str] = ['P_nu', 'P_nu', 'P_t', 'P_t']
    plot_titles: List[str] = ["Original Pulses",
                              "Pulses at the end of the fiber"]
    plot_titles.extend(plot_titles)

    oc.plot2d(x_datas, y_datas, x_labels=x_labels, y_labels=y_labels,
              plot_titles=plot_titles, split=True, line_opacities=[0.3])
