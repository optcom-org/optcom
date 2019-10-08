# This file is part of Optcom.
#
# Optcom is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Optcom is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Optcom.  If not, see <https://www.gnu.org/licenses/>.

""".. moduleauthor:: Sacha Medaer"""

from typing import Callable, List, Optional, Sequence, Tuple, Union

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.components.abstract_pass_comp import AbstractPassComp
from optcom.domain import Domain
from optcom.equations.anlse import ANLSE
from optcom.equations.gnlse import GNLSE
from optcom.equations.nlse import NLSE
from optcom.field import Field
from optcom.solvers.stepper import Stepper


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
        If True, the last wave to enter/exit a port will be saved.

    Notes
    -----
    Component diagram::

        [0] __________________ [1]

    """

    _nbr_instances: int = 0
    _nbr_instances_with_default_name: int = 0

    def __init__(self, name: str = default_name, length: float = 1.0,
                 alpha: Optional[Union[List[float], Callable]] = None,
                 alpha_order: int = 1,
                 beta: Optional[Union[List[float], Callable]] = None,
                 beta_order: int = 2,
                 gamma: Optional[Union[float, Callable]] = None,
                 sigma: float = cst.KERR_COEFF, eta: float = cst.XPM_COEFF,
                 T_R: float = cst.RAMAN_COEFF,
                 tau_1: float = cst.TAU_1, tau_2: float = cst.TAU_2,
                 f_R: float = cst.F_R, core_radius: float = cst.CORE_RADIUS,
                 NA: Union[float, Callable] = cst.NA, nl_approx: bool = True,
                 ATT: bool = True, DISP: bool = True, SPM: bool = True,
                 XPM: bool = False, FWM: bool = False, SS: bool = False,
                 RS: bool = False, approx_type: int = cst.DEFAULT_APPROX_TYPE,
                 method: str = "rk4ip", steps: int = 100,
                 medium: str = cst.DEF_FIBER_MEDIUM, save: bool = False,
                 save_all: bool = False) -> None:
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
            Positive term multiplying the XPM term of the NLSE
        eta :
            Positive term muiltiplying the XPM in other non linear
            terms of the NLSE
        T_R :
            The raman coefficient. :math:`[]`
        tau_1 :
            The inverse of vibrational frequency of the fiber core
            molecules. :math:`[ps]`
        tau_2 :
            The damping time of vibrations. :math:`[ps]`
        f_R :
            The fractional contribution of the delayed Raman response.
            :math:`[]`
        core_radius :
            The radius of the core. :math:`[\mu m]`
        NA :
            The numerical aperture.
        nl_approx :
            If True, the approximation of the NLSE is used.
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
        approx_type :
            The type of the NLSE approximation.
        method :
            The solver method type
        steps :
            The number of steps for the solver
        medium :
            The main medium of the fiber.
        save :
            If True, the last wave to enter/exit a port will be saved.
        save_all :
            If True, save the wave at each spatial step in the
            component.

        """
        # Parent constructor -------------------------------------------
        ports_type = [cst.OPTI_ALL, cst.OPTI_ALL]
        super().__init__(name, default_name, ports_type, save)
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
        util.check_attr_type(tau_1, 'tau_1', float)
        util.check_attr_type(tau_2, 'tau_2', float)
        util.check_attr_type(f_R, 'f_R', float)
        util.check_attr_type(nl_approx, 'nl_approx', bool)
        util.check_attr_type(core_radius, 'core_radius', float)
        util.check_attr_type(NA, 'NA', float, Callable)
        util.check_attr_type(ATT, 'ATT', bool)
        util.check_attr_type(DISP, 'DISP', bool)
        util.check_attr_type(SPM, 'SPM', bool)
        util.check_attr_type(XPM, 'XPM', bool)
        util.check_attr_type(FWM, 'FWM', bool)
        util.check_attr_type(SS, 'SS', bool)
        util.check_attr_type(RS, 'RS', bool)
        util.check_attr_type(approx_type, 'approx_type', int)
        util.check_attr_type(method, 'method', str)
        util.check_attr_type(steps, 'steps', int)
        util.check_attr_type(medium, 'medium', str)
        # Attr ---------------------------------------------------------
        if (nl_approx):
            nlse = ANLSE(alpha, alpha_order, beta, beta_order, gamma, sigma,
                         eta, T_R, core_radius, NA, ATT, DISP, SPM, XPM, FWM,
                         SS, RS, approx_type, medium)
        else:
            if (SS):
                nlse = GNLSE(alpha, alpha_order, beta, beta_order, gamma,
                             sigma, tau_1, tau_2, f_R, core_radius, NA, ATT,
                             DISP, SPM, XPM, FWM, medium)
            else:
                nlse = NLSE(alpha, alpha_order, beta, beta_order, gamma, sigma,
                            tau_1, tau_2, f_R, core_radius, NA, ATT, DISP, SPM,
                            XPM, FWM, medium)
        # Special case for gnlse and rk4ip method
        if (SS and not nl_approx and (method == "rk4ip") and cst.RK4IP_GNLSE):
            method = "rk4ip_gnlse"
        step_method = "fixed"   # to change later when implementing adaptative
        self._stepper = Stepper([nlse], [method], length, [steps],
                                [step_method], save_all=save_all)
        # Policy -------------------------------------------------------
        self.add_port_policy(([0], [1], True))
    # ==================================================================
    def __call__(self, domain: Domain, ports: List[int], fields: List[Field]
                 ) -> Tuple[List[int], List[Field]]:

        output_ports: List[int] = []
        output_fields: List[Field] = []
        output_fields = self._stepper(domain, fields)
        output_ports = self.output_ports(ports)

        return output_ports, output_fields


if __name__ == "__main__":

    import sys
    import numpy as np
    from scipy import interpolate
    import optcom.utils.plot as plot
    import optcom.layout as layout
    import optcom.components.gaussian as gaussian
    import optcom.domain as domain
    from optcom.utils.utilities_user import temporal_power, spectral_power,\
        CSVFit


    lt = layout.Layout(domain.Domain(samples_per_bit=512,bit_width=20.0))
    pulse = gaussian.Gaussian(channels=2, peak_power=[10.0, 10e-1],
                              width=[1.0, 5.0], center_lambda=[1050.0, 1048.0])
    gamma_data = CSVFit('./data/gamma_test.txt')
    fiber = Fiber(length=.10, method="ssfm_symmetric", alpha=[0.046],
                  alpha_order=4, beta_order=4, gamma=1.5,
                  nl_approx=False, ATT=True, DISP=True,
                  SPM=True, XPM=False, SS=False, RS=False, approx_type=1,
                  steps=1000, medium='sio2')
    lt.link((pulse[0], fiber[0]))
    lt.run(pulse)

    x_datas = [pulse.fields[0].nu, fiber.fields[1].nu,
               pulse.fields[0].time, fiber.fields[1].time]

    y_datas = [spectral_power(pulse.fields[0].channels),
               spectral_power(fiber.fields[1].channels),
               temporal_power(pulse.fields[0].channels),
               temporal_power(fiber.fields[1].channels)]

    x_labels = ['nu', 'nu', 't', 't']
    y_labels = ['P_nu', 'P_nu', 'P_t', 'P_t']
    plot_titles = ["Original Pulse", "Pulse at the end of the fiber"]
    plot_titles.extend(plot_titles)

    plot.plot(x_datas, y_datas, x_labels = x_labels, y_labels = y_labels,
              plot_titles=plot_titles, plot_groups=[0,1,2,3], opacity=0.3)
