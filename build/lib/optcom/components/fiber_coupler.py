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

import math
import copy

import numpy as np
from typing import Callable, List, Optional, Sequence, Tuple, Union

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.components.abstract_pass_comp import AbstractPassComp
from optcom.domain import Domain
from optcom.equations.canlse import CANLSE
from optcom.equations.cgnlse import CGNLSE
from optcom.equations.cnlse import CNLSE
from optcom.field import Field
from optcom.solvers.stepper import Stepper


default_name = 'Fiber Coupler'
OPTIONAL_LIST_CALL_FLOAT = Optional[Union[List[List[float]], List[Callable]]]


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
        If True, the last wave to enter/exit a port will be saved.

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
                 max_nbr_pass: Optional[List[int]] = None, nbr_fibers: int = 2,
                 alpha: OPTIONAL_LIST_CALL_FLOAT = None,
                 alpha_order: int = 1,
                 beta: OPTIONAL_LIST_CALL_FLOAT = None,
                 beta_order: int = 2,
                 gamma: Optional[Union[List[float], List[Callable]]] = None,
                 kappa: Optional[List[List[List[float]]]] = None,
                 sigma: List[float] = [cst.KERR_COEFF],
                 sigma_cross: List[List[float]] = [[cst.KERR_COEFF_CROSS]],
                 eta: float = cst.XPM_COEFF, T_R: float = cst.RAMAN_COEFF,
                 tau_1: float =cst.TAU_1, tau_2: float =cst.TAU_2,
                 f_R: float = cst.F_R, nl_approx: bool = True,
                 NA: Union[List[float], List[Callable]] = [cst.NA],
                 ATT: bool = True, DISP: bool = True, SPM: bool = True,
                 XPM: bool = False, FWM: bool = False, SS: bool = False,
                 RS: bool = False, ASYM: bool = True, COUP: bool = True,
                 approx_type: int = cst.DEFAULT_APPROX_TYPE,
                 medium: str = cst.DEF_FIBER_MEDIUM,
                 c2c_spacing: List[List[float]] = [[cst.C2C_SPACING]],
                 core_radius: List[float] = [cst.CORE_RADIUS],
                 V: List[float] = [cst.V], n_0: List[float] = [cst.REF_INDEX],
                 method: str = "rk4ip", steps: int = 100, save: bool = False,
                 save_all: bool = False, wait: bool = False) -> None:
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
            Positive term multiplying the XPM term of the NLSE.
        sigma_cross :
            Positive term multiplying the XPM term of the NLSE inbetween
            the fibers.
        eta :
            Positive term muiltiplying the XPM in other non linear
            terms of the NLSE.
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
        SS :
            If True, trigger the self-steepening.
        RS :
            If True, trigger the Raman scattering.
        ASYM :
            If True, trigger the asymmetry effects between cores.
        COUP :
            If True, trigger the coupling effects between cores.
        approx_type :
            The type of the NLSE approximation.
        medium :
            The main medium of the fiber.
        c2c_spacing :
            The center to center distance between two cores.
            :math:`[\mu m]`
        core_radius :
            The core radius. :math:`[\mu m]`
        V :
            The fiber parameter.
        n_0 :
            The refractive index outside of the waveguides.
        method :
            The solver method type
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

        """
        # Parent constructor -------------------------------------------
        ports_type = [cst.OPTI_ALL for i in range(4)]
        super().__init__(name, default_name, ports_type, save, wait=wait,
                         max_nbr_pass=max_nbr_pass)
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
        util.check_attr_type(eta, 'eta', float)
        util.check_attr_type(T_R, 'T_R', float)
        util.check_attr_type(tau_1, 'tau_1', float)
        util.check_attr_type(tau_2, 'tau_2', float)
        util.check_attr_type(f_R, 'f_R', float)
        util.check_attr_type(nl_approx, 'nl_approx', bool)
        util.check_attr_type(NA, 'NA', float, list, Callable)
        util.check_attr_type(ATT, 'ATT', bool)
        util.check_attr_type(DISP, 'DISP', bool)
        util.check_attr_type(SPM, 'SPM', bool)
        util.check_attr_type(XPM, 'XPM', bool)
        util.check_attr_type(FWM, 'FWM', bool)
        util.check_attr_type(SS, 'SS', bool)
        util.check_attr_type(RS, 'RS', bool)
        util.check_attr_type(ASYM, 'ASYM', bool)
        util.check_attr_type(COUP, 'COUP', bool)
        util.check_attr_type(approx_type, 'approx_type', int)
        util.check_attr_type(medium, 'medium', str)
        util.check_attr_type(c2c_spacing, 'c2c_spacing', float, list)
        util.check_attr_type(core_radius, 'core_radius', float, list)
        util.check_attr_type(V, 'V', float, list)
        util.check_attr_type(n_0, 'n_0', float, list)
        util.check_attr_type(method, 'method', str)
        util.check_attr_type(steps, 'steps', int)
        # Attr ---------------------------------------------------------
        if (nl_approx):
            cnlse = CANLSE(nbr_fibers, alpha, alpha_order, beta, beta_order,
                           gamma, kappa, sigma, sigma_cross, eta, T_R, NA,
                           ATT, DISP, SPM, XPM, FWM, SS, RS, ASYM, COUP,
                           approx_type, medium, c2c_spacing, core_radius, V,
                           n_0)
        else:
            if (SS):
                cnlse = CGNLSE(nbr_fibers, alpha, alpha_order, beta,
                               beta_order, gamma, kappa, sigma, sigma_cross,
                               tau_1, tau_2, f_R, NA, ATT, DISP, SPM, XPM, FWM,
                               ASYM, COUP, medium, c2c_spacing, core_radius, V,
                               n_0)
            else:
                cnlse = CNLSE(nbr_fibers, alpha, alpha_order, beta, beta_order,
                              gamma, kappa, sigma, sigma_cross, tau_1, tau_2,
                              f_R, NA, ATT, DISP, SPM, XPM, FWM, ASYM, COUP,
                              medium, c2c_spacing, core_radius, V, n_0)

        method_2 = 'euler'    # only euler for now
        step_method = 'fixed'   # only fixed for now
        self._stepper = Stepper([cnlse, cnlse], [method, method_2],
                                length, [steps, steps], [step_method],
                                "alternating", save_all=save_all)
        # Policy -------------------------------------------------------
        self.add_port_policy(([0, 1], [2, 3], True))
        self.add_wait_policy([0, 1], [2, 3])
    # ==================================================================
    def __call__(self, domain: Domain, ports: List[int], fields: List[Field]
                 ) -> Tuple[List[int], List[Field]]:

        output_fields: List[Field] = []
        # 1 input ------------------------------------------------------
        uni_ports = util.unique(ports)
        if (len(uni_ports) == 1):
            # 0 -> 1 /\ 1 -> 0 /\ 2 -> 3 /\ 3 -> 2
            analog_port = uni_ports[0] ^ 1
            ports.append(analog_port)
            null_field = copy.deepcopy(fields[0])
            null_field.reset_channel()
            fields.append(null_field)
        # 2 inputs -----------------------------------------------------
        uni_ports = util.unique(ports)
        if (len(uni_ports) == 2):
            fields_1 = []
            fields_2 = []
            for i in range(len(ports)):
                if (ports[i] == uni_ports[0]):
                    fields_1.append(fields[i])
                else:
                    fields_2.append(fields[i])
            output_fields = self._stepper(domain, fields_1, fields_2)

        return self.output_ports(ports), output_fields


if __name__ == "__main__":

    import random

    import optcom.utils.plot as plot
    import optcom.layout as layout
    import optcom.components.gaussian as gaussian
    import optcom.domain as domain
    from optcom.utils.utilities_user import temporal_power
    from optcom.effects.coupling import Coupling

    lt = layout.Layout(domain.Domain(bit_width=1.0, samples_per_bit=1024))

    Lambda = 1030.0
    pulse_1 = gaussian.Gaussian(channels=1, peak_power=[1.0, 0.5], width=[.1],
                                center_lambda=[Lambda])
    pulse_2 = gaussian.Gaussian(channels=2, peak_power=[1.0, 0.5], width=[.1],
                                center_lambda=[1050.0])

    steps = int(1e3)
    alpha = [[0.046], [0.046]]
    beta = [[1e5,10.0,-0.0],[1e5,10.0,-0.0]]
    gamma = [4.3, 4.3]
    fitting_kappa = True
    V = [2.0]
    core_radius = [5.0]
    c2c_spacing = [[15.0]]
    n_0 = [1.02]
    if (fitting_kappa):
        omega = Domain.lambda_to_omega(Lambda)
        kappa_ = Coupling.calc_kappa(omega, V[0], core_radius[0],
                                     c2c_spacing[0][0], n_0[0])
        kappa = None
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
    delta_a = 0.5*(beta[0][0] - beta[1][0])
    length_c = cst.PI/(2*math.sqrt(delta_a**2 + kappa_**2))
    length = length_c / 2

    coupler = FiberCoupler(length=length, alpha=alpha, beta=beta,# gamma=gamma,
                           kappa=kappa, V=V, n_0=n_0, core_radius=core_radius,
                           c2c_spacing=c2c_spacing, ATT=True, DISP=True,
                           nl_approx=False, SPM=True, SS=True, RS=True,
                           XPM=True, ASYM=True, COUP=True, approx_type=1,
                           method='ssfm_super_sym', steps=steps, save=True,
                           wait=False)

    lt.link((pulse_1[0], coupler[0]))
    lt.link((pulse_2[0], coupler[1]))

    lt.run(pulse_1)
    #lt.run(pulse_1, pulse_2)

    fields = [temporal_power(pulse_1.fields[0].channels),
              temporal_power(coupler.fields[2].channels),
              temporal_power(coupler.fields[3].channels)]
    time = [pulse_1.fields[0].time, coupler.fields[2].time,
            coupler.fields[3].time]
    plot_groups = [0, 1, 2]
    plot_titles = ["Original pulse", "Pulse coming out of the coupler with " +
                    "Lk = {}".format(str(round(length*kappa_,2)))]
    plot_titles.append(plot_titles[-1])

    plot_labels = ["port 0", "port 2", "port 3"]


    plot.plot(time, fields, plot_groups=plot_groups, plot_titles=plot_titles,
              x_labels=['t'], y_labels=['P_t'], plot_labels=plot_labels,
              opacity=0.3)
