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

import copy
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
from nptyping import Array

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.components.abstract_pass_comp import AbstractPassComp
from optcom.domain import Domain
from optcom.equations.ampanlse import AmpANLSE
from optcom.equations.ampgnlse import AmpGNLSE
from optcom.equations.ampnlse import AmpNLSE
from optcom.equations.re2_fiber import RE2Fiber
from optcom.solvers.stepper import Stepper
from optcom.field import Field


default_name = 'Fiber Amplifier'


class FiberAmplifier(AbstractPassComp):
    r"""A non ideal Fiber Amplifier.

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
    N_0 : numpy.ndarray of float
        The population in the ground state.
    N_1 : numpy.ndarray of float
        The population in the metastable state.
    power_signal_forward : numpy.ndarray of float
        Power of the foward propagating signal for each channel.
    power_signal_backward : numpy.ndarray of float
        Power of the backward propagating signal for each channel.
    power_ase_forward : numpy.ndarray of float
        Power of the foward propagating ase for each channel.
    power_ase_backward : numpy.ndarray of float
        Power of the backward propagating ase for each channel.
    power_pump_forward : numpy.ndarray of float
        Power of the foward propagating pump for each channel.
    power_pump_backward : numpy.ndarray of float
        Power of the backward propagating pump for each channel.

    Notes
    -----
    Component diagram::

        [0] _____________________ [1]
            /                   \
           /                     \
        [2]                       [3]


    [0] and [1] : signal and [2] and [3] : pump

    """

    _nbr_instances: int = 0
    _nbr_instances_with_default_name: int = 0

    def __init__(self, name: str = default_name, length: float = 1.0,
                 alpha: Optional[Union[List[float], Callable]] = None,
                 alpha_order: int = 1,
                 beta: Optional[Union[List[float], Callable]] = None,
                 beta_order: int = 2,
                 gamma: Optional[Union[float, Callable]] = None,
                 gain_order: int = 1, sigma: float = cst.KERR_COEFF,
                 eta: float = cst.XPM_COEFF, T_R: float = cst.RAMAN_COEFF,
                 tau_1: float = cst.TAU_1, tau_2: float = cst.TAU_2,
                 f_R: float = cst.F_R, nl_approx: bool = True,
                 sigma_a: Optional[Union[List[float], Callable]] = None,
                 sigma_e: Optional[Union[List[float], Callable]] = None,
                 n_core: Optional[Union[float, List[float]]] = None,
                 n_clad: Optional[Union[float, List[float]]] = None,
                 NA: Optional[Union[float, List[float]]] = None,
                 temperature: float = 293.15, tau_meta: float = cst.TAU_META,
                 N_T: float = cst.N_T, core_radius: float = cst.CORE_RADIUS,
                 clad_radius: float = cst.CLAD_RADIUS,
                 area_doped: Optional[float] = None,
                 eta_s: float = cst.ETA_SIGNAL, eta_p: float = cst.ETA_PUMP,
                 R_0: float = cst.R_0, R_L: float = cst.R_L,
                 signal_width: List[float] = [1.0],
                 nl_index: Optional[Union[float, Callable]] = None,
                 ATT: bool = True, DISP: bool = True, SPM: bool = True,
                 XPM: bool = False, FWM: bool = False, SS: bool = False,
                 RS: bool = False, GS: bool = True,
                 approx_type: int = cst.DEFAULT_APPROX_TYPE,
                 medium: str = cst.DEF_FIBER_MEDIUM,
                 dopant: str = cst.DEF_FIBER_DOPANT,
                 method: str = "rk4ip", steps: int = 100,
                 solver_order: str = 'following', error: float = 0.01,
                 propagate_pump: bool = False, save: bool = False,
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
        gain_order :
            The order of the gain coefficients to take into account.
            (from the Rate Equations resolution)
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
        nl_approx :
            If True, the approximation of the NLSE is used.
        sigma_a :
            The absorption cross sections of the signal and the pump
            (1<=len(sigma_a)<=2). :math:`[nm^2]` If a callable is
            provided, varibale must be wavelength. :math:`[nm]`
        sigma_e :
            The emission cross sections of the signal and the pump
            (1<=len(sigma_a)<=2). :math:`[nm^2]` If a callable is
            provided, varibale must be wavelength. :math:`[nm]`
        n_core :
            The refractive index of the core. If the medium is not
            recognised by Optcom, at least two elements should be
            provided out of those three: n_core, n_clad, NA.
        n_clad :
            The refractive index of the cladding. If the medium is not
            recognised by Optcom, at least two elements should be
            provided out of those three: n_core, n_clad, NA.
        NA :
            The numerical aperture. If the medium is not
            recognised by Optcom, at least two elements should be
            provided out of those three: n_core, n_clad, NA.
        temperature :
            The temperature of the medium. :math:`[K]`
        tau_meta :
            The metastable level lifetime. :math:`[\mu s]`
        N_T :
            The total doping concentration. :math:`[nm^{-3}]`
        core_radius :
            The radius of the core. :math:`[\mu m]`
        clad_radius :
            The radius of the cladding. :math:`[\mu m]`
        area_doped :
            The doped area. :math:`[\mu m^2]` If None, will be
            approximated to the core area.
        eta_s :
            The background signal loss. :math:`[km^{-1}]`
        eta_p :
            The background pump loss. :math:`[km^{-1}]`
        R_0 :
            The reflectivity at the fiber start.
        R_L :
            The reflectivity at the fiber end.
        signal_width :
            The width of each channel of the signal. :math:`[ps]`
        nl_index :
            The non linear index. Used to calculate the non linear
            parameter. :math:`[m^2\cdot W^{-1}]`
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
        GS :
            If True, trigger the gain saturation.
        approx_type :
            The type of the NLSE approximation.
        medium :
            The main medium of the fiber amplifier.
        dopant :
            The doped medium of the fiber amplifier.
        method :
            The solver method type
        steps :
            The number of steps for the solver
        solver_order:
            The order in which to solve RE and NLSE. Can be either
            "following" or "alternating".
        error :
            The error for convergence criterion of stepper resolution.
        propagate_pump :
            If True, the pump is propagated forward in the layout.
        save :
            If True, the last wave to enter/exit a port will be saved.
        save_all :
            If True, save the wave at each spatial step in the
            component.

        """
        # Parent constructor -------------------------------------------
        ports_type = [cst.OPTI_ALL, cst.OPTI_ALL, cst.OPTI_IN, cst.OPTI_IN]
        super().__init__(name, default_name, ports_type, save, wait=True)
        # Attr types check ---------------------------------------------
        util.check_attr_type(length, 'length', float)
        util.check_attr_type(alpha, 'alpha', None, Callable, float, List)
        util.check_attr_type(alpha_order, 'alpha_order', int)
        util.check_attr_type(beta, 'beta', None, Callable, float, list)
        util.check_attr_type(beta_order, 'beta_order', int)
        util.check_attr_type(gamma, 'gamma', None, float, Callable)
        util.check_attr_type(gain_order, 'gain_order', int)
        util.check_attr_type(sigma, 'sigma', float)
        util.check_attr_type(eta, 'eta', float)
        util.check_attr_type(T_R, 'T_R', float)
        util.check_attr_type(tau_1, 'tau_1', float)
        util.check_attr_type(tau_2, 'tau_2', float)
        util.check_attr_type(f_R, 'f_R', float)
        util.check_attr_type(nl_approx, 'nl_approx', bool)
        util.check_attr_type(sigma_a, 'sigma_a', None, float, list, Callable)
        util.check_attr_type(sigma_e, 'sigma_e', None, float, list, Callable)
        util.check_attr_type(n_core, 'n_core', None, float, list)
        util.check_attr_type(n_clad, 'n_clad', None, float, list)
        util.check_attr_type(NA, 'NA', None, float, list)
        util.check_attr_type(temperature, 'temperature', float)
        util.check_attr_type(tau_meta, 'tau_meta', float)
        util.check_attr_type(N_T, 'N_T', float)
        util.check_attr_type(core_radius, 'core_radius', float)
        util.check_attr_type(clad_radius, 'clad_radius', float)
        util.check_attr_type(area_doped, 'area_doped', None, float)
        util.check_attr_type(eta_s, 'eta_s', float)
        util.check_attr_type(eta_p, 'eta_p', float)
        util.check_attr_type(R_0, 'R_0', float)
        util.check_attr_type(R_L, 'R_L', float)
        util.check_attr_type(signal_width, 'signal_width', list)
        util.check_attr_type(nl_index, 'nl_index', None, float, Callable)
        util.check_attr_type(ATT, 'ATT', bool)
        util.check_attr_type(DISP, 'DISP', bool)
        util.check_attr_type(SPM, 'SPM', bool)
        util.check_attr_type(XPM, 'XPM', bool)
        util.check_attr_type(FWM, 'FWM', bool)
        util.check_attr_type(SS, 'SS', bool)
        util.check_attr_type(RS, 'RS', bool)
        util.check_attr_type(GS, 'GS', bool)
        util.check_attr_type(approx_type, 'approx_type', int)
        util.check_attr_type(medium, 'medium', str)
        util.check_attr_type(dopant, 'dopant', str)
        util.check_attr_type(method, 'method', str)
        util.check_attr_type(steps, 'steps', int)
        util.check_attr_type(solver_order, 'solver_order', str)
        util.check_attr_type(error, 'error', float)
        util.check_attr_type(propagate_pump, 'propagate_pump', bool)
        # Attr ---------------------------------------------------------
        # Component equations ------------------------------------------
        step_update: bool = False if (solver_order == 'following') else True
        re = RE2Fiber(sigma_a, sigma_e, n_core, n_clad, NA, temperature,
                      tau_meta, N_T, core_radius, clad_radius, area_doped,
                      eta_s, eta_p, R_0, R_L, signal_width, medium, dopant,
                      step_update)
        if (nl_approx):
            nlse = AmpANLSE(re, alpha, alpha_order, beta, beta_order, gamma,
                            gain_order, sigma, eta, T_R, R_0, R_L, nl_index,
                            ATT, DISP, SPM, XPM, FWM, SS, RS, approx_type, GS,
                            medium, dopant)
        else:
            if (SS):
                nlse = AmpGNLSE(re, alpha, alpha_order, beta, beta_order,
                                gamma, gain_order, sigma, tau_1, tau_2, f_R,
                                R_0, R_L, nl_index, ATT, DISP, SPM, XPM, FWM,
                                GS, medium, dopant)
            else:
                nlse = AmpNLSE(re, alpha, alpha_order, beta, beta_order, gamma,
                               gain_order, sigma, tau_1, tau_2, f_R, R_0, R_L,
                               nl_index, ATT, DISP, SPM, XPM, FWM, GS, medium,
                               dopant)
        # Component stepper --------------------------------------------
        # Special case for gnlse and rk4ip method
        if (SS and not nl_approx and (method == "rk4ip")
                and cst.RK4IP_GNLSE):
            method = "rk4ip_gnlse"
        step_method = "fixed"   # for now, only works with "fixed"
        eqs = [re, nlse]
        stepper_method: List[str]
        if (solver_order == 'following'):
            stepper_method = ['shooting', 'forward']
        else:
            stepper_method = ['shooting', 'shooting']
        # if method is empty '', will directly call the equation object
        self._stepper = Stepper(eqs, ['', method], length, [steps],
                                [step_method], solver_order, stepper_method,
                                error=error,save_all=save_all)
        self.N_0: Array[float]
        self.N_1: Array[float]
        self.power_ase_forward: Array[float]
        self.power_ase_backward: Array[float]
        self.power_signal_forward: Array[float]
        self.power_signal_backward: Array[float]
        self.power_pump_forward: Array[float]
        self.power_pump_backward: Array[float]
        # Policy -------------------------------------------------------
        if (propagate_pump):
            self.add_port_policy(([0,2], [1,1], False),
                                    ([0,3], [1,1], False),
                                    ([1,2], [0,0], False),
                                    ([1,3], [0,1], False))
        else:
            self.add_port_policy(([0,2], [1,-1], False),
                                    ([0,3], [1,-1], False),
                                    ([1,2], [0,-1], False),
                                    ([1,3], [0,-1], False))
        self.add_wait_policy([0,2], [0,3], [1,2], [1,3])
    # ==================================================================
    def __call__(self, domain: Domain, ports: List[int], fields: List[Field]
                 ) -> Tuple[List[int], List[Field]]:
        output_ports: List[int] = []
        output_fields: List[Field] = []
        pump_ports: List[int] = []
        pump_fields: List[Field] = []
        # Sort fields --------------------------------------------------
        i = 0
        while (i < len(ports)):
            if (ports[i] == 2 or ports[i] == 3): # Port from pump
                pump_fields.append(fields.pop(i))
                pump_ports.append(ports.pop(i))
            i += 1
        ports.extend(pump_ports)
        # Set shooting method ------------------------------------------
        if (util.sum_elem_list(util.unique(ports)) == 3):
            self._stepper.start_shooting_backward()
            util.warning_terminal("Counter propagating pulses in fiber "
                "amplifier not totally handled yet, might lead to unrealistic "
                "results.")
        else:
            self._stepper.start_shooting_forward()
        # Compute ------------------------------------------------------
        output_fields = self._stepper(domain, fields, pump_fields)
        # Record RE parameters -----------------------------------------
        re = self._stepper.equations[0]
        # If shooting method stops earlier than expected, those values
        # won't be accurate anymore. -> need a back up in RE2Fiber ?
        self.N_0 = re.N_0
        self.N_1 = re.N_1
        self.power_ase_forward = re.power_ase_forward
        self.power_ase_backward = re.power_ase_backward
        self.power_signal_forward = re.power_signal_forward
        self.power_signal_backward = re.power_signal_backward
        self.power_pump_forward = re.power_pump_forward
        self.power_pump_backward = re.power_pump_backward

        return self.output_ports(ports), output_fields


if __name__ == "__main__":

    import numpy as np
    import optcom.utils.plot as plot
    import optcom.layout as layout
    import optcom.components.cw as cw
    import optcom.components.gaussian as gaussian
    import optcom.domain as domain
    from optcom.utils.utilities_user import temporal_power, spectral_power,\
        CSVFit

    lt = layout.Layout(domain.Domain(samples_per_bit=512, bit_width=5.0,
                                     memory_storage=1.0))
    nbr_ch_s = 3
    signal_width = [0.1, 0.2, 0.1]
    or_p = 1e-4
    pulse = gaussian.Gaussian(channels=nbr_ch_s,
                              peak_power=[1.6*or_p, 1.3*or_p, 1.2*or_p],
                              width=signal_width,
                              center_lambda=[1030.0, 1025.0, 1019.0])
    nbr_ch_p = 2
    #pump = gaussian.Gaussian(channels=nbr_ch_p, peak_power=[45.0, 35.0],
    #                         center_lambda=[940.0, 977.0], width=[7.0, 6.0])
    pump = cw.CW(channels=nbr_ch_p, peak_power=[4*or_p, 3*or_p],
                 center_lambda=[976.0, 940.0])

    steps = 500
    length = 0.0005     # km

    file_sigma_a = ('./data/fiber_amp/cross_section/absorption/yb.txt')
    file_sigma_e = ('./data/fiber_amp/cross_section/emission/yb.txt')
    sigma_a = CSVFit(file_sigma_a, conv_factor=[1e9, 1e18])
    sigma_e = CSVFit(file_sigma_e, conv_factor=[1e9, 1e18])
    #sigma_a = [3e-8, 2e-6]
    #sigma_e = [2e-7, 1e-6]
    #sigma_e = None
    n_core = None
    n_clad = None
    NA = 0.2
    r_core = 5.0    # um
    r_clad = 62.5   # um
    temperature = 293.15    # K
    tau_meta = 840.0  # us
    N_T = 6.3e-2    # nm^{-3}
    eta_s = 1.26    # km^-1
    eta_p = 1.41    # km^-1
    R_0 = 8e-4
    R_L = R_0
    medium = 'sio2'
    dopant = 'yb'
    error = 2.


    fiber = FiberAmplifier(length=length, method="ssfm_super_sym",
                           alpha=[0.046], beta_order = 3,
                           #beta=[0.0,0.0,10.0,-19.83,0.031],
                           #gamma=0.43,
                           gain_order=1, GS=True,
                           nl_approx=False, ATT=True, DISP=True,
                           SPM=True, XPM=True, SS=True, RS=True,
                           approx_type=1, sigma_a=sigma_a, sigma_e=sigma_e,
                           n_core=n_core, n_clad=n_clad, NA=NA,
                           core_radius=r_core, clad_radius=r_clad,
                           temperature=temperature, tau_meta=tau_meta,
                           N_T=N_T, eta_s=eta_s, eta_p=eta_p, R_0=R_0, R_L=R_L,
                           signal_width=signal_width,
                           medium=medium, dopant=dopant, steps=steps,
                           solver_order='alternating', error=error,
                           propagate_pump=True, save_all=True)
    lt.link((pulse[0], fiber[0]), (pump[0], fiber[2]))
    lt.run(pulse, pump)

    x_datas = [pulse.fields[0].nu, pump.fields[0].nu, pulse.fields[0].time,
               pump.fields[0].time, fiber.fields[1].nu, fiber.fields[1].time]
    y_datas = [spectral_power(pulse.fields[0].channels),
               spectral_power(pump.fields[0].channels),
               temporal_power(pulse.fields[0].channels),
               temporal_power(pump.fields[0].channels),
               spectral_power(fiber.fields[1].channels),
               temporal_power(fiber.fields[1].channels)]
    x_labels = ['nu', 'nu', 't', 't', 'nu', 't']
    y_labels = ['P_nu', 'P_nu', 'P_t', 'P_t', 'P_nu', 'P_t']

    plot.plot(x_datas, y_datas, x_labels = x_labels, y_labels = y_labels,
              plot_groups=[0,0,1,1,2,3], opacity=0.3)


    x_data_raw = np.linspace(0.0, length, steps)
    x_data = np.zeros((0, len(x_data_raw)))
    for i in range(nbr_ch_s + 2 + nbr_ch_p):
        x_data = np.vstack((x_data, x_data_raw))

    x_data = [x_data, x_data_raw.reshape((1, -1))]

    power_signal = fiber.power_signal_forward + fiber.power_signal_backward
    power_ase_forward = np.sum(fiber.power_ase_forward, axis=0)
    power_ase_backward = np.sum(fiber.power_ase_backward, axis=0)
    power_pump = fiber.power_pump_forward + fiber.power_pump_backward
    powers = np.vstack((power_signal, power_ase_forward, power_ase_backward,
                        power_pump))
    population = (fiber.N_1/fiber.N_0).reshape((1, -1))

    y_data = [powers, population]

    plot_label = (['Signal' for i in range(nbr_ch_s)]
                   + ['ASE forward', 'ASE backward']
                   + ['Pump' for i in range(nbr_ch_p)])
    plot_labels: List[Optional[List[str]]] = [plot_label, None]

    plot_groups = [0, 1]

    plot_titles = ['Power evolution of signal, ase and pump along the '
                   'fiber amplifier.', 'Inversion of population evolution '
                   'along the fiber amplifier.']

    plot.plot(x_data, y_data, x_labels=['z'], y_labels=['P_t', 'Inversion of '
              'population'], plot_titles=plot_titles,
              plot_labels=plot_labels, plot_groups=plot_groups, opacity=0.3)
