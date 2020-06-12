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
from optcom.components.abstract_fiber_amp_2levels import\
    AbstractFiberAmp2Levels
from optcom.domain import Domain
from optcom.equations.abstract_ampnlse import SEED_SPLIT
from optcom.field import Field


default_name = 'Ytterbium Fiber'
TAYLOR_COEFF_TYPE_OPTIONAL = List[Union[List[float], Callable, None]]
FLOAT_COEFF_TYPE_OPTIONAL = List[Union[float, Callable, None]]


class FiberYb(AbstractFiberAmp2Levels):
    r"""A non ideal Ytterbium Fiber Amplifier.

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

    _nbr_instances: int = 0
    _nbr_instances_with_default_name: int = 0

    def __init__(self, name: str = default_name, length: float = 1.0,
                 alpha: TAYLOR_COEFF_TYPE_OPTIONAL = [None],
                 alpha_order: int = 0,
                 beta: TAYLOR_COEFF_TYPE_OPTIONAL = [None],
                 beta_order: int = 2, gain_order: int = 0,
                 gamma: FLOAT_COEFF_TYPE_OPTIONAL = [None],
                 sigma: float = cst.XPM_COEFF,
                 eta: float = cst.XNL_COEFF, T_R: float = cst.RAMAN_COEFF,
                 h_R: Optional[Union[float, Callable]] = None,
                 f_R: float = cst.F_R,
                 n_core: FLOAT_COEFF_TYPE_OPTIONAL = [None],
                 n_clad: FLOAT_COEFF_TYPE_OPTIONAL = [None],
                 NA: FLOAT_COEFF_TYPE_OPTIONAL = [None],
                 v_nbr: FLOAT_COEFF_TYPE_OPTIONAL = [None],
                 eff_area: FLOAT_COEFF_TYPE_OPTIONAL = [None],
                 nl_index: FLOAT_COEFF_TYPE_OPTIONAL = [None],
                 overlap: FLOAT_COEFF_TYPE_OPTIONAL = [None],
                 sigma_a: FLOAT_COEFF_TYPE_OPTIONAL = [None],
                 sigma_e: FLOAT_COEFF_TYPE_OPTIONAL = [None],
                 en_sat: FLOAT_COEFF_TYPE_OPTIONAL = [None],
                 doped_area: Optional[float] = None,
                 tau: float = cst.TAU_META_YB, N_T: float = cst.N_T,
                 core_radius: float = cst.CORE_RADIUS,
                 clad_radius: float = cst.CLAD_RADIUS,
                 R_0: Union[float, Callable] = cst.R_0,
                 R_L: Union[float, Callable] = cst.R_L,
                 medium_core: str = cst.FIBER_MEDIUM_CORE,
                 medium_clad: str = cst.FIBER_MEDIUM_CLAD,
                 temperature: float = cst.TEMPERATURE, nl_approx: bool = True,
                 RESO_INDEX: bool = True, CORE_PUMPED: bool = True,
                 CLAD_PUMPED: bool = False, ATT: List[bool] = [True, True],
                 DISP: List[bool] = [True, False],
                 SPM: List[bool] = [True, False],
                 XPM: List[bool] = [False, False],
                 FWM: List[bool] = [False, False],
                 SS: List[bool] = [False, False],
                 RS: List[bool] = [False, False],
                 XNL: List[bool] = [False, False],
                 GAIN_SAT: bool = True, NOISE: bool = True,
                 approx_type: int = cst.DEFAULT_APPROX_TYPE,
                 split_noise_option: str = SEED_SPLIT,
                 noise_ode_method: str = 'rk4',
                 UNI_OMEGA: List[bool] = [True, True],
                 STEP_UPDATE: bool = False, INTRA_COMP_DELAY: bool = True,
                 INTRA_PORT_DELAY: bool = True, INTER_PORT_DELAY: bool = False,
                 REFL_SEED: bool = False, REFL_PUMP: bool = True,
                 PRE_PUMP_PROP: bool = False, nlse_method: str = "rk4ip",
                 step_method: str = "fixed", steps: int = 100,
                 max_nbr_iter: int = 100, error: float = 1e-2,
                 PROP_PUMP: bool = False, PROP_REFL: bool = False,
                 BISEED: bool = False, BIPUMP: bool = False,
                 save: bool = False, save_all: bool = False,
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
        super().__init__(name, default_name, length, alpha, alpha_order, beta,
                         beta_order, gain_order, gamma, sigma, eta, T_R, h_R,
                         f_R, n_core, n_clad, NA, v_nbr, eff_area, nl_index,
                         overlap, sigma_a, sigma_e, en_sat, doped_area, tau,
                         N_T, core_radius, clad_radius, R_0, R_L, medium_core,
                         medium_clad, temperature, nl_approx, RESO_INDEX,
                         CORE_PUMPED, CLAD_PUMPED, ATT, DISP, SPM, XPM, FWM,
                         SS, RS, XNL, GAIN_SAT, NOISE, approx_type,
                         split_noise_option, noise_ode_method, UNI_OMEGA,
                         STEP_UPDATE, INTRA_COMP_DELAY, INTRA_PORT_DELAY,
                         INTER_PORT_DELAY, REFL_SEED, REFL_PUMP,
                         PRE_PUMP_PROP, nlse_method, step_method, steps,
                         max_nbr_iter, error, PROP_PUMP, PROP_REFL,
                         BISEED, BIPUMP, save, save_all, max_nbr_pass,
                         pre_call_code, post_call_code)


if __name__ == "__main__":
    """Give an example of FiberYb usage.
    This piece of code is standalone, i.e. can be used in a separate
    file as an example.
    """

    from typing import Callable, List, Optional, Union

    import matplotlib.pyplot as plt
    import numpy as np

    import optcom as oc

    noise_samples = 250
    noise_range = (950.0,1100.0)
    domain = oc.Domain(samples_per_bit=1024, bit_width=20.0,
                       memory_storage=8.0, noise_range=noise_range,
                       noise_samples=noise_samples)

    lt: oc.Layout = oc.Layout(domain)

    # Can play with number of channels, depending on the number and your
    # computer specs, it may take a while.
    nbr_ch_s: int = 2
    nbr_ch_p: int = 1

    or_p: float = 1e0
    lambdas_s: List[float] = [1030.0, 1020.0, 1025.0, 1015., 1010., 1100.]
    fwhm_s: List[float] =  [1.0 for i in range(nbr_ch_s)]
    peak_power_s: List[float] = [7.0, 2.0, 1.0, 0.5, 0.3, 0.1]
    rep_freq = [20e-6]   # THz
    pulse: oc.Gaussian = oc.Gaussian(channels=nbr_ch_s,
                                     peak_power=peak_power_s,
                                     fwhm=fwhm_s, center_lambda=lambdas_s,
                                     rep_freq=rep_freq)

    lambdas_p: List[float] = [976.0, 940.0]
    peak_power_p: List[float] = [0.05, 0.1]
    pump: oc.CW = oc.CW(channels=nbr_ch_p, peak_power=peak_power_p,
                        center_lambda=lambdas_p)

    steps: int = 100
    length: float = 0.002   # km

    # Values taken partially from:
    # https://www.osapublishing.org/oe/abstract.cfm?uri=oe-15-6-3236
    sigma_a_s: float = 6.4e-9   # nm^{2}
    sigma_a_p: float = 2.5e-6   # nm^{2}
    sigma_a: List[Union[float, Callable, None]] = [sigma_a_s, sigma_a_p]
    sigma_e_s: float = 3.2e-7   # nm^{2}
    sigma_e_p: float = 2.5e-6   # nm^{2}
    sigma_e: List[Union[float, Callable, None]] = [sigma_e_s, sigma_e_p]
    NA: List[Union[float, Callable, None]] = [0.06]
    r_core: float = 6.0 / 2.0   # um
    r_clad: float = 125.0 / 2.0  # um
    temperature: float = 293.15    # K
    tau: float = 840.0    # us
    N_T: float = 4.6e-2    # nm^{-3}
    alpha_s: float = 2.0     # km^-1
    alpha_p: float = 0.4     # km^-1
    alpha: List[Union[List[float], Callable, None]] = [[alpha_s], [alpha_p]]
    R_0: float = 8e-4
    R_L: float = R_0
    medium_core: str = 'sio2'
    error: float = 1e-5
    max_nbr_iter: int = 10
    fiber: oc.FiberYb
    fiber = oc.FiberYb(length=length, nlse_method="rk4ip",
                        alpha=alpha, beta_order=2, alpha_order=0,
                        gain_order=0, GAIN_SAT=True,
                        nl_approx=False, ATT=[True, True], DISP=[True, False],
                        SPM=[True, False], XPM=[True, False], SS=[True, False],
                        RS=[True, False], XNL=[False, False],
                        approx_type=1, split_noise_option='seed_split',
                        sigma_a=sigma_a, sigma_e=sigma_e, NA=NA,
                        core_radius=r_core, clad_radius=r_clad,
                        temperature=temperature, tau=tau,
                        N_T=N_T, R_0=R_0, R_L=R_L,
                        medium_core=medium_core, steps=steps,
                        error=error, BISEED=False, BIPUMP=False,
                        PROP_PUMP=True, CORE_PUMPED=True, CLAD_PUMPED=False,
                        UNI_OMEGA=[True, True], save=True, save_all=True,
                        STEP_UPDATE=False, NOISE=True, INTRA_COMP_DELAY=False,
                        INTRA_PORT_DELAY=False, INTER_PORT_DELAY=False,
                        REFL_SEED=True, REFL_PUMP=True, noise_ode_method='rk4',
                        max_nbr_iter=max_nbr_iter, PROP_REFL=False,
                        PRE_PUMP_PROP=False, RESO_INDEX=True)

    lt.add_links((pulse[0], fiber[0]), (pump[0], fiber[2]))
    lt.run(pulse, pump)

    # Power vs time plotting -------------------------------------------
    x_datas: List[np.ndarray] = [pulse[0][0].nu, pump[0][0].nu,
                            pulse[0][0].time, pump[0][0].time,
                            np.vstack((fiber[1][0].nu, fiber[1][0].nu)),
                            np.vstack((fiber[1][0].time, fiber[1][0].time))]
    y_datas: List[np.ndarray] = [oc.spectral_power(pulse[0][0].channels),
                                 oc.spectral_power(pump[0][0].channels),
                                 oc.temporal_power(pulse[0][0].channels),
                                 oc.temporal_power(pump[0][0].channels),
                                 oc.spectral_power(np.vstack((
                                    fiber[1][0].channels,
                                    fiber[1][1].channels))),
                                 oc.temporal_power(np.vstack((
                                    fiber[1][0].channels,
                                    fiber[1][1].channels)))]
    x_labels: List[str] = ['nu', 't', 'nu', 't']
    y_labels: List[str] = ['P_nu', 'P_t', 'P_nu', 'P_t']
    plot_titles: Optional[List[str]]
    plot_titles = ['Initial pulse', 'Initial pulse', 'Pulse after amplifier',
                    'Pulse after amplifier']

    oc.plot2d(x_datas, y_datas, x_labels=x_labels, y_labels=y_labels,
              plot_groups=[0,0,1,1,2,3], line_opacities=[0.3],
              plot_titles=plot_titles)

    # Power vs space plotting -------------------------------------------
    # Storage stores first all input seeds and pumps and then their reflections
    storage = fiber.storage
    if (storage is not None):
        steps_ = storage.steps
        nbr_channels_ = storage.nbr_channels
        # Preparing pumps and seeds
        x_datas = [storage.space]
        y_datas = [np.zeros(steps_) for _ in range(nbr_channels_)]
        for i in range(steps_):
            for j in range(nbr_channels_):
                y_datas[j][i] = oc.average_power(storage.channels[j][i],
                                                 domain.dtime,
                                                 storage.rep_freq[j])
        # Prepare noises - have been split among all seeds
        ase_forward = np.zeros(steps_)
        ase_backward = np.zeros(steps_)
        for i in range(len(ase_forward)):
            ase_forward[i] += np.sum(storage.noises[0][i])
            ase_backward[i] += np.sum(storage.noises[2][i])
        y_datas.extend([ase_forward, ase_backward])
        # Preparing labels and other plotting parameters
        line_labels: List[Optional[str]]
        line_labels = ["Seed {}".format(i+1) for i in range(nbr_ch_s)]
        line_labels.extend(["Pump {}".format(i+1) for i in range(nbr_ch_p)])
        line_labels.extend(["Reflection seed {}"
                            .format(i+1) for i in range(nbr_ch_s)])
        line_labels.extend(["Reflection pump {}"
                          .format(i+1) for i in range(nbr_ch_p)])
        line_labels.extend(['ASE forward', 'ASE backward'])
        line_styles: List[str]
        line_styles = ['-' for i in range(nbr_ch_s)]
        line_styles.extend(['-.' for i in range(nbr_ch_p)])
        line_styles.extend(['-' for i in range(nbr_ch_s)])
        line_styles.extend(['-.' for i in range(nbr_ch_p)])
        line_styles.extend([':' for i in range(2)])
        plot_titles = ['Power evolution of signal, ase and pump along the '
                       'fiber amplifier.']

        oc.plot2d(x_datas, y_datas, x_labels=[r'Fiber length, $\, z\,(km)$'],
                  y_labels=['Average Power (W)'], line_labels=line_labels,
                  line_styles=line_styles, split=False,
                  plot_titles=plot_titles)

        # Animation power vs space plotting ----------------------------
        line_labels = [str(lambdas_s[i]) + ' nm channel '
                       for i in range(len(lambdas_s))]

        oc.animation2d(storage.time,
                       oc.temporal_power(storage.channels[:(nbr_ch_s)]),
                       storage.space, x_label='t', y_label='P_t',
                       plot_title='Channels propagation in fiber amplifier.',
                       line_labels=line_labels)
