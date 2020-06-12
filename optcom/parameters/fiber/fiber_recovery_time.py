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
from typing import Callable, List, Optional, overload, Union

import numpy as np

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.parameters.abstract_parameter import AbstractParameter
from optcom.utils.callable_container import CallableContainer


class FiberRecoveryTime(AbstractParameter):
    r"""Compute the recovery time of a doped fiber at steday
    state. [13]_

    References
    ----------
    .. [13] Lindberg, R., Zeil, P., Malmström, M., Laurell, F. and
            Pasiskevicius, V., 2016. Accurate modeling of
            high-repetition rate ultrashort pulse amplification in
            optical fibers. Scientific reports, 6, p.34742.

    """

    def __init__(self, core_area: float,
                 sigma_a: Union[float, np.ndarray, Callable],
                 sigma_e: Union[float, np.ndarray, Callable],
                 overlap: Union[float, np.ndarray, Callable],
                 power: Union[float, np.ndarray, Callable], tau: float
                 ) -> None:
        r"""
        Parameters
        ----------
        core_area :
            The fiber core area. :math:`[\mu m^2]`
        sigma_a :
            The absorption cross sections. :math:`[nm^2]`  If a
            callable is provided, the variable must be angular
            frequency. :math:`[ps^{-1}]`
        sigma_e :
            The emission cross sections. :math:`[nm^2]`  If a callable
            is provided, the variable must be angular frequency.
            :math:`[ps^{-1}]`
        overlap :
            The overlap factors. If a callable is provided, the variable
            must be angular frequency. :math:`[ps^{-1}]`
        power :
            The pump powers. :math:`[W]`
        tau :
            The lifetime of the metastable level. :math:`[\mu s]`

        """
        self._core_area: float = core_area
        self._sigma_a: Union[float, np.ndarray, Callable] = sigma_a
        self._sigma_e: Union[float, np.ndarray, Callable] = sigma_e
        self._overlap: Union[float, np.ndarray, Callable] = overlap
        self._power: Union[float, np.ndarray, Callable] = power
        self._tau: float = tau
    # ==================================================================
    @property
    def power(self) -> Union[float, np.ndarray, Callable]:

        return self._power
    # ------------------------------------------------------------------
    @power.setter
    def power(self, power: Union[float, np.ndarray, Callable]) -> None:

        self._power = power
    # ==================================================================
    @overload
    def __call__(self, omega: float) -> float: ...
    # ------------------------------------------------------------------
    @overload
    def __call__(self, omega: np.ndarray) -> np.ndarray: ...
    # ------------------------------------------------------------------
    def __call__(self, omega):
        r"""Compute the recovery time.

        Parameters
        ----------
        omega :
            The angular frequency. :math:`[rad\cdot ps^{-1}]`

        Returns
        -------
        :
            The value of the recovery time. :math:`[\mu s]`

        """
        fct = CallableContainer(FiberRecoveryTime.calc_recovery_time,
                                [omega, self._core_area, self._sigma_a,
                                 self._sigma_e, self._overlap, self._power,
                                 self._tau])

        return fct(omega)
    # ==================================================================
    @overload
    @staticmethod
    def calc_recovery_time(omega: float, core_area: float, sigma_a: float,
                           sigma_e: float, overlap: float, power: float,
                           tau: float) -> float: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def calc_recovery_time(omega: np.ndarray, core_area: float,
                           sigma_a: np.ndarray, sigma_e: np.ndarray,
                           overlap: np.ndarray, power: np.ndarray, tau: float
                           ) -> np.ndarray: ...
    # ------------------------------------------------------------------
    @staticmethod
    def calc_recovery_time(omega, core_area, sigma_a, sigma_e, overlap,
                           power, tau):
        r"""Calculate the recovery time.

        Parameters
        ----------
        omega :
            The angular frequency. :math:`[rad\cdot ps^{-1}]`
        core_area :
            The area of the core. :math:`[\mu m^2]`
        sigma_a :
            The absorption cross sections. :math:`[nm^2]`
        sigma_e :
            The emission cross sections. :math:`[nm^2]`
        overlap :
            The overlap factors.
        power :
            The pump powers. :math:`[W]`
        tau :
            The lifetime of the metastable level. :math:`[\mu s]`

        Returns
        -------
        :
            Value of the recovery time. :math:`[\mu s]`

        Notes
        -----

        .. math:: t_c = \frac{1}{\frac{2\pi\Gamma}{h \omega A_c}
                         \Big(\sigma_a(\omega)+\sigma_e(\omega)\Big)
                         P_p + \frac{1}{z}}

        """
        factor = (2*cst.PI*overlap) / (cst.H * omega)
        first_term = factor * (sigma_a + sigma_e) * power
        first_term *= 1e-18 # ps^2 m^2 um^{-2} s^{-3} -> us^{-1}
        second_term = 1.0 / tau

        return 1.0 / (first_term + second_term)


if __name__ == "__main__":
    """Plot recovery time as a function of the wavelength.
    This piece of code is standalone, i.e. can be used in a separate
    file as an example.
    """

    from typing import List

    import numpy as np

    import optcom as oc

    medium: str = "sio2"
    dopant: str = "yb"
    A_doped: float = oc.PI*25.0   # um^2
    A_core: float = A_doped         # um^2
    P_pump: float = 0.001 # W
    tau: float = 840.0  # us
    # With float
    omega: float = oc.lambda_to_omega(1000)
    core_radius: float = 5.0    # um
    n_core: oc.Sellmeier = oc.Sellmeier(medium)
    n_clad: float = 1.44
    NA_inst: oc.NumericalAperture = oc.NumericalAperture(n_core, n_clad)
    v_nbr_inst: oc.VNumber = oc.VNumber(NA_inst, core_radius)
    A_eff_inst: oc.EffectiveArea = oc.EffectiveArea(v_nbr_inst, core_radius)
    of_inst: oc.OverlapFactor = oc.OverlapFactor(A_eff_inst, A_doped)
    sigma_a_inst: oc.AbsorptionSection = oc.AbsorptionSection(dopant=dopant,
                                                              medium=medium)
    T: float = 293.1   # K
    sigma_e_inst: oc.EmissionSection = oc.EmissionSection(dopant=dopant,
                                                          medium=medium, T=T,
                                                          sigma_a=sigma_a_inst)
    recov_t: oc.FiberRecoveryTime = oc.FiberRecoveryTime(A_core, sigma_a_inst,
                                                         sigma_e_inst, of_inst,
                                                         P_pump, tau)
    print(recov_t(omega))
    A_eff: float = A_eff_inst(omega)
    sigma_a: float = sigma_a_inst(omega)
    sigma_e: float = sigma_e_inst(omega)
    of: float = of_inst(omega)
    print(oc.FiberRecoveryTime.calc_recovery_time(omega, A_core, sigma_a,
                                                  sigma_e, of, P_pump, tau))
    # With np.ndarray
    lambdas: np.ndarray = np.linspace(900, 1050, 1000)
    omegas: np.ndarray = oc.lambda_to_omega(lambdas)
    ens_sat: np.ndarray = recov_t(omegas)
    plot_titles: List[str] = ["Recovery time as a function of the "
                              "wavelenght for Ytterbium doped fiber."]

    oc.plot2d([lambdas], [ens_sat], x_labels=['Lambda'],
              y_labels=[r'Recovery time, $\,t_c\,(\mu s)$'],
              split=True, line_colors=['red'], plot_titles=plot_titles,
              line_styles=['-.'], line_opacities=[0.0])
