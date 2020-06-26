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


class EnergySaturation(AbstractParameter):

    def __init__(self, eff_area: Union[float, np.ndarray, Callable],
                 sigma_a: Union[float, np.ndarray, Callable],
                 sigma_e: Union[float, np.ndarray, Callable],
                 overlap: Union[float, np.ndarray, Callable]) -> None:
        r"""
        Parameters
        ----------
        eff_area :
            The effective area. :math:`[\mu m^2]`  If a callable is
            provided, the variable must be angular frequency.
            :math:`[ps^{-1}]`
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

        """
        self._eff_area: Union[float, np.ndarray, Callable] = eff_area
        self._sigma_a: Union[float, np.ndarray, Callable] = sigma_a
        self._sigma_e: Union[float, np.ndarray, Callable] = sigma_e
        self._overlap: Union[float, np.ndarray, Callable] = overlap
    # ==================================================================
    @overload
    def __call__(self, omega: float) -> float: ...
    # ------------------------------------------------------------------
    @overload
    def __call__(self, omega: np.ndarray) -> np.ndarray: ...
    # ------------------------------------------------------------------
    def __call__(self, omega):
        r"""Compute the energy saturation.

        Parameters
        ----------
        omega :
            The angular frequency. :math:`[rad\cdot ps^{-1}]`

        Returns
        -------
        :
            The value of the energy saturation. :math:`[J]`

        """
        fct = CallableContainer(EnergySaturation.calc_energy_saturation,
                                [omega, self._eff_area, self._sigma_a,
                                 self._sigma_e, self._overlap])

        return fct(omega)
    # ==================================================================
    @overload
    @staticmethod
    def calc_energy_saturation(omega: float, eff_area: float, sigma_a: float,
                               sigma_e: float, overlap: float) -> float: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def calc_energy_saturation(omega: np.ndarray, eff_area: np.ndarray,
                               sigma_a: np.ndarray, sigma_e: np.ndarray,
                               overlap: np.ndarray) -> np.ndarray: ...
    # ------------------------------------------------------------------
    @staticmethod
    def calc_energy_saturation(omega, eff_area, sigma_a, sigma_e, overlap):
        r"""Calculate the energy saturation.

        Parameters
        ----------
        omega :
            The angular frequency. :math:`[rad\cdot ps^{-1}]`
        eff_area :
            The effective area. :math:`[\mu m^2]`
        sigma_a :
            The absorption cross sections. :math:`[nm^2]`
        sigma_e :
            The emission cross sections. :math:`[nm^2]`
        overlap :
            The overlap factors.

        Returns
        -------
        :
            Value of the energy saturation. :math:`[J]`

        Notes
        -----

        .. math:: E_{sat} = \frac{A_{eff}h\omega_0}{2\pi\Gamma
                  \big(\sigma_e(\lambda_0)+\sigma_a(\lambda_0)\big)}

        """
        eff_area *= 1e6 # um^2 -> nm^2
        num = eff_area * cst.HBAR * omega
        den = overlap * (sigma_a+sigma_e)
        if (isinstance(den, float)):
            if (den):
                res = num / den
            else:
                res = math.inf
        else:
            res = np.divide(num, den, where=den!=0)
            res[den==0] = np.inf
        res *= 1e6 # nm^2 kg ps^{-2} -> m^2 kg s^{-2} = J

        return res



if __name__ == "__main__":
    """Plot energy saturation as a function of the wavelength.
    This piece of code is standalone, i.e. can be used in a separate
    file as an example.
    """

    from typing import List

    import numpy as np

    import optcom as oc

    medium: str = "sio2"
    dopant: str = "yb"
    A_doped: float = oc.PI*25.0
    # With float
    omega: float = oc.lambda_to_omega(1000)
    core_radius: float = 5.0
    n_core: oc.Sellmeier = oc.Sellmeier(medium)
    n_clad: float = 1.44
    NA_inst: oc.NumericalAperture = oc.NumericalAperture(n_core, n_clad)
    v_nbr_inst: oc.VNumber = oc.VNumber(NA_inst, core_radius)
    A_eff_inst: oc.EffectiveArea = oc.EffectiveArea(v_nbr_inst, core_radius)
    of_inst: oc.OverlapFactor = oc.OverlapFactor(A_eff_inst, A_doped)
    sigma_a_inst: oc.AbsorptionSection = oc.AbsorptionSection(dopant=dopant,
                                                              medium=medium)
    T: float = 293.1
    sigma_e_inst: oc.EmissionSection = oc.EmissionSection(dopant=dopant,
                                                          medium=medium, T=T,
                                                          sigma_a=sigma_a_inst)
    en_sat: oc.EnergySaturation = oc.EnergySaturation(A_eff_inst, sigma_a_inst,
                                                      sigma_e_inst, of_inst)
    print(en_sat(omega))
    A_eff: float = A_eff_inst(omega)
    sigma_a: float = sigma_a_inst(omega)
    sigma_e: float = sigma_e_inst(omega)
    of: float = of_inst(omega)
    print(oc.EnergySaturation.calc_energy_saturation(omega, A_eff, sigma_a,
                                                    sigma_e, of))
    # With np.ndarray
    lambdas: np.ndarray = np.linspace(900, 1050, 1000)
    omegas: np.ndarray = oc.lambda_to_omega(lambdas)
    ens_sat: np.ndarray = en_sat(omegas)
    plot_titles: List[str] = ["Energy saturation as a function of the "
                              "wavelenght for Ytterbium doped fiber."]

    oc.plot2d([lambdas], [ens_sat], x_labels=['Lambda'],
              y_labels=[r'Energy saturation, $\,E_{sat}\,(J)$'],
              split=True, line_colors=['red'], plot_titles=plot_titles,
              line_styles=['-.'], line_opacities=[0.0])
