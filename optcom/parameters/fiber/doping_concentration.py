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


class DopingConcentration(AbstractParameter):

    def __init__(self, alpha: Union[float, np.ndarray, Callable],
                 sigma_a: Union[float, np.ndarray, Callable],
                 overlap: Union[float, np.ndarray, Callable], length: float
                 ) -> None:
        r"""
        Parameters
        ----------
        alpha :
            The doped area absorption. :math:`[dB m^{-1}]` If a callable
            is provided, the variable must be angular frequency.
            :math:`[ps^{-1}]`
        sigma_a :
            The absorption cross sections. :math:`[nm^2]`  If a
            callable is provided, the variable must be angular
            frequency. :math:`[ps^{-1}]`
        overlap :
            The overlap factors. If a callable is provided, the variable
            must be angular frequency. :math:`[ps^{-1}]`
        length :
            The length of the fiber. :math:`[km]`

        """
        self._alpha: Union[float, np.ndarray, Callable] = alpha
        self._sigma_a: Union[float, np.ndarray, Callable] = sigma_a
        self._overlap: Union[float, np.ndarray, Callable] = overlap
        self._length: float = length
    # ==================================================================
    @overload
    def __call__(self, omega: float) -> float: ...
    # ------------------------------------------------------------------
    @overload
    def __call__(self, omega: np.ndarray) -> np.ndarray: ...
    # ------------------------------------------------------------------
    def __call__(self, omega):
        r"""Compute the doping concentration.

        Parameters
        ----------
        omega :
            The angular frequency. :math:`[rad\cdot ps^{-1}]`

        Returns
        -------
        :
            The value of the doping concentration. :math:`[nm^{-3}]`

        """
        fct = CallableContainer(DopingConcentration.calc_doping_concentration,
                                [omega, self._alpha, self._sigma_a,
                                 self._overlap, self._length])

        return fct(omega)
    # ==================================================================
    @overload
    @staticmethod
    def calc_doping_concentration(omega: float, alpha: float, sigma_a: float,
                                  overlap: float, length: float) -> float: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def calc_doping_concentration(omega: np.ndarray, alpha: np.ndarray,
                                  sigma_a: np.ndarray, overlap: np.ndarray,
                                  length: float) -> np.ndarray: ...
    # ------------------------------------------------------------------
    @staticmethod
    def calc_doping_concentration(omega, alpha, sigma_a, overlap, length):
        r"""Calculate the doping concentration.

        Parameters
        ----------
        omega :
            The angular frequency. :math:`[rad\cdot ps^{-1}]`
        alpha :
            The doped area absorption. :math:`[dB m^{-1}]`
        sigma_a :
            The absorption cross sections. :math:`[nm^2]`
        overlap :
            The overlap factors.
        length :
            The fiber length. :math:`[km]`

        Returns
        -------
        :
            The value of the doping concentration. :math:`[nm^{-3}]`

        Notes
        -----

        .. math:: N_T = \frac{\text{ln}\Big(10^{\frac{\alpha*L}{10}}
                        \Big)}{\Gamma \sigma_a L}

        """
        alpha *= 1e3    # db m^-1 -> dB km^-1
        if (isinstance(alpha, float)):
            num = math.log(10**(alpha*length*0.1))
        else:
            num = np.log(np.power(10., alpha*length*0.1))
        den = overlap * sigma_a * length * 1e12  # km -> nm
        if (isinstance(den, float)):
            if (den):
                res = num / den
            else:
                res = math.inf
        else:
            res = np.divide(num, den, where=den!=0)
            res[den==0] = np.inf

        return res



if __name__ == "__main__":
    """Plot doping concentration as a function of the wavelength.
    This piece of code is standalone, i.e. can be used in a separate
    file as an example.
    """

    from typing import List

    import numpy as np

    import optcom as oc

    length = 0.0015    # km
    alpha = 250.0    # dB m^-1
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
    N_T_inst: oc.DopingConcentration = oc.DopingConcentration(alpha,
                                                              sigma_a_inst,
                                                              of_inst, length)
    print(N_T_inst(omega))
    A_eff: float = A_eff_inst(omega)
    sigma_a: float = sigma_a_inst(omega)
    sigma_e: float = sigma_e_inst(omega)
    of: float = of_inst(omega)
    print(oc.DopingConcentration.calc_doping_concentration(omega, alpha,
                                                           sigma_a, of,
                                                           length))
    # With np.ndarray
    lambdas: np.ndarray = np.linspace(900, 1050, 1000)
    omegas: np.ndarray = oc.lambda_to_omega(lambdas)
    ens_sat: np.ndarray = N_T_inst(omegas)
    plot_titles: List[str] = ["Doping concentration as a function of the "
                              "wavelenght for Ytterbium doped fiber."]

    oc.plot2d([lambdas], [ens_sat], x_labels=['Lambda'],
              y_labels=[r'Doping concentration, $\,N_T\,(nm^{-3})$'],
              split=True, line_colors=['red'], plot_titles=plot_titles,
              line_styles=['-.'], line_opacities=[0.0])
