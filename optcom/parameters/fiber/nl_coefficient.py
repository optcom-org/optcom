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

from typing import Callable, List, overload, Union

import numpy as np

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.parameters.abstract_parameter import AbstractParameter
from optcom.utils.callable_container import CallableContainer


class NLCoefficient(AbstractParameter):

    def __init__(self, nl_index: Union[float, Callable],
                 eff_area: Union[float, Callable]) -> None:
        r"""
        Parameters
        ----------
        nl_index :
            The non linear index. :math:`[m^2\cdot W^{-1}]`  If a
            callable is provided, the variable must be angular
            frequency. :math:`[ps^{-1}]`
        eff_area:
            The effective area. :math:`[\mu m^2]`  If a
            callable is provided, the variable must be angular
            frequency. :math:`[ps^{-1}]`

        """
        self._nl_index: Union[float, Callable] = nl_index
        self._eff_area: Union[float, Callable] = eff_area
    # ==================================================================
    @property
    def nl_index(self) -> Union[float, Callable]:

        return self._nl_index
    # ------------------------------------------------------------------
    @nl_index.setter
    def nl_index(self, nl_index: Union[float, Callable]) -> None:

        self._nl_index = nl_index
    # ==================================================================
    @property
    def eff_area(self) -> Union[float, Callable]:

        return self._eff_area
    # ------------------------------------------------------------------
    @eff_area.setter
    def eff_area(self, eff_area: Union[float, Callable]) -> None:

        self._eff_area = eff_area
    # ==================================================================
    @overload
    def __call__(self, omega: float) -> float: ...
    # ------------------------------------------------------------------
    @overload
    def __call__(self, omega: np.ndarray) -> np.ndarray: ...
    # ------------------------------------------------------------------
    def __call__(self, omega):
        r"""Calculate the non linear parameter.

        Parameters
        ----------
        omega :
            The angular frequency.  :math:`[rad\cdot ps^{-1}]`


        Returns
        -------
        :
            Value of the non linear parameter.
            :math:`[rad\cdot W^{-1}\cdot km^{-1}]`

        """
        fct = CallableContainer(NLCoefficient.calc_nl_coefficient,
                                [omega, self._nl_index, self._eff_area])

        return fct(omega)
    # ==================================================================
    # Static methods ===================================================
    # ==================================================================
    @overload
    @staticmethod
    def calc_nl_coefficient(omega: float, nl_index: float, eff_area: float,
                            ) -> float: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def calc_nl_coefficient(omega: np.ndarray, nl_index: np.ndarray,
                            eff_area: np.ndarray) -> np.ndarray: ...
    # ------------------------------------------------------------------
    @staticmethod
    def calc_nl_coefficient(omega, nl_index, eff_area):
        r"""Calculate the non linear parameter. [6]_

        Parameters
        ----------
        omega :
            The center angular frequency.  :math:`[ps^{-1}]`
        nl_index :
            The non linear refractive index. :math:`[m^{2}\cdot W^{-1}]`
        eff_area :
            The effective mode area.  :math:`[\mu m^{2}]`

        Returns
        -------
        :
            Value of the non linear parameter.
            :math:`[rad\cdot W^{-1}\cdot km^{-1}]`

        Notes
        -----

        .. math::  \gamma(\omega_0) = \frac{\omega_0 n_2}{c A_{eff}}

        References
        ----------
        .. [6] Govind Agrawal, Chapter 2: Pulse Propaga(\omega_0)tion in
           Fibers, Nonlinear Fiber Optics (Fifth Edition), Academic
           Press, 2013, Page 38.

        """
        # Unit conversion
        nl_index *=  1e-6  # m^2 W^{-1} -> km^2 W^{-1}
        eff_area *= 1e-18    # um^2 -> km^2
        c = cst.C * 1e-12    # ps/nm -> ps/km

        return (nl_index*omega) / (eff_area*c)
    # ==================================================================
    @overload
    @staticmethod
    def calc_nl_length(power: float, nl_coeff: float) -> float: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def calc_nl_length(power: np.ndarray, nl_coeff: np.ndarray
                       ) -> np.ndarray: ...
    # ------------------------------------------------------------------
    @staticmethod
    def calc_nl_length(power, nl_coeff):
        r"""Calculate the non linear length.

        Parameters
        ----------
        power :
            The power. :math:`[W]`
        nl_coeff :
            The non linear coefficient.
            :math:`[rad\cdot W^{-1}\cdot km^{-1}]`

        Returns
        -------
        :
            The non-linear length :math:`[km]`

        Notes
        -----

        .. math::  L_{NL} = \frac{1}{\gamma P_0}

        """

        return 1.0 / (power * nl_coeff)


if __name__ == "__main__":
    """Plot the non linear coefficient as a function of the wavelength.
    This piece of code is standalone, i.e. can be used in a separate
    file as an example.
    """

    from typing import List

    import numpy as np

    import optcom as oc

    medium: str = "SiO2"
    # With float
    omega: float = oc.lambda_to_omega(1552.0)
    core_radius: float = 5.0
    n_clad: float = 1.44
    sellmeier: oc.Sellmeier = oc.Sellmeier(medium)
    NA_inst: oc.NumericalAperture = oc.NumericalAperture(sellmeier, n_clad)
    v_nbr_inst: oc.VNumber = oc.VNumber(NA_inst, core_radius)
    eff_area_inst: oc.EffectiveArea = oc.EffectiveArea(v_nbr_inst, core_radius)
    nl_ind_inst: oc.NLIndex = oc.NLIndex(medium)
    nl_coeff: oc.NLCoefficient = oc.NLCoefficient(nl_ind_inst, eff_area_inst)
    print(nl_coeff(omega))
    nl_ind: float = nl_ind_inst(omega)
    eff_area: float = eff_area_inst(omega)
    print(oc.NLCoefficient.calc_nl_coefficient(omega, nl_ind, eff_area))
    # With np.ndarray
    lambdas: np.ndarray = np.linspace(900., 1550., 1000)
    omegas: np.ndarray = oc.lambda_to_omega(lambdas)
    res: np.ndarray = nl_coeff(omegas)
    x_labels: List[str] = ['Lambda']
    y_labels: List[str] = ['Non-linear coefficient, '
                           '$\,\gamma\,(rad\cdot W^{-1}\cdot km^{-1})$']
    plot_titles: List[str] = ["Non linear coefficient as a function of the "
                              "wavelength \n for Silica core with constant "
                              "cladding refractive index."]

    oc.plot2d([lambdas], [res], x_labels=x_labels, y_labels=y_labels,
              plot_titles=plot_titles, line_opacities=[0.0],
              y_ranges=[(1., 5.)])
