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

from typing import Callable, List, overload, Union

import numpy as np
from nptyping import Array

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.parameters.abstract_parameter import AbstractParameter


class NLCoefficient(AbstractParameter):

    def __init__(self, nl_index: Union[float, Array[float], Callable],
                 eff_area: Union[float, Array[float], Callable]) -> None:
        """
        Parameters
        ----------
        nl_index :
            The non linear index. :math:`[m^2\cdot W^{-1}]`
        eff_erea:
            The effective area. :math:`[\mu m^2]`

        """
        self._nl_index: Union[float, Array[float], Callable] = nl_index
        self._eff_area: Union[float, Array[float], Callable] = eff_area
    # ==================================================================
    @overload
    def __call__(self, omega: float) -> float: ...
    # ------------------------------------------------------------------
    @overload
    def __call__(self, omega: Array[float]) -> Array[float]: ...
    # ------------------------------------------------------------------
    def __call__(self, omega):
        nl_index = self._nl_index(omega) if callable(self._nl_index)\
            else self._nl_index
        eff_area = self._eff_area(omega) if callable(self._eff_area)\
            else self._eff_area

        return NLCoefficient.calc_nl_coefficient(omega, nl_index, eff_area)
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
    def calc_nl_coefficient(omega: Array[float], nl_index: Array[float],
                            eff_area: Array[float]) -> Array[float]: ...
    # ------------------------------------------------------------------
    @staticmethod
    def calc_nl_coefficient(omega, nl_index, eff_area):
        r"""Calculate the non linear parameter. [2]_

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
        .. [2] Govind Agrawal, Chapter 2: Pulse Propaga(\omega_0)tion in
           Fibers, Nonlinear Fiber Optics (Fifth Edition), Academic
           Press, 2013, Page 38.

        """
        # Unit conversion
        nl_index *=  1e-6  # m^2 W^{-1} -> km^2 W^{-1}
        eff_area *= 1e-18    # um^2 -> km^2
        c = cst.LIGHT_SPEED * 1e-12    # ps/nm -> ps/km

        return (nl_index*omega) / (eff_area*c)


if __name__ == "__main__":

    import numpy as np

    import optcom.utils.plot as plot
    from optcom.domain import Domain
    from optcom.parameters.fiber.effective_area import EffectiveArea
    from optcom.parameters.fiber.nl_index import NLIndex

    medium = "SiO2"
    # With float
    omega = Domain.lambda_to_omega(1552.0)
    core_radius = 5.0
    n_core = 1.43
    n_clad = 1.425

    eff_area = EffectiveArea.calc_effective_area(omega,
                                                 core_radius=core_radius,
                                                 n_core=n_core, n_clad=n_clad)
    nl_index = NLIndex.calc_nl_index(omega, medium=medium)
    print(NLCoefficient.calc_nl_coefficient(omega, nl_index=nl_index,
                                            eff_area=eff_area))

    # With numpy ndarray
    lambdas = np.linspace(900, 1550, 10)
    omegas = Domain.lambda_to_omega(lambdas)
    n_core = np.linspace(1.42, 1.43, 10)
    n_clad = np.linspace(1.415, 1.425, 10)

    eff_area = EffectiveArea.calc_effective_area(omegas,
                                                 core_radius=core_radius,
                                                 n_core=n_core, n_clad=n_clad)
    nl_index = NLIndex.calc_nl_index(omega, medium=medium)
    res = NLCoefficient.calc_nl_coefficient(omegas, nl_index=nl_index,
                                            eff_area=eff_area)

    x_labels = ['Lambda']
    y_labels = ['gamma']
    plot_titles = ["Non linear coefficient of Silica"]

    plot.plot(lambdas, res, x_labels=x_labels, y_labels=y_labels,
              plot_titles=plot_titles, opacity=0.0)
