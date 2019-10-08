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
from typing import overload

import numpy as np
from nptyping import Array

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.domain import Domain
from optcom.equations.abstract_equation import AbstractEquation


class McCumber(AbstractEquation):
    r"""McCumber relations.

    Represent McCumber relations which give a relationship between the
    effective cross-sections of absorption and emission of light in the
    physics of solid-state lasers. [2]_

    Notes
    -----

    .. math:: \frac{\sigma_e(\omega)}{\sigma_a(\omega)}
              \exp\Big[\frac{\hbar \omega}{k_B T}\Big]
              = \Big(\frac{N_0}{N_1}\Big)_T

    where :math:`\Big(\frac{N_0}{N_1}\Big)_T` is the thermal
    steady-state ratio of populations at temperature :math:`T`.

    References
    ----------
    .. [2] Miniscalco, W.J. and Quimby, R.S., 1991. General procedure
           for the analysis of Er 3+ cross sections. Optics letters,
           16(4), pp.258-260.

    """

    def __init__(self):

        return None
    # ==================================================================
    @overload
    @staticmethod
    def calc_cross_section_absorption(sigma_e: float, omega: float,
                                      center_omega: float, N_0: float,
                                      N_1: float, T: float) -> float: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def calc_cross_section_absorption(sigma_e: Array[float],
                                      omega: Array[float], center_omega: float,
                                      N_0: float, N_1: float, T: float
                                      ) -> Array[float]: ...
    # ------------------------------------------------------------------
    @staticmethod
    def calc_cross_section_absorption(sigma_e, omega, center_omega, N_0, N_1,
                                      T):
        r"""Calcul the absorption cross section.

        Parameters
        ----------
        sigma_e :
            The emission cross section. :math:`[nm^2]`
        omega :
            The angular frequency.  :math:`[rad\cdot ps^{-1}]`
        center_omega :
            The center angular frequency.  :math:`[rad\cdot ps^{-1}]`
        N_0 :
            The population in ground state. :math:`[nm^{-3}]`
        N_1 :
            The population in the excited state. :math:`[nm^{-3}]`
        T :
            The temperature. :math:`[K]`

        Returns
        -------
        :
            Value of the absorption cross section. :math:`[nm^2]`

        Notes
        -----

        .. math:: \sigma_a(\omega) = \sigma_e(\omega)
                  \exp\Big[\frac{\hbar (\omega -\omega_0}{k_B T}\Big]
                  \Big(\frac{N_0}{N_1}\Big)_T^{-1}

        """
        inv_pop_ratio = N_1 / N_0

        KB = cst.KB
        HBAR = cst.HBAR

        omega_red = omega - center_omega

        if (isinstance(omega, float)):

            return (sigma_e * math.exp(HBAR*omega_red/(KB*T)) *inv_pop_ratio)
        else:

            return (sigma_e * np.exp(HBAR*omega_red/(KB*T)) * inv_pop_ratio)
    # ==================================================================
    @overload
    @staticmethod
    def calc_cross_section_emission(sigma_a: float, omega: float,
                                    center_omega: float, N_0: float,
                                    N_1: float, T: float) -> float: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def calc_cross_section_emission(sigma_a: Array[float], omega: Array[float],
                                    center_omega: float, N_0: float,
                                    N_1: float, T: float) -> Array[float]: ...
    # ------------------------------------------------------------------
    @staticmethod
    def calc_cross_section_emission(sigma_a, omega, center_omega, N_0, N_1, T):
        r"""Calcul the emission cross section.

        Parameters
        ----------
        sigma_a :
            The absorption cross section. :math:`[nm^2]`
        omega :
            The angular frequency.  :math:`[rad\cdot ps^{-1}]`
        center_omega :
            The center angular frequency.  :math:`[rad\cdot ps^{-1}]`
        N_0 :
            The population in ground state. :math:`[nm^{-3}]`
        N_1 :
            The population in the excited state. :math:`[nm^{-3}]`
        T :
            The temperature. :math:`[K]`

        Returns
        -------
        :
            Value of the absorption cross section. :math:`[nm^2]`

        Notes
        -----

        .. math:: \sigma_e(\omega) = \sigma_a(\omega)
                  \exp\Big[-\frac{\hbar (\omega - \omega_0)}{k_B T}\Big]
                  \Big(\frac{N_0}{N_1}\Big)_T

        """
        pop_ratio = N_0 / N_1

        KB = cst.KB
        HBAR = cst.HBAR

        omega_red = omega - center_omega

        if (isinstance(omega, float)):

            return (sigma_a * math.exp(-HBAR*omega_red/(KB*T)) * pop_ratio)
        else:

            return (sigma_a * np.exp(-HBAR*omega_red/(KB*T)) * pop_ratio)


if __name__ == "__main__":

    # With float
    omega = 1220
    center_omega = 1150
    sigma_a = 50
    sigma_e = 25
    N_0 = 1
    N_1 = 2
    T = 293.3
    print(McCumber.calc_cross_section_absorption(sigma_e, omega, center_omega,
                                                 N_0, N_1, T))
    print(McCumber.calc_cross_section_emission(sigma_a, omega, center_omega,
                                               N_0, N_1, T))
    # With numpy array
    omega = np.arange(12, dtype=float) * 1000
    sigma_a = np.arange(12, dtype=float) * 50
    sigma_e = np.arange(12, dtype=float) * 25
    print(McCumber.calc_cross_section_absorption(sigma_e, omega, center_omega,
                                                 N_0, N_1, T))
    print(McCumber.calc_cross_section_emission(sigma_a, omega, center_omega,
                                               N_0, N_1, T))
