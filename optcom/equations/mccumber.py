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
from typing import Dict, List, overload, Tuple

import numpy as np

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.domain import Domain
from optcom.equations.abstract_equation import AbstractEquation


# typing variables
STARK_ENERGIES_TYPE = Tuple[List[float], List[float]]


DOPANT: List[str] = ["yb"]
MEDIA: Dict[str, List[str]] = {"yb": ["sio2", "po4", "bo3", "teo2"]}
# ref for values of yb in silica: Huseina, A.H.M. and EL-Nahalb, F.I.,
# 2011. Model of temperature dependence shape of ytterbium-doped fiber
# amplifier operating at 915 nm pumping configuration. IJACSA Editorial.
# ref other yb glasses: Jiang, C., Liu, H., Zeng, Q., Wang, Y.C., Zhang,
# J. and Gan, F., 2000, April. Stark energy split characteristics of
# ytterbium ion in glasses. In Rare-Earth-Doped Materials and Devices
# IV (Vol. 3942, pp. 312-317). International Society for Optics and
# Photonics.
# STARK = {dopant: {medium: (1st_level_stark,
#                                     2nd_level_stark)}} # in cm^{-1}
STARK: Dict[str, Dict[str, STARK_ENERGIES_TYPE]]
STARK = {"yb": {"sio2": ([0, 492, 970, 1365], [10239, 10909, 11689]),
                 #([0,338,445,872], [10288,10438,11038]),
                 "po4": ([0, 150, 372], [10269, 10616]),
                 "bo3": ([0, 139, 357], [10301, 10638]),
                 "teo2": ([0, 223, 449], [10269, 10730])}}


class McCumber(AbstractEquation):
    r"""McCumber relations.

    Represent McCumber relations which give a relationship between the
    effective cross-sections of absorption and emission of light in the
    solid-state laser physics. [2]_

    Notes
    -----

    .. math:: \frac{\sigma_e(\omega)}{\sigma_a(\omega)}
              = \frac{Z_L}{Z_U} \exp\Big[\frac{E_{UL}-\hbar\omega}
              {k_{B}T}\Big]

    where

    .. math:: \begin{split}
               Z_L = \sum_i \exp\Big[-\frac{\Delta E_{Li}}{k_B T}\Big]\\
               Z_U = \sum_j \exp\Big[-\frac{\Delta E_{Uj}}{k_B T}\Big]
              \end{split}

    References
    ----------
    .. [2] Florea, C. and Winick, K.A., 1999. Ytterbium-doped glass
           waveguide laser fabricated by ion exchange. Journal of
           Lightwave Technology, 17(9), pp.1593-1601.

    """

    def __init__(self, dopant: str = cst.DEF_FIBER_DOPANT,
                 medium: str = cst.DEF_FIBER_DOPANT,
                 stark_energies: STARK_ENERGIES_TYPE = ([], [])) -> None:
        """
        Parameters
        ----------
        dopant :
            The doping medium.
        medium :
            The fiber medium.
        stark_energies :
            The stark energies of the first level and the second level.
            The first element of the tuple is a list with all stark
            energies of the first level and the second element of the
            tuple is a list with all stark energies of the second level.
            :math:`[cm^{-1}]` (dopant and medium will be ignored if
            stark_energies are provided)

        """
        self._dopant: str = util.check_attr_value(dopant.lower(), DOPANT,
                                                  cst.DEF_FIBER_DOPANT)
        self._medium: str = util.check_attr_value(medium.lower(),
                                                  MEDIA[self._dopant],
                                                  cst.DEF_FIBER_MEDIUM)
        self._stark_energies: STARK_ENERGIES_TYPE = ([], [])
        if (stark_energies != ([], [])):
            self._stark_energies = stark_energies
        else:
            data = STARK.get(self._dopant)
            if (data is not None):
                stark = data.get(self._medium)
                if (stark is not None):
                    self._stark_energies = stark
                else:
                    util.warning_terminal("The specify medium {} for the "
                        "dopant {} for Mccumber relations is not valid."
                        .format(self._medium, self._dopant))
                    stark_ = data.get(cst.DEF_FIBER_MEDIUM)
                    if (stark_ is not None): # Should be obvious (for typing)
                        self._stark_energies = stark_
            else:
                util.warning_terminal("The specify dopant medium {} for "
                    "Mccumber relations is not valid.".format(self._dopant))
                data_ = STARK.get(cst.DEF_FIBER_DOPANT)
                if (data_ is not None): # Should be obvious (for typing)
                    stark_ = data_.get(cst.DEF_FIBER_MEDIUM)
                    if (stark_ is not None): # Same, it is obvious
                        self._stark_energies = stark_
    # ==================================================================
    @overload
    def cross_section_absorption(self, omega: float, sigma_e: float, T: float
                                 ) -> float: ...
    # ------------------------------------------------------------------
    @overload
    def cross_section_absorption(self, omega: np.ndarray,
                                 sigma_e: np.ndarray, T: float) -> float: ...
    # ------------------------------------------------------------------
    def cross_section_absorption(self, omega, sigma_e, T):
        r"""Calcul the absorption cross section.

        Parameters
        ----------
        omega :
            The angular frequency.  :math:`[rad\cdot ps^{-1}]`
        sigma_e :
            The emission cross section. :math:`[nm^2]`
        T :
            The temperature. :math:`[K]`

        """

        return McCumber.calc_cross_section_absorption(omega, sigma_e, T,
                                                      self._stark_energies)
    # ==================================================================
    @overload
    def cross_section_emission(self, omega: float, sigma_a: float, T: float
                               ) -> float: ...
    # ------------------------------------------------------------------
    @overload
    def cross_section_emission(self, omega: np.ndarray,
                               sigma_a: np.ndarray, T: float) -> float: ...
    # ------------------------------------------------------------------
    def cross_section_emission(self, omega, sigma_a, T):
        r"""Calcul the emission cross section.

        Parameters
        ----------
        omega :
            The angular frequency.  :math:`[rad\cdot ps^{-1}]`
        sigma_a :
            The absorption cross section. :math:`[nm^2]`
        T :
            The temperature. :math:`[K]`

        """

        return McCumber.calc_cross_section_emission(omega, sigma_a, T,
                                                    self._stark_energies)
    # ==================================================================
    @overload
    @staticmethod
    def calc_cross_section_absorption(omega: float, sigma_e: float, T: float,
                                      stark_energies: STARK_ENERGIES_TYPE
                                      ) -> float: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def calc_cross_section_absorption(omega: np.ndarray,
                                      sigma_e: np.ndarray, T: float,
                                      stark_energies: STARK_ENERGIES_TYPE
                                      ) -> np.ndarray: ...
    # ------------------------------------------------------------------
    @staticmethod
    def calc_cross_section_absorption(omega, sigma_e, T, stark_energies):
        r"""Calcul the absorption cross section.

        Parameters
        ----------
        omega :
            The angular frequency.  :math:`[rad\cdot ps^{-1}]`
        sigma_e :
            The emission cross section. :math:`[nm^2]`
        T :
            The temperature. :math:`[K]`
        stark_energies :
            The stark energies of the first level and the second level.
            The first element of the tuple is a list with all stark
            energies of the first level and the second element of the
            tuple is a list with all stark energies of the second level.
            :math:`[cm^{-1}]`

        Returns
        -------
        :
            Value of the absorption cross section. :math:`[nm^2]`

        Notes
        -----

        .. math:: \sigma_a(\omega) = \sigma_e(\omega)
                  \frac{Z_U}{Z_L} \exp\Big[\frac{\hbar\omega - E_{UL}}
                  {k_{B}T}\Big]

        """
        KB = cst.KB
        HBAR = cst.HBAR
        H = cst.H
        C = cst.C
        Z_L = 0.0
        for i in range(1, len(stark_energies[0])):
            delta_energy = (stark_energies[0][i]-stark_energies[0][0]) * H * C
            delta_energy *= 1e-7    # cm^{-1} -> nm^{-1}
            Z_L += math.exp(-1 * delta_energy / (KB*T))
        Z_U = 0.0
        for i in range(1, len(stark_energies[1])):
            delta_energy = (stark_energies[1][i]-stark_energies[1][0]) * H * C
            delta_energy *= 1e-7    # cm^{-1} -> nm^{-1}
            Z_U += math.exp(-1 * delta_energy / (KB*T))

        E_UL = stark_energies[1][0] * H * C * 1e-7  # cm^{-1} -> nm^{-1}
        if (isinstance(omega, float)):
            factor = math.exp((HBAR*omega - E_UL) / (KB*T))
        else:
            factor = np.exp((HBAR*omega - E_UL) / (KB*T))

        #return sigma_e * (Z_U/Z_L) * factor

        return sigma_e * factor
    # ==================================================================
    @overload
    @staticmethod
    def calc_cross_section_emission(omega: float, sigma_a: float, T: float,
                                    stark_energies: STARK_ENERGIES_TYPE
                                    ) -> float: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def calc_cross_section_emission(omega: np.ndarray, sigma_a: np.ndarray,
                                    T: float,
                                    stark_energies: STARK_ENERGIES_TYPE
                                    ) -> np.ndarray: ...
    # ------------------------------------------------------------------
    @staticmethod
    def calc_cross_section_emission(omega, sigma_a, T, stark_energies):
        r"""Calcul the emission cross section.

        Parameters
        ----------
        omega :
            The angular frequency.  :math:`[rad\cdot ps^{-1}]`
        sigma_a :
            The absorption cross section. :math:`[nm^2]`
        T :
            The temperature. :math:`[K]`
        stark_energies :
            The stark energies of the first level and the second level.
            The first element of the tuple is a list with all stark
            energies of the first level and the second element of the
            tuple is a list with all stark energies of the second level.
            :math:`[cm^{-1}]`

        Returns
        -------
        :
            Value of the absorption cross section. :math:`[nm^2]`

        Notes
        -----

        .. math:: \sigma_e(\omega) = \sigma_a(\omega)
                  \frac{Z_L}{Z_U} \exp\Big[\frac{E_{UL}-\hbar\omega}
                  {k_{B}T}\Big]

        """
        KB = cst.KB
        HBAR = cst.HBAR
        H = cst.H
        C = cst.C
        Z_L = 0.0
        for i in range(1, len(stark_energies[0])):
            delta_energy = (stark_energies[0][i]-stark_energies[0][0]) * H * C
            delta_energy *= 1e-7    # cm^{-1} -> nm^{-1}
            Z_L += math.exp(-1 * delta_energy / (KB*T))
        Z_U = 0.0
        for i in range(1, len(stark_energies[1])):
            delta_energy = (stark_energies[1][i]-stark_energies[1][0]) * H * C
            delta_energy *= 1e-7    # cm^{-1} -> nm^{-1}
            Z_U += math.exp(-1 * delta_energy / (KB*T))

        #Z_L *= 1. / len(stark_energies[0])
        #Z_U *= 1. / len(stark_energies[1])

        E_UL = stark_energies[1][0] * H * C * 1e-7  # cm^{-1} -> nm^{-1}
        #print('E_UL', E_UL)
        #print('HBAR', HBAR)
        #print('omega', omega[0])
        #print('kb', KB)
        #print('T', T)
        #print('kbt', KB*T)
        #print('num', E_UL - HBAR*omega[len(omega)//2])
        #print((E_UL - HBAR*omega[0]) / (KB*T))
        if (isinstance(omega, float)):
            factor = math.exp((E_UL - HBAR*omega) / (KB*T))
        else:
            factor = np.exp((E_UL - HBAR*omega) / (KB*T))

        #print(Z_L/Z_U, factor.shape, factor)
        #print(Z_L, Z_U, Z_L/Z_U)
        #return sigma_a * (Z_L/Z_U) * factor

        return sigma_a * factor


if __name__ == "__main__":

    # With float
    omega: float = 1220
    sigma_a: float = 50.
    sigma_e: float = 25.
    T: float = 293.3
    mc = McCumber(dopant='Yb', medium='SiO2')
    print(mc.cross_section_absorption(omega, sigma_e, T=T))
    print(mc.cross_section_emission(omega, sigma_a, T=T))
    # With numpy array
    omegas: np.ndarray = np.arange(12, dtype=float) * 1000
    sigmas_a: np.ndarray = np.arange(12, dtype=float) * 50
    sigmas_e: np.ndarray = np.arange(12, dtype=float) * 25
    print(mc.cross_section_absorption(omegas, sigmas_e, T=T))
    print(mc.cross_section_emission(omegas, sigmas_a, T=T))
