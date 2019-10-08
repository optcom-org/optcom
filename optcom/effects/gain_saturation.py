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

from typing import List, Optional, Union

import numpy as np
from nptyping import Array

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.effects.abstract_effect import AbstractEffect
from optcom.effects.attenuation import Attenuation
from optcom.equations.re_fiber import REFiber


class GainSaturation(AbstractEffect):
    r"""Generic class for effect object.

    Attributes
    ----------
    omega : numpy.ndarray of float
        The angular frequency array. :math:`[ps^{-1}]`
    time : numpy.ndarray of float
        The time array. :math:`[ps]`
    center_omega : numpy.ndarray of float
        The center angular frequency. :math:`[ps^{-1}]`

    """

    def __init__(self, re: REFiber, alpha: Optional[List[float]] = None
                 ) -> None:
        r"""
        Parameters
        ----------
        re : REFiber
            The rate equations object.
        alpha :
            The attenuation coefficient. :math:`[km^{-1}]`

        """
        super().__init__()
        self._re: REFiber = re
        self._alpha: Array[float]
        if (alpha is None):
            self._alpha = np.zeros(0.0)
        else:
            alpha = util.make_list(alpha)
            self._alpha = np.asarray(alpha)
        self._factor: Array[float]
    # ==================================================================
    def update(self, center_omega: Optional[Array[float]] = None,
               step: int = 0) -> None:
        if (center_omega is None):
            center_omega = self._center_omega
        Gamma = np.zeros(len(center_omega))
        sigma_a = np.zeros(len(center_omega))
        sigma_e = np.zeros(len(center_omega))
        self._factor = np.zeros(len(center_omega))
        for i, omega in enumerate(center_omega):
            # sigma in m^2 from self._re
            sigma_a[i] = self._re.get_sigma_a(omega, step)
            sigma_e[i] = self._re.get_sigma_e(omega, step)
            Gamma[i] = self._re.get_Gamma(omega, step)
            self._factor[i] = ((-1*Gamma[i]*(sigma_a[i]+sigma_e[i]))
                               / (omega*cst.HBAR))
            self._factor[i] *= 1e-24    # ps^2 kg^-1 -> s^2 kg^-1
        if (len(center_omega) < len(self._alpha)):
            self._alpha = self._alpha[:len(center_omega)]
        else:
            for i in range(len(self._alpha), len(center_omega)):
                self._alpha = np.vstack((self._alpha, self._alpha[-1]))
    # ==================================================================
    @property
    def center_omega(self) -> Optional[Array[float]]:

        return self._center_omega
    # ==================================================================
    @center_omega.setter
    def center_omega(self, center_omega: Array[float]) -> None:
        # Overloading to update the betas(\omega)
        self.update(center_omega)
        self._center_omega = center_omega
    # ==================================================================
    @property
    def alpha(self) -> Optional[Array[float]]:

        return self._alpha
    # ==================================================================
    @alpha.setter
    def alpha(self, alpha: Array[float]) -> None:
        self._alpha = alpha
    # ==================================================================
    def op(self, waves: Array[cst.NPFT], id: int,
           corr_wave: Optional[Array[cst.NPFT]] = None) -> Array[cst.NPFT]:
        """The operator of the gain saturation effect."""

        res = 0.0
        power = np.zeros(waves[id].shape, dtype=cst.NPFT)
        for i in range(len(waves)):
            power += waves[i] * np.conj(waves[id])  # |A|^2

        res = (-0.5*self._alpha[id]
               * np.exp(self._factor[id]*np.real(np.sum(power))))

        return res
