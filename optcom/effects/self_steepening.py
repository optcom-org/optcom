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

from typing import Optional, Union

import numpy as np

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.effects.abstract_effect import AbstractEffect
from optcom.utils.fft import FFT


class SelfSteepening(AbstractEffect):
    r"""The self-steepening effect.

    Attributes
    ----------
    omega : numpy.ndarray of float
        The angular frequency array. :math:`[ps^{-1}]`
    time : numpy.ndarray of float
        The time array. :math:`[ps]`
    domega : float
        The angular frequency step. :math:`[ps^{-1}]`
    dtime : float
        The time step. :math:`[ps]`

    """

    def __init__(self) -> None:
        super().__init__()
        self._S: np.ndarray = np.array([])
    # ==================================================================
    def set(self, center_omega: np.ndarray = np.array([]),
            abs_omega: np.ndarray = np.array([])) -> None:
        self._S = 1.0 / center_omega
    # ==================================================================
    def op(self, waves: np.ndarray, id: int,
           corr_wave: Optional[np.ndarray] = None) -> np.ndarray:
        """The approximation of the operator of the
        self-steepening effect.
        """
        factor = (1 + self._omega*self._S[id])

        return FFT.ifft_mult_fft(corr_wave, factor)
    # ==================================================================
    def term(self, waves: np.ndarray, id: int,
             corr_wave: Optional[np.ndarray] = None) -> np.ndarray:

        factor = (1 + self._omega*self._S[id])

        return FFT.ifft_mult_fft(corr_wave*waves[id], factor)
    # ==================================================================
    def term_rk4ip(self, waves: np.ndarray, id: int,
                   corr_wave: Optional[np.ndarray] = None) -> np.ndarray:

        factor = (1 + self._omega*self._S[id])

        return factor * FFT.fft(corr_wave*waves[id])
