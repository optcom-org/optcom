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
