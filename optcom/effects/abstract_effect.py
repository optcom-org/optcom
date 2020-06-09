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

from abc import ABCMeta, abstractmethod
from typing import List, Optional

import numpy as np

import optcom.utils.constants as cst


class AbstractEffect(metaclass=ABCMeta):
    r"""Generic class for effect object.

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

        self._omega: np.ndarray = np.array([])
        self._time: np.ndarray = np.array([])
        self._rep_freq: np.ndarray = np.array([])
        self._domega: float = 0.0
        self._dtime: float = 0.0
    # ==================================================================
    @property
    def omega(self) -> np.ndarray:

        return self._omega
    # ------------------------------------------------------------------
    @omega.setter
    def omega(self, omega: np.ndarray) -> None:
        self._omega = omega
    # ==================================================================
    @property
    def time(self) -> np.ndarray:

        return self._time
    # ------------------------------------------------------------------
    @time.setter
    def time(self, time: np.ndarray) -> None:
        self._time = time
    # ==================================================================
    @property
    def domega(self) -> float:

        return self._domega
    # ------------------------------------------------------------------
    @domega.setter
    def domega(self, domega: float) -> None:
        self._domega = domega
    # ==================================================================
    @property
    def dtime(self) -> float:

        return self._dtime
    # ------------------------------------------------------------------
    @dtime.setter
    def dtime(self, dtime: float) -> None:
        self._dtime = dtime
    # ==================================================================
    @property
    def rep_freq(self) -> np.ndarray:

        return self._rep_freq
    # ------------------------------------------------------------------
    @rep_freq.setter
    def rep_freq(self, rep_freq: np.ndarray) -> None:
        self._rep_freq = rep_freq
    # ==================================================================
    def delay_factors(self, id: int) -> List[float]:
        """Return the time delay induced by the effect."""

        return []
    # ==================================================================
    @abstractmethod
    def set(self, center_omega: np.ndarray = np.array([]),
            abs_omega: np.ndarray = np.array([])) -> None: pass
    # ==================================================================
    @abstractmethod
    def op(self, waves: np.ndarray, id: int,
           corr_wave: Optional[np.ndarray] = None) -> np.ndarray:
        """The operator of the effect.

        Parameters
        ----------
        waves :
            The wave packet.
        id :
            The ID of the considered wave in the wave packet.
        corr_wave :
            Corrective wave, optional, use if needed.

        Returns
        -------
        :
            The operator of the effect.

        """
        ...
    # ==================================================================
    def term(self, waves: np.ndarray, id: int,
             corr_wave: Optional[np.ndarray] = None) -> np.ndarray:
        """The term of the effect.

        Parameters
        ----------
        waves :
            The wave packet.
        id :
            The ID of the considered wave in the wave packet.
        corr_wave :
            Corrective wave, optional, use if needed.

        Returns
        -------
        :
            The term of the effect.

        """
        if (corr_wave is None):
            corr_wave = waves[id]

        return self.op(waves, id, corr_wave) * corr_wave
