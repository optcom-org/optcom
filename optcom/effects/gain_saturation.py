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

from typing import Callable, List, Optional, Union

import numpy as np

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.domain import Domain
from optcom.effects.abstract_effect import AbstractEffect
from optcom.effects.attenuation import Attenuation
from optcom.field import Field
from optcom.utils.taylor import Taylor


class GainSaturation(AbstractEffect):
    r"""The active fiber saturation effect.

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

    def __init__(self, gain: Union[float, np.ndarray, Callable],
                 en_sat: Union[float, np.ndarray, Callable],
                 UNI_OMEGA: bool = True) -> None:
        r"""
        Parameters
        ----------
        gain :
            The gain coefficient. :math:`[km^{-1}]`  If a callable is
            provided, variable must be angular frequency.
            :math:`[ps^{-1}]`
        en_sat :
            The energy saturation. :math:`[J]`  If a callable is
            provided, the variable must be angular frequency.
            :math:`[ps^{-1}]`
        UNI_OMEGA :
            If True, consider only the center omega for computation.
            Otherwise, considered omega discretization.

        """
        super().__init__()
        self._UNI_OMEGA = UNI_OMEGA
        # The gain coefficient -----------------------------------------
        self._gain_op: np.ndarray = np.array([])
        self._gain: Union[np.ndarray, Callable]
        if (callable(gain)):
            self._gain = gain
        else:
            self._gain = lambda omega: np.ones_like(omega) * gain
        # The energy saturation ----------------------------------------
        self._en_sat_op: np.ndarray = np.array([])
        self._en_sat: Union[np.ndarray, Callable]
        if (callable(en_sat)):
            self._en_sat = en_sat
        else:
            self._en_sat = lambda omega: np.ones(omega) * en_sat
    # ==================================================================
    def set(self, center_omega: np.ndarray = np.array([]),
            abs_omega: np.ndarray = np.array([])) -> None:
        if (self._UNI_OMEGA):
            self._gain_op = np.zeros_like(center_omega)
            self._gain_op = self._gain(center_omega)
            self._en_sat_op = np.zeros_like(center_omega)
            self._en_sat_op = self._en_sat(center_omega)
        else:
            self._gain_op = np.zeros_like(abs_omega)
            self._en_sat_op = np.zeros_like(abs_omega)
            for i in range(len(center_omega)):
                self._gain_op[i] = self._gain(abs_omega[i])
                self._en_sat_op[i] = self._en_sat(abs_omega[i])
    # ==================================================================
    def op(self, waves: np.ndarray, id: int,
           corr_wave: Optional[np.ndarray] = None) -> np.ndarray:
        """The operator of the gain saturation effect."""
        dtime: float = self._dtime * 1e-12     # ps -> s
        energy: float = 0.0
        for i in range(len(waves)):
            energy += Field.energy(waves[i], dtime)

        return 0.5 * self._gain_op[id] * np.exp(-1*energy/self._en_sat_op[id])
