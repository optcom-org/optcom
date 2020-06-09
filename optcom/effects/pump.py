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

from typing import Optional

import numpy as np

from optcom.effects.abstract_effect import AbstractEffect


class Pump(AbstractEffect):
    r"""The laser pump effect.

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
    pump_rate :
        The pump rate of the pump effect. :math:`[ps^{-1} nm^{-3}]`

    """

    def __init__(self, pump_rate: float) -> None:
        r"""
        Parameters
        ----------
        pump_rate :
            The pump rate of the pump effect. :math:`[ps^{-1} nm^{-3}]`

        """
        super().__init__()
        self.pump_rate: float = pump_rate
    # ==================================================================
    def set(self, center_omega: np.ndarray = np.array([]),
            abs_omega: np.ndarray = np.array([])) -> None:

        return None
    # ==================================================================
    def op(self, waves: np.ndarray, id: int,
           corr_wave: Optional[np.ndarray] = None) -> np.ndarray:
        """The operator of the pump effect."""

        return self.pump_rate
