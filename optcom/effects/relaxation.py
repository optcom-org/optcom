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


class Relaxation(AbstractEffect):
    r"""The relaxation effect.

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

    def __init__(self, decay_time: float) -> None:
        r"""
        Parameters
        ----------
        decay_time :
            The decay time of the relaxation effect. :math:`[\mu s]`

        """
        super().__init__()
        self._decay_time: float = decay_time * 1e-6  # mu s -> s
        self._decay_rate: float = 1.0 / self._decay_time
    # ==================================================================
    @property
    def decay_time(self):

        return self._decay_time
    # ==================================================================
    @property
    def decay_rate(self):

        return self._decay_rate
    # ==================================================================
    def set(self, center_omega: np.ndarray = np.array([]),
            abs_omega: np.ndarray = np.array([])) -> None:

        return None
    # ==================================================================
    def op(self, waves: np.ndarray, id: int,
           corr_wave: Optional[np.ndarray] = None) -> np.ndarray:
        """The operator of the relaxation effect."""

        return self._decay_rate
    # ==================================================================
    def term(self, waves: np.ndarray, id: int,
             corr_wave: Optional[np.ndarray] = None) -> np.ndarray:
        """The operator of the relaxation effect."""

        return self._decay_rate
