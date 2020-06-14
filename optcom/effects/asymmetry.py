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

from typing import Callable, List, Optional, overload, Union

import numpy as np

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.effects.abstract_effect import AbstractEffect
from optcom.effects.dispersion import Dispersion
from optcom.parameters.refractive_index.sellmeier import Sellmeier
from optcom.utils.taylor import Taylor


class Asymmetry(AbstractEffect):
    r"""The core asymmetry effect.

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

    def __init__(self, delta: Union[float, Callable] = None) -> None:
        r"""
        Parameters
        ----------
        delta :
            The asymmetry measure coefficient. If a callable is provided,
            variable must be angular frequency. :math:`[ps^{-1}]`

        """
        super().__init__()
        # The asymmetry coefficient ------------------------------------
        self._delta_op: np.ndarray = np.array([])
        self._delta: Callable
        if (callable(delta)):
            self._delta = delta
        else:
            self._delta = lambda omega: np.ones_like(omega) * delta
    # ==================================================================
    @property
    def delta(self) -> Callable:

        return self._delta
    # ==================================================================
    def set(self, center_omega: np.ndarray = np.array([]),
            abs_omega: np.ndarray = np.array([])) -> None:

        self._delta_op = self._delta(center_omega).T
    # ==================================================================
    def op(self, waves: np.ndarray, id: int,
           corr_wave: Optional[np.ndarray] = None) -> np.ndarray:
        """The operator of the asymmetry effect."""

        return 1j * self._delta_op[id]
