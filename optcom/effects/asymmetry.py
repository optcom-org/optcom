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
        self._delta: Union[np.ndarray, Callable]
        if (callable(delta)):
            self._delta = delta
        else:
            self._delta = lambda omega: np.ones_like(omega) * delta
    # ==================================================================
    def set(self, center_omega: np.ndarray = np.array([]),
            abs_omega: np.ndarray = np.array([])) -> None:

        self._delta_op = self._delta(center_omega).T
    # ==================================================================
    def op(self, waves: np.ndarray, id: int,
           corr_wave: Optional[np.ndarray] = None) -> np.ndarray:
        """The operator of the asymmetry effect."""

        return 1j * self._delta_op[id]
