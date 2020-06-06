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
