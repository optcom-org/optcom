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

from typing import Callable, List, Optional, Union

import numpy as np
from nptyping import Array

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.effects.abstract_effect import AbstractEffect
from optcom.utils.taylor import Taylor


class Attenuation(AbstractEffect):
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

    def __init__(self, alpha: Optional[Union[List[float], Callable]] = None,
                 order: int = 1, medium: str = cst.DEF_FIBER_MEDIUM,
                 start_taylor: int = 0) -> None:
        r"""
        Parameters
        ----------
        alpha :
            The derivatives of the attenuation coefficients.
            :math:`[km^{-1}, ps\cdot km^{-1}, ps^2\cdot km^{-1},
            ps^3\cdot km^{-1}, \ldots]` If a callable is provided,
            variable must be angular frequency. :math:`[ps^{-1}]`
        order :
            The order of alpha coefficients to take into account. (will
            be ignored if alpha values are provided - no file)
        medium :
            The medium in which the attenuation is considered.
        start_taylor :
            The order of the derivative from which to start the Taylor
            Series expansion.

        """
        super().__init__()
        self._order: int = order
        self._medium: str = medium
        self._start_taylor: int = start_taylor
        self._predict: Optional[Callable] = None
        self._alpha: Array[float]
        self._future_eq_to_calc_alpha: bool = False
        if (alpha is not None):
            if (callable(alpha)):
                self._predict = alpha
            else:   # alpha is a array of float
                alpha = util.make_list(alpha) # make sure is list
                self._alpha = np.asarray(alpha).reshape((1,-1))
                self._order = self._alpha.shape[1] - 1
        else:
            self._future_eq_to_calc_alpha = True
    # ==================================================================
    @property
    def alpha(self) -> List[float]:

        return self._alpha
    # ==================================================================
    @alpha.setter
    def alpha(self, alpha: List[float]) -> None:
        self._alpha = alpha
    # ==================================================================
    def __getitem__(self, key: int) -> float:

        return self._alpha[key]
    # ==================================================================
    def __setitem__(self, key: int, alpha: float) -> None:
        self._alpha[key] = alpha
    # ==================================================================
    def __delitem__(self, key: int) -> None:
        self._alpha[key] = 0.0
    # ==================================================================
    def __len__(self) -> int:

        return len(self._alpha)
    # ==================================================================
    @property
    def center_omega(self) -> Optional[Array[float]]:

        return self._center_omega
    # ==================================================================
    @center_omega.setter
    def center_omega(self, center_omega: Array[float]) -> None:
        # Overloading to update the alphas(\omega)
        self.update(center_omega)
        self._center_omega = center_omega
    # ==================================================================
    def update(self, center_omega: Optional[Array[float]] = None) -> None:
        # Do no test if alpha is None to be able to do multi pass
        if (center_omega is None):
            center_omega = self._center_omega
        if (self._predict is not None):
            self._alpha = np.zeros((len(center_omega), self._order+1))
            if (self._future_eq_to_calc_alpha):
                # to do - consider the material provided
                util.warning_terminal("Alphas set to zeros, must still code "
                    "alpha calculation depending on material.")
            else:
                self._alpha = self._predict(center_omega, self._order).T
                print('alpha', self._order, self._alpha)
        else:
            if (len(center_omega) < len(self._alpha)):
                self._alpha = self._alpha[:len(center_omega)]
            else:
                for i in range(len(self._alpha), len(center_omega)):
                    self._alpha = np.vstack((self._alpha, self._alpha[-1]))
    # ==================================================================
    def op(self, waves: Array[cst.NPFT], id: int,
           corr_wave: Optional[Array[cst.NPFT]] = None) -> Array[cst.NPFT]:
        """The operator of the attenuation effect."""

        op = Taylor.series(self._alpha[id], self._omega, self._start_taylor)

        return -0.5 * op
