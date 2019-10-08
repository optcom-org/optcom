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

import copy
import math
from typing import Dict, List, Optional, overload, Tuple

import numpy as np
from nptyping import Array
from scipy.misc import derivative
from scipy import interpolate

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.domain import Domain
from optcom.equations.abstract_equation import AbstractEquation


class AbstractRefractiveIndex(AbstractEquation):
    r"""Parent of refractive index generator class."""

    def __init__(self, medium):
        r"""
        Parameters
        ----------
        medium :
            The medium in which the wave propagates.

        """
        self._medium: str = medium.lower()
        self._predict: Optional[Callable] = None # refractive index fitting fct
    # ==================================================================
    @overload
    def n(self, omega: float) -> float: ...
    # ------------------------------------------------------------------
    @overload
    def n(self, omega: Array[float]) -> Array[float]: ...
    # ------------------------------------------------------------------
    def n(self, omega):
        """Compute the refractive index.

        Parameters
        ----------
        omega :
            The angular frequency. :math:`[rad\cdot ps^{-1}]`

        Returns
        -------
        :
            The refractive index.

        """
        ...
    # ==================================================================
    @overload
    def n_deriv(self, omega: float, order_deriv: int) -> float: ...
    # ------------------------------------------------------------------
    @overload
    def n_deriv(self, omega: Array[float], order_deriv: int) -> Array[float]: ...
    # ------------------------------------------------------------------
    def n_deriv(self, omega, order_deriv):
        r"""Compute the derivative of the refractive index.

        Parameters
        ----------
        omega :
            The angular frequency. :math:`[rad\cdot ps^{-1}]`
        order :
            The order of the derivative. (0 <= order <= 5)

        Returns
        -------
        :
            The derivative of the refractive index.

        """
        order = max(3, order_deriv+1+(order_deriv%2))
        if (isinstance(omega, float)):
            res = 0.0
            res = derivative(self.n, omega, n=order_deriv, order=order)
        else:
            res = np.zeros_like(omega)
            for i in range(len(omega)):
                res[i] = derivative(self.n, omega[i], n=order_deriv,
                                    order=order)
        return res
