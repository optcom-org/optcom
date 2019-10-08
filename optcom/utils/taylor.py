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

import math
from typing import Any, List, Optional, Union

import numpy as np
from nptyping import Array

import optcom.utils.constants as cst


class Taylor(object):
    """Represent a Taylor series."""

    def __init__(self):

        return None
    @staticmethod
    def series(derivative: Array[float], x_diff: Array[float],
               start: Optional[int] = 0, stop: Optional[int] = None
               ) -> Array[float]:
        r"""Calculate the Taylor series according to the given
        parameters.

        Parameters
        ----------
        derivative : numpy.ndarray of float
            The derivatives of the function to approximate.
        x_diff : numpy.ndarray of float
            The values of the variables.

        Returns
        -------
        float
            Taylor series evaluation.

        Notes
        -----

        .. math::   f(x) &= \sum_{n=0}^N \frac{f_n}{n!}
                    (x - x_0)^n \quad \text{where}
                    \quad f_n = \left. \frac{d^n f}{dx^n}
                    \right\rvert_{x = x_0}\\

        """
        res = np.zeros(x_diff.shape, dtype=cst.NPFT)
        facto_i = math.factorial(start)
        x_power = np.power(x_diff, start)
        if (stop is None):
            stop = len(derivative)
        for i in range(start, stop):
            res += derivative[i] * x_power / facto_i
            x_power *= x_diff
            facto_i *= i+1

        return res
