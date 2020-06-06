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

import optcom.utils.constants as cst


class Taylor(object):
    """Represent a Taylor series."""

    def __init__(self):

        return None
    @staticmethod
    def series(derivative: np.ndarray, x_diff: np.ndarray,
               start: int = 0, stop: int = -1, skip: List[int] = []
               ) -> np.ndarray:
        r"""Calculate the Taylor series according to the given
        parameters.

        Parameters
        ----------
        derivative :
            The derivatives of the function to approximate.
        x_diff :
            The values of the variables.

        Returns
        -------
        :
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
        if (stop == -1):
            stop = len(derivative)
        for i in range(start, stop):
            if (i not in skip):
                res += derivative[i] * x_power / facto_i
            x_power *= x_diff
            facto_i *= i+1

        return res
