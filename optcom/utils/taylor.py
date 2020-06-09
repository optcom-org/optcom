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
