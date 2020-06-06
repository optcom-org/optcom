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

from abc import ABCMeta, abstractmethod
from typing import Optional

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.domain import Domain


def has_converged_decorator(has_converged):
    def method_wrapper(self, *args, **kwargs):
        back_up_residual: float = self.residual
        res: bool = has_converged(self, *args, **kwargs)
        # Stop if divergence not accepted
        if (self._crt_iter and self.residual > back_up_residual):
            util.warning_terminal("Divergent method !")
            if (self.stop_if_divergent):
                res = True
        self._crt_iter += 1
        # Stop if the maximum nbr of steps is reached
        if (not res and self._crt_iter >= self._max_nbr_iter):
            util.print_terminal("Maximum number of iterations reached.", '')
            res = True
        if (res):
            self.reset()

        return res

    return method_wrapper


class AbstractConvergenceChecker(metaclass=ABCMeta):
    """Define a common framework for convergence checking.
    """

    def __init__(self, tolerance: float, max_nbr_iter: int,
                 stop_if_divergent: bool) -> None:
        self._tolerance: float = tolerance
        self._max_nbr_iter: int = max_nbr_iter
        self._crt_iter: int = 0    # incremented in has_converged decorator
        self._residual: float = 0.0
        self.stop_if_divergent: bool = stop_if_divergent
    # ==================================================================
    def initialize(self, domain: Domain) -> None: ...
    # ==================================================================
    @property
    def max_nbr_iter(self) -> int:

        return self._max_nbr_iter
    # ------------------------------------------------------------------
    @max_nbr_iter.setter
    def max_nbr_iter(self, max_nbr_iter: int) -> None:
        self._max_nbr_iter = max_nbr_iter
    # ==================================================================
    def reset(self):
        self._crt_iter = 0
    # ==================================================================
    @abstractmethod
    def has_converged(self, *args) -> bool: ...
    # ==================================================================
    @property
    def residual(self) -> float:

        return self._residual
    # ------------------------------------------------------------------
    @residual.setter
    def residual(self, new_residual: float) -> None:
        self._residual = new_residual
