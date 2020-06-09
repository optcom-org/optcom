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
