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

import numpy as np

from optcom.equations.convergence_checker.abstract_convergence_checker import\
    AbstractConvergenceChecker, has_converged_decorator
from optcom.domain import Domain
from optcom.field import Field


class ConvergenceCheckerConsecutive(AbstractConvergenceChecker):
    """General convergence conditions for taking waves and noises into
    account.
    """

    def __init__(self, tolerance: float = 1e-2, max_nbr_iter: int = 100,
                 stop_if_divergent: bool = False) -> None:
        super().__init__(tolerance, max_nbr_iter, stop_if_divergent)
        self._prev_energy: float = 0.0
        self._dtime: float = 0.
        self._noise_domega: float = 0.
    # ==================================================================
    def initialize(self, domain: Domain) -> None:
        self._dtime = domain.dtime
        self._noise_domega = domain.noise_domega
    # ==================================================================
    @has_converged_decorator
    def has_converged(self, waves: np.ndarray, noises: np.ndarray) -> bool:
        """Compare the previous and current iterate and return True if
        the difference is below the specified threshold.

        Parameters
        ----------
        waves :
            The current values of the wave envelopes.
        noises :
            The current values of the noise powers.

        Returns
        -------
        :
            Boolean specifying the result of the comparison.

        """
        crt_energy: float = 0.
        crt_energy += float(np.sum(Field.energy(waves, self._dtime)))
        crt_energy += float(np.sum(noises*self._noise_domega))
        if (not self._crt_iter):
            self._prev_energy = crt_energy
            self.residual = math.inf

            return False
        else:
            eps: float = 1e-30
            self.residual = abs((self._prev_energy - crt_energy)
                                / (self._prev_energy + eps))
            self._prev_energy = crt_energy

            return self._residual < self._tolerance
