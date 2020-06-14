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

from typing import Callable, Optional, overload, Union

import numpy as np

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.equations.abstract_equation import AbstractEquation


COEFF_TYPE = Union[float, np.ndarray, Callable]


class RE2Levels(AbstractEquation):
    r"""Implement the 2-level rate equations.

    Notes
    -----

    .. math::
            \begin{split}
                \frac{\partial N_0}{\partial t} &= +\gamma_{10} N_1 -
                R_p + W_{E,10} N_1 - W_{A,01} N_0\\
                \frac{\partial N_1}{\partial t} &= -\gamma_{10} N_1 +
                R_p - W_{E,10} N_1 + W_{A,01} N_0
            \end{split}

    In the case of 2-level rate equations in steady state
    (:math:`\frac{\partial N}{\partial t}=0`), considering that
    :math:`N_0 + N_1 = N_T` an analytical solution is feasible:

    .. math::
            \begin{split}
                N_0 = \frac{\gamma_{10}N_1 - R_p
                      + W_{E,10}N_1}{W_{A,01}}\\
                N_1  = N_T - N_0
            \end{split}

    """

    def __init__(self, N_T: float, gamma_10: COEFF_TYPE, R_p: COEFF_TYPE,
                 W_E10: COEFF_TYPE, W_A01: COEFF_TYPE) -> None:
        """
        Parameters
        ----------
        N_T :

        gamma_10 :

        R_P :

        W_E10 :

        W_A01 :

        """
        # Total population density -------------------------------------
        self._N_T: float = 0
        self.N_T = N_T
        # Relaxation ---------------------------------------------------
        self.gamma_10 = gamma_10
        # Pump ---------------------------------------------------------
        self.R_p = R_p
        # Emission -----------------------------------------------------
        self._W_E10 = W_E10
        # Absorption ---------------------------------------------------
        self._W_A10 = W_A01
    # ==================================================================
    @property
    def N_T(self) -> float:

        return self._N_T
    # ------------------------------------------------------------------
    @N_T.setter
    def N_T(self, N_T) -> None:
        util.check_attr_type(N_T, 'N_T', float)
        self._N_T = N_T
    # ==================================================================
    @property
    def gamma_10(self) -> COEFF_TYPE:

        return self._gamma_10
    # ------------------------------------------------------------------
    @gamma_10.setter
    def gamma_10(self, gamma_10: COEFF_TYPE) -> None:
        util.check_attr_type(gamma_10, 'gamma_10', float, np.ndarray, Callable)
        self._gamma_10: COEFF_TYPE = gamma_10
        self._gamma: Callable
        if (callable(gamma_10)):
            self._gamma = gamma_10
        else:
            if (isinstance(gamma_10, float)):
                self._gamma = lambda t: gamma_10
            else:
                self._gamma = lambda vec_t: np.ones_like(vec_t) * gamma_10
    # ==================================================================
    @property
    def R_p(self) -> COEFF_TYPE:

        return self._R_p
    # ------------------------------------------------------------------
    @R_p.setter
    def R_p(self, R_p: COEFF_TYPE) -> None:
        util.check_attr_type(R_p, 'R_p', float, np.ndarray, Callable)
        self._R_p = R_p
        self._pump: Callable
        if (callable(R_p)):
            self._pump = R_p
        else:
            if (isinstance(R_p, float)):
                self._pump = lambda t: R_p
            else:
                self._pump = lambda vec_t: np.ones_like(vec_t) * R_p
    # ==================================================================
    @property
    def W_E10(self) -> COEFF_TYPE:

        return self._W_E10
    # ------------------------------------------------------------------
    @W_E10.setter
    def W_E10(self, W_E10: COEFF_TYPE) -> None:
        util.check_attr_type(W_E10, 'W_E10', float, np.ndarray, Callable)
        self._W_E10 = W_E10
        self._emission: Callable
        if (callable(W_E10)):
            self._emission = W_E10
        else:
            if (isinstance(W_E10, float)):
                self._emission = lambda t: W_E10
            else:
                self._emission = lambda vec_t: np.ones_like(vec_t) * W_E10
    # ==================================================================
    @property
    def W_A01(self) -> COEFF_TYPE:

        return self._W_A01
    # ------------------------------------------------------------------
    @W_A01.setter
    def W_A01(self, W_A01: COEFF_TYPE) -> None:
        util.check_attr_type(W_A01, 'W_A01', float, np.ndarray, Callable)
        self._W_A01 = W_A01
        self._absorption: Callable
        if (callable(W_A01)):
            self._absorption = W_A01
        else:
            if (isinstance(W_A01, float)):
                self._absorption = lambda t: W_A01
            else:
                self._absorption = lambda vec_t: np.ones_like(vec_t) * W_A01
    # ==================================================================
    @overload
    def __call__(self, vectors: np.ndarray, z: float, h: float
                 )-> np.ndarray: ...
    # ------------------------------------------------------------------
    @overload
    def __call__(self, vectors: np.ndarray, z: float, h: float, ind: int
                 ) -> np.ndarray: ...
    # ------------------------------------------------------------------
    def __call__(self, *args):
        pop_ = np.zeros_like(pop)
        pop_[0] = ((self._gamma(t)*pop[1]) - self._pump(t)
                   + (self._emission(t)*pop[1]) - (self._absorption(t)*pop[0]))
        pop_[1] = -1 * pop_[0]

        return pop_
    # ==================================================================
    def calc_pop(self, pop: np.ndarray, t: float, h: float):
        pop_ = np.zeros_like(pop)
        pop_[0] = (((self._gamma(t)*pop[1]) - self._pump(t)
                    + (self._emission(t)*pop[1])) / self._absorption(t))
        pop_[1] = self._N_T - pop_[0]

        return pop_
