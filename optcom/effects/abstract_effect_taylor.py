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

import multiprocessing as mp
import warnings
from typing import Callable, List, Optional, overload, Union

import numpy as np

import optcom.config as cfg
import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.effects.abstract_effect import AbstractEffect
from optcom.utils.callable_container import CallableContainer
from optcom.utils.fft import FFT
from optcom.utils.taylor import Taylor


# Exceptions
class AbstractEffectTaylorWarning(UserWarning):
    pass

class TaylorOrderWarning(AbstractEffectTaylorWarning):
    pass


class AbstractEffectTaylor(AbstractEffect):
    r"""An effect which need a Taylor Series expansion.

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
    order_taylor :
        The order of coeff coefficients Taylor series expansion to
        take into account.

    """

    def __init__(self, coeff: Union[List[float], Callable],
                 order_taylor: int = 1, start_taylor: int = 0,
                 skip_taylor: List[int] = [], UNI_OMEGA: bool = False) -> None:
        r"""
        Parameters
        ----------
        coeff :
            The derivatives of the coefficients.
        order_taylor :
            The order of coeff coefficients Taylor series expansion to
            take into account. (will be set to the length of the coeff
            array if one is provided)
        start_taylor :
            The order of the derivative from which to start the Taylor
            series expansion.
        skip_taylor :
            The order_taylors of the derivative to not consider.
        UNI_OMEGA :
            If True, consider only the center omega for computation.
            Otherwise, considered omega discretization.

        """
        super().__init__()
        self._UNI_OMEGA = UNI_OMEGA
        self._order_taylor: int = order_taylor
        self._start_taylor: int = start_taylor
        self._skip_taylor: List[int] = skip_taylor
        # The attenuation constant -------------------------------------
        self._op: np.ndarray = np.array([])
        self._coeff_op: np.ndarray = np.array([])
        self._coeff: Union[np.ndarray, Callable]
        if (callable(coeff)):
            self._coeff = coeff
        else:
            self._coeff_values = np.asarray(util.make_list(coeff))
            #fct2pickle = lambda omega, order: util.hstack_like(coeff_, omega)
            #self._coeff = CallableContainer(fct2pickle)
            self._coeff = self._hstack_like
            max_order_taylor: int = len(self._coeff_values) - 1
            if (self._order_taylor > max_order_taylor):
                self._order_taylor = max_order_taylor
                warning_message = ("The requested order is higher than the "
                    "provided coefficients, max order of {} will be set."
                    .format(max_order_taylor))
                warnings.warn(warning_message, TaylorOrderWarning)
    # ==================================================================
    def _hstack_like(self, omega: np.ndarray, order: int) -> np.ndarray:
        # repetition of the hstack_like from utilities but need top
        # level function definition that can be pickle in the module
        # multiprocessing, lambda fct thus does not work. example:
        # coeff = lambda omega, order: util.hstack_like(coeff_, omega)

        return util.hstack_like(self._coeff_values, omega)
    # ==================================================================
    @property
    def order_taylor(self) -> int:

        return self._order_taylor
    # ------------------------------------------------------------------
    @order_taylor.setter
    def order_taylor(self, order_taylor: int) -> None:
        self._order_taylor = order_taylor
    # ==================================================================
    def coeff(self, omega: np.ndarray, order: Optional[int] = None
              ) -> np.ndarray:
        order_: int = self._order_taylor if order is None else order

        return self._coeff(omega, order_)
    # ==================================================================
    def delay_factors(self, id: int) -> List[float]:
        res = []
        if (self._UNI_OMEGA):
            for i in range(1, len(self._coeff_op[id]), 2):
                res.append(self._coeff_op[id][i])
        else:  # Assume the central frequency is the midpoint
            midpoint = (self._coeff_op.shape[2]//2) - 1
            #print(self._coeff_op.shape[2], midpoint)
            for i in range(1, len(self._coeff_op[id]), 2):
                res.append(self._coeff_op[id][i][midpoint])

        return res
    # ==================================================================
    def set(self, center_omega: np.ndarray = np.array([]),
            abs_omega: np.ndarray = np.array([])) -> None:
        if (self._UNI_OMEGA):
            self._coeff_op = np.zeros((len(center_omega), self._order_taylor))
            self._op = np.zeros((len(center_omega), len(self._omega)),
                                dtype=complex)
            self._coeff_op = self._coeff(center_omega, self._order_taylor).T
            for i in range(len(center_omega)):
                self._op[i] = Taylor.series(self._coeff_op[i], self._omega,
                                            self._start_taylor,
                                            skip=self._skip_taylor)
        else:
            shape = (abs_omega.shape[0], self._order_taylor+1,
                     abs_omega.shape[1])
            self._op = np.zeros(abs_omega.shape, dtype=complex)
            self._coeff_op = np.zeros(shape)
            if (cfg.MULTIPROCESSING):
                args = [(abs_omega[i], self._order_taylor)
                        for i in range(len(center_omega))]
                pool = mp.Pool(processes=mp.cpu_count())  # use all avail cores
                res_pool = pool.starmap(self._coeff, args)
                pool.close()
                for i in range(len(center_omega)):
                    self._coeff_op[i] = res_pool[i]
                    self._op[i] = Taylor.series(self._coeff_op[i], self._omega,
                                                self._start_taylor,
                                                skip=self._skip_taylor)
            else:
                for i in range(len(center_omega)):
                    self._coeff_op[i] = self._coeff(abs_omega[i],
                                                    self._order_taylor)
                    self._op[i] = Taylor.series(self._coeff_op[i], self._omega,
                                                self._start_taylor,
                                                skip=self._skip_taylor)
    # ==================================================================
    def term(self, waves: np.ndarray, id: int,
             corr_wave: Optional[np.ndarray] = None) -> np.ndarray:
        if (corr_wave is None):
            corr_wave = waves[id]

        return FFT.ifft_mult_fft(corr_wave, self.op(waves, id, corr_wave))
