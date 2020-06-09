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

from typing import Callable, Optional, Union

import numpy as np

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.effects.abstract_effect import AbstractEffect
from optcom.utils.fft import FFT


class Raman(AbstractEffect):
    r"""The Raman scattering effect.

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
    self_term :
        If True, trigger the self-term of the effect.
    cross_term :
        If True, trigger the cross-term influence in the effect.
    eta :
        Positive term multiplying the cross terms in the effect.


    """

    def __init__(self, h_R: Union[float, Callable], self_term: bool = True,
                 cross_term: bool = False, eta: float = cst.XNL_COEFF) -> None:
        r"""
        Parameters
        ----------
        h_R :
            The Raman response function values.  If a callable is
            provided, variable must be time. :math:`[ps]`
        self_term :
            If True, trigger the self-term of the effect.
        cross_term :
            If True, trigger the cross-term influence in the effect.
        eta :
            Positive term multiplying the cross terms in the effect.

        """
        super().__init__()
        self.self_term: bool = self_term
        self.cross_term: bool = cross_term
        self.eta: float = eta
        # Raman response function --------------------------------------
        self._h_R_op: np.ndarray = np.array([])
        self._h_R: Union[np.ndarray, Callable]
        if (callable(h_R)):
            self._h_R = h_R
        else:
            self._h_R = lambda omega: np.ones(omega) * h_R
    # ==================================================================
    def set(self, center_omega: np.ndarray = np.array([]),
            abs_omega: np.ndarray = np.array([])) -> None:
        self._h_R_op = self._h_R(self._time)
    # ==================================================================
    def op(self, waves: np.ndarray, id: int,
           corr_wave: Optional[np.ndarray] = None) -> np.ndarray:
        """The operator of the Raman effect."""
        res = np.zeros(waves[id].shape, dtype=cst.NPFT)
        if (self.self_term):
            res += self._op_self(waves, id, corr_wave)
        if (self.cross_term):
            res += self.eta * self._op_cross(waves, id, corr_wave)

        return res
    # ==================================================================
    def _op_self(self, waves: np.ndarray, id: int,
                corr_wave: Optional[np.ndarray] = None) -> np.ndarray:
        """The operator of the Raman effect for the considered wave."""
        square_mod = waves[id]*np.conj(waves[id])

        return (1j * FFT.conv_to_fft(self._h_R_op, square_mod))
    # ==================================================================
    def _op_cross(self, waves: np.ndarray, id: int,
                 corr_wave: Optional[np.ndarray] = None) -> np.ndarray:
        """The cross terms of the Raman effect."""
        square_mod = np.zeros(waves[id].shape, dtype=cst.NPFT)
        for i in range(len(waves)):
            if (i != id):
                square_mod += waves[i]*np.conj(waves[i])

        return (1j * FFT.conv_to_fft(self._h_R_op, square_mod))
