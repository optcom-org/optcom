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

from typing import Optional, Union

import numpy as np

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.effects.self_steepening import SelfSteepening
from optcom.utils.fft import FFT


class SelfSteepeningApprox(SelfSteepening):
    r"""The approximation of the self-steepening effect.

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
    approx_type :
        The type of the NLSE approximation.

    """

    def __init__(self, self_term: bool = True, cross_term: bool = False,
                 eta: float = cst.XNL_COEFF,
                 approx_type: int = cst.DEFAULT_APPROX_TYPE) -> None:
        r"""
        Parameters
        ----------
        self_term :
            If True, trigger the self-term of the effect.
        cross_term :
            If True, trigger the cross-term influence in the effect.
        eta :
            Positive term multiplying the cross terms in the effect.
        approx_type :
            The type of the NLSE approximation.

        """
        super().__init__()
        self.eta = eta
        self.self_term: bool = self_term
        self.cross_term: bool = cross_term
        self.approx_type: int = approx_type
    # ==================================================================
    def op(self, waves: np.ndarray, id: int,
           corr_wave: Optional[np.ndarray] = None) -> np.ndarray:
        """The approximation of the operator of the
        self-steepening effect."""
        res = np.zeros(waves[id].shape, dtype=cst.NPFT)
        if (self.self_term):
            res += self._op_approx_self(waves, id, corr_wave)
        if (self.cross_term):
            res += self.eta * self._op_approx_cross(waves, id, corr_wave)

        return res
    # ==================================================================
    def _op_approx_self(self, waves: np.ndarray, id: int,
                        corr_wave: Optional[np.ndarray] = None) -> np.ndarray:
        """The approximation of the operator of the
        self-steepening effect for the considered wave."""
        A = waves[id]
        res = np.zeros(A.shape, dtype=cst.NPFT)

        if (self.approx_type == cst.approx_type_1):
            res = FFT.dt_to_fft(A*np.conj(A)*A, self._omega, 1)
            if (corr_wave is None):
                corr_wave = waves[id]
            res[corr_wave==0] = 0
            res = np.divide(res, corr_wave, out=res, where=corr_wave!=0)

        if (self.approx_type == cst.approx_type_2):

            res = (np.conj(A)*FFT.dt_to_fft(A, self._omega, 1)
                   + FFT.dt_to_fft(A*np.conj(A), self._omega, 1))

        if (self.approx_type == cst.approx_type_3):

            res = (2*np.conj(A)*FFT.dt_to_fft(A, self._omega, 1)
                   + A*FFT.dt_to_fft(np.conj(A), self._omega, 1))

        return -1 * self._S[id] * res
    # ==================================================================
    def _op_approx_cross(self, waves: np.ndarray, id: int,
                         corr_wave: Optional[np.ndarray] = None) -> np.ndarray:
        """The approximation of the operator of the cross terms of the
        self-steepening effect."""
        A = waves[id]
        res = np.zeros(A.shape, dtype=cst.NPFT)

        if (corr_wave is None):
            corr_wave = waves[id]

        if (self.approx_type == cst.approx_type_1):
            for i in range(len(waves)):
                if (i != id):
                    res += waves[i] * np.conj(waves[i]) * A
            res = FFT.dt_to_fft(res, self._omega, 1)
            res[corr_wave==0] = 0.
            res = np.divide(res, corr_wave, out=res, where=corr_wave!=0)

        if (self.approx_type == cst.approx_type_2
                or self.approx_type == cst.approx_type_3):
            for i in range(len(waves)):
                if (i != id):
                    res += waves[i] * np.conj(waves[i])
            res_ = np.zeros(A.shape, dtype=cst.NPFT)
            res_ = np.divide(res, corr_wave, out=res_, where=corr_wave!=0)
            res = (res_*FFT.dt_to_fft(A, self._omega, 1)
                   + FFT.dt_to_fft(res, self._omega, 1))

        return -1 * self._S[id] * res
    # ==================================================================
    def term(self, waves: np.ndarray, id: int,
             corr_wave: Optional[np.ndarray] = None) -> np.ndarray:

        return self.op(waves, id, np.ones(waves[id].shape, dtype=cst.NPFT))
