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
from typing import Callable, List, Optional, Union

import numpy as np

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.effects.abstract_effect import AbstractEffect
from optcom.field import Field
from optcom.utils.fft import FFT


class ActiveFiberPhotonProcess(AbstractEffect):
    r"""The process photon can encounter in an active fiber, i.e.
    stimulated emission, spontaneous emission, absorption.

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

    """

    def __init__(self, sigma: Union[float, Callable],
                 Gamma: Union[float, Callable], doped_area: float,
                 UNI_OMEGA: bool = True) -> None:
        r"""
        Parameters
        ----------
        sigma :
            The absorption cross sections. :math:`[nm^2]`  If a callable
            is prodived, variable must be wavelength. :math:`[nm]`
        Gamma :
            The overlap factor. If a callable is provided, variable must
            be angular frequency. :math:`[ps^{-1}]`
        doped_area :
            The doped area. :math:`[\mu m^2]`
        UNI_OMEGA :
            If True, consider only the center omega for computation.
            Otherwise, considered omega discretization.

        """
        super().__init__()
        self._UNI_OMEGA = UNI_OMEGA
        self._doped_area: float = doped_area * 1e6  # um^2 -> nm^2
        self._factor: float = 1.0 / (cst.HBAR*self._doped_area)
        # Cross section sigma ------------------------------------------
        self._sigma_op: np.ndarray = np.array([])
        self._sigma: Union[np.ndarray, Callable]
        if (callable(sigma)):
            self._sigma = sigma
        else:
            self._sigma = lambda omega: np.ones_like(omega) * sigma
        # Overlap factor Gamma -----------------------------------------
        self._op: np.ndarray = np.array([])
        self._Gamma_op: np.ndarray = np.array([])
        self._Gamma: Union[np.ndarray, Callable]
        if (callable(Gamma)):
            self._Gamma = Gamma
        else:
            self._Gamma = lambda omega: np.ones_like(omega) * Gamma
    # ==================================================================
    def set(self, center_omega: np.ndarray = np.array([]),
            abs_omega: np.ndarray = np.array([])) -> None:
        if (self._UNI_OMEGA):
            self._op = np.zeros_like(center_omega)
            self._sigma_op = np.zeros_like(center_omega)
            self._Gamma_op = np.zeros_like(center_omega)
            self._sigma_op = self._sigma(center_omega)
            self._Gamma_op = self._Gamma(center_omega)
            for i in range(len(center_omega)):
                self._op[i] = (self._factor * self._sigma_op[i]
                               * self._Gamma_op[i] / center_omega[i])
        else:
            self._op = np.zeros_like(abs_omega)
            self._sigma_op = np.zeros_like(abs_omega)
            self._Gamma_op = np.zeros_like(abs_omega)
            for i in range(len(abs_omega)):
                self._sigma_op[i] = self._sigma(abs_omega[i])
                self._Gamma_op[i] = self._Gamma(abs_omega[i])
                self._op[i] = (self._factor * self._sigma_op[i]
                               * self._Gamma_op[i] / abs_omega[i])
    # ==================================================================
    def op(self, waves: np.ndarray, id: int,
           corr_wave: Optional[np.ndarray] = None) -> np.ndarray:
        # ps^2 nm^{-2} kg^{-1} -> s^2 m^{-2} kg^{-1} = J^{-1}

        return self._op[id] * 1e-6
    # ==================================================================
    def term(self, waves: np.ndarray, id: int,
             corr_wave: Optional[np.ndarray] = None) -> np.ndarray:
        """The operator of the emission effect."""

        power = FFT.fftshift(Field.spectral_power(waves[id]))
        power_sum = np.sum(power)
        power = (power / power_sum) if (power_sum) else (power * 0.0)

        power *= Field.average_power(waves[id], self.dtime, self.rep_freq[id])

        return np.real(np.sum(self.op(waves, id, corr_wave) * power))
