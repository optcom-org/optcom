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

from typing import Optional, Union

import numpy as np
from nptyping import Array

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.effects.abstract_effect import AbstractEffect
from optcom.utils.fft import FFT


class SelfSteepening(AbstractEffect):
    r"""Generic class for effect object.

    Attributes
    ----------
    omega : numpy.ndarray of float
        The angular frequency array. :math:`[ps^{-1}]`
    time : numpy.ndarray of float
        The time array. :math:`[ps]`
    center_omega : numpy.ndarray of float
        The center angular frequency. :math:`[ps^{-1}]`

    """

    def __init__(self, eta: float = cst.XPM_COEFF,
                 approx_type: int = cst.DEFAULT_APPROX_TYPE,
                 cross_term: bool = False,
                 omega: Optional[Array[float]] = None,
                 center_omega: Optional[Array[float]] = None) -> None:
        r"""
        Parameters
        ----------
        eta :
            Positive term multiplying the cross terms in the effect.
        approx_type :
            The type of the NLSE approximation.
        cross_term :
            If True, trigger the cross-term influence in the effect.
        omega :
            The angular frequencies values of the field.
            :math:`[rad\cdot ps^{-1}]`
        center_omega :
            The center angular frequency. :math:`[rad\cdot ps^{-1}]`

        """
        super().__init__(omega=omega, center_omega=center_omega)
        self._eta: float = eta
        self._approx_type: int = approx_type
        self._cross_term: float = cross_term
        self._S: Optional[Array[float]] = None
    # ==================================================================
    @property
    def S(self) -> Array[float]:
        if (self._S is not None):

            return self._S
        else:
            if (self._omega is None):
                util.warning_terminal("Must specified the center omega "
                    "to calculate the self steepening coefficient, has been "
                    "evaluated at zero.")

                return np.zeros(self._center_omega.shape)
            else:

                return 1.0 / self._center_omega
    # ==================================================================
    def op_approx(self, waves: Array[cst.NPFT], id: int,
                  corr_wave: Optional[Array[cst.NPFT]] = None
                  ) -> Array[cst.NPFT]:
        """The approximation of the operator of the
        self-steepening effect."""
        res = self.op_approx_self(waves, id, corr_wave)
        if (self._cross_term):
            res += self.op_approx_cross(waves, id, corr_wave)

        return res
    # ==================================================================
    def op_approx_self(self, waves: Array[cst.NPFT], id: int,
                       corr_wave: Optional[Array[cst.NPFT]] = None
                       ) -> Array[cst.NPFT]:
        """The approximation of the operator of the
        self-steepening effect for the considered wave."""
        A = waves[id]
        res = np.zeros(A.shape, dtype=cst.NPFT)

        if (self._approx_type == cst.approx_type_1):
            res = FFT.dt_to_fft(A*np.conj(A)*A, self._omega, 1)
            if (corr_wave is None):
                corr_wave = waves[id]
            res[corr_wave==0] = 0
            res = np.divide(res, corr_wave, out=res, where=corr_wave!=0)

        if (self._approx_type == cst.approx_type_2):

            res = (np.conj(A)*FFT.dt_to_fft(A, self._omega, 1)
                   + FFT.dt_to_fft(A*np.conj(A), self._omega, 1))

        if (self._approx_type == cst.approx_type_3):

            res = (2*np.conj(A)*FFT.dt_to_fft(A, self._omega, 1)
                   + A*FFT.dt_to_fft(np.conj(A), self._omega, 1))

        return -1 * self.S[id] * res
    # ==================================================================
    def op_approx_cross(self, waves: Array[cst.NPFT], id: int,
                        corr_wave: Optional[Array[cst.NPFT]] = None
                        ) -> Array[cst.NPFT]:
        """The approximation of the operator of the cross terms of the
        self-steepening effect."""
        A = waves[id]
        res = np.zeros(A.shape, dtype=cst.NPFT)

        if (corr_wave is None):
            corr_wave = waves[id]

        if (self._approx_type == cst.approx_type_1):
            for i in range(len(waves)):
                if (i != id):
                    res += waves[i]*np.conj(waves[i])*A
            res = FFT.dt_to_fft(res, self._omega, 1)
            res[corr_wave==0] = 0
            res = np.divide(res, corr_wave, out=res, where=corr_wave!=0)

        if (self._approx_type == cst.approx_type_2
                or self._approx_type == cst.approx_type_3):
            for i in range(len(waves)):
                if (i != id):
                    res += waves[i] * np.conj(waves[i])
            res_ = np.zeros(A.shape, dtype=cst.NPFT)
            res_ = np.divide(res, corr_wave, out=res_, where=corr_wave!=0)
            res = (res_*FFT.dt_to_fft(A, self._omega, 1)
                   + FFT.dt_to_fft(res, self._omega, 1))

        return -1 * self.S[id] * self._eta * res
    # ==================================================================
    def term_approx(self, waves: Array[cst.NPFT], id: int,
                    corr_wave: Optional[Array[cst.NPFT]] = None
                    ) -> Array[cst.NPFT]:
        corr_wave = np.ones(waves[id].shape, dtype=cst.NPFT)

        return self.op_approx(waves, id, corr_wave)
