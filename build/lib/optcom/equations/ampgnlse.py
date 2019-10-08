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

from typing import Callable, List, Optional, Union

import numpy as np
from nptyping import Array

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.equations.ampnlse import AmpNLSE
from optcom.equations.abstract_equation import AbstractEquation
from optcom.utils.fft import FFT


class AmpGNLSE(AmpNLSE):
    r"""General non linear Schrodinger equations for fiber amplifier.

    Represent the different effects in the GNLSE for fiber amplifier.

    """

    def __init__(self, re_eq: AbstractEquation,
                 alpha: Optional[Union[List[float], Callable]] = None,
                 alpha_order: int = 1,
                 beta: Optional[Union[List[float], Callable]] = None,
                 beta_order: int = 2,
                 gamma: Optional[Union[float, Callable]] = None,
                 gain_order: int = 1,
                 sigma: float = cst.KERR_COEFF, tau_1: float = cst.TAU_1,
                 tau_2: float = cst.TAU_2, f_R: float = cst.F_R,
                 R_0: float = cst.R_0, R_L: float = cst.R_L,
                 nl_index: Union[float, Callable] = None,
                 ATT: bool = True, DISP: bool = True, SPM: bool = True,
                 XPM: bool = False, FWM: bool = False, GS: bool = True,
                 medium: str = cst.DEF_FIBER_MEDIUM,
                 dopant: str = cst.DEF_FIBER_DOPANT) -> None:
        r"""
        Parameters
        ----------
        re_eq : REFiber
            A fiber rate equation object.
        alpha :
            The derivatives of the attenuation coefficients.
            :math:`[km^{-1}, ps\cdot km^{-1}, ps^2\cdot km^{-1},
            ps^3\cdot km^{-1}, \ldots]` If a callable is provided,
            variable must be angular frequency. :math:`[ps^{-1}]`
        alpha_order :
            The order of alpha coefficients to take into account. (will
            be ignored if alpha values are provided - no file)
        beta :
            The derivatives of the propagation constant.
            :math:`[km^{-1}, ps\cdot km^{-1}, ps^2\cdot km^{-1},
            ps^3\cdot km^{-1}, \ldots]` If a callable is provided,
            variable must be angular frequency. :math:`[ps^{-1}]`
        beta_order :
            The order of beta coefficients to take into account. (will
            be ignored if beta values are provided - no file)
        gamma :
            The non linear coefficient.
            :math:`[rad\cdot W^{-1}\cdot km^{-1}]` If a callable is
            provided, variable must be angular frequency.
            :math:`[ps^{-1}]`
        gain_order :
            The order of the gain coefficients to take into account.
            (from the Rate Equations resolution)
        sigma :
            Positive term multiplying the XPM term of the NLSE
        tau_1 :
            The inverse of vibrational frequency of the fiber core
            molecules. :math:`[ps]`
        tau_2 :
            The damping time of vibrations. :math:`[ps]`
        f_R :
            The fractional contribution of the delayed Raman response.
            :math:`[]`
        R_0 :
            The reflectivity at the fiber start.
        R_L :
            The reflectivity at the fiber end.
        nl_index :
            The non linear index. Used to calculate the non linear
            parameter. :math:`[m^2\cdot W^{-1}]`
        ATT :
            If True, trigger the attenuation.
        DISP :
            If True, trigger the dispersion.
        SPM :
            If True, trigger the self-phase modulation.
        XPM :
            If True, trigger the cross-phase modulation.
        FWM :
            If True, trigger the Four-Wave mixing.
        GS :
            If True, trigger the gain saturation.
        medium :
            The main medium of the fiber amplifier.
        dopant :
            The doped medium of the fiber amplifier.

        """
        super().__init__(re_eq, alpha, alpha_order, beta, beta_order, gamma,
                         gain_order, sigma, tau_1, tau_2, f_R, R_0, R_L,
                         nl_index, ATT, DISP, SPM, XPM, FWM, GS, medium,
                         dopant)
    # ==================================================================
    def op_non_lin_rk4ip(self, waves: Array[cst.NPFT], id: int,
                         corr_wave: Optional[Array[cst.NPFT]] = None
                         ) -> Array[cst.NPFT]:
        C_nl = 1 + self._omega/self._center_omega[id]
        kerr = ((1.0-self._f_R)
                * self._effects_non_lin[0].op(waves, id, corr_wave))
        raman = self._f_R * self._effects_non_lin[1].op(waves, id, corr_wave)

        return C_nl * FFT.fft(kerr + raman)
    # ==================================================================
    def op_non_lin(self, waves: Array[cst.NPFT], id: int,
                   corr_wave: Optional[Array[cst.NPFT]] = None
                   ) -> Array[cst.NPFT]:
        r"""Represent the non linear effects of the approximated NLSE.

        Parameters
        ----------
        waves :
            The wave packet propagating in the fiber.
        id :
            The ID of the considered wave in the wave packet.
        corr_wave :
            Correction wave, use for consistency.

        Returns
        -------
        :
            The non linear term for the considered wave.

        Notes
        -----

        .. math:: \hat{N} = \mathcal{F}^{-1}\bigg\{i \gamma
                  \Big(1+\frac{\omega}{\omega_0}\Big)
                  \mathcal{F}\Big\{ (1-f_R) |A|^2
                  + f_R \mathcal{F}^{-1}\big\{\mathcal{F}\{h_R\}
                  \mathcal{F}\{|A|^2\}\big\}\Big\}\bigg\}

        """
        res = FFT.ifft(self.op_non_lin_rk4ip(waves, id) * waves[id])
        if (corr_wave is None):
            corr_wave = waves[id]
        res = np.zeros_like(res)
        res = np.divide(res, corr_wave, out=res, where=corr_wave!=0)

        return res
    # ==================================================================
    def term_non_lin(self, waves: Array[cst.NPFT], id: int,
                     corr_wave: Optional[Array[cst.NPFT]] = None
                     ) -> Array[cst.NPFT]:

        return self.op_non_lin(waves, id, np.ones(waves[id].shape,
                                                  dtype=cst.NPFT))
