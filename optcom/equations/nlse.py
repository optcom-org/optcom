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

from nptyping import Array

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.effects.kerr import Kerr
from optcom.effects.raman import Raman
from optcom.equations.abstract_nlse import AbstractNLSE


class NLSE(AbstractNLSE):
    r"""Non linear Schrodinger equations.

    Represent the different effects in the NLSE.

    Notes
    -----

    .. math:: \begin{split}
                \frac{\partial A_j}{\partial t}
                &= i\gamma \Bigg[\bigg[(1-f_R)|A_j(z,t)|^2
                + f_R\int_{-\infty}^{+\infty}h_R(s)|A_j(z,t-s)|^2 ds
                \bigg] \\
                &\quad + \sum_{k\neq j} \bigg[\sigma (1-f_R)|A_k(z,t)|^2
                + \eta f_R\int_{-\infty}^{+\infty}h_R(s)|A_k(z,t-s)|^2
                ds\bigg] \Bigg] A_j
              \end{split}

    """

    def __init__(self, alpha: Optional[Union[List[float], Callable]] = None,
                 alpha_order: int = 1,
                 beta: Optional[Union[List[float], Callable]] = None,
                 beta_order: int = 2,
                 gamma: Optional[Union[float, Callable]] = None,
                 sigma: float = cst.KERR_COEFF, tau_1: float = cst.TAU_1,
                 tau_2: float = cst.TAU_2, f_R: float = cst.F_R,
                 core_radius: float = cst.CORE_RADIUS,
                 NA: Union[float, Callable] = cst.NA, ATT: bool = True,
                 DISP: bool = True, SPM: bool = True, XPM: bool = False,
                 FWM: bool = False, medium: str = cst.DEF_FIBER_MEDIUM
                 ) -> None:
        r"""
        Parameters
        ----------
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
        core_radius :
            The radius of the core. :math:`[\mu m]`
        NA :
            The numerical aperture.
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
        medium :
            The main medium of the fiber.

        """
        super().__init__(alpha, alpha_order, beta, beta_order, gamma,
                         core_radius, NA, ATT, DISP, medium)
        self._f_R = f_R
        self._effects_non_lin.append(Kerr(SPM, XPM, FWM, sigma))
        self._effects_non_lin.append(Raman())
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
            The non linear term for the considered wave

        Notes
        -----

        .. math:: \hat{N} = i \gamma \Big[ (1-f_R) |A|^2
                  + f_R \mathcal{F}^{-1}
                  \big\{\mathcal{F}\{h_R\}\mathcal{F}\{|A|^2\}
                  \big\}\Big]

        """
        kerr = ((1.0-self._f_R)
                * self._effects_non_lin[0].op(waves, id, corr_wave))
        raman = self._f_R * self._effects_non_lin[1].op(waves, id, corr_wave)

        return self._gamma[id] * (kerr + raman)
