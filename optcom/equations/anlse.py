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
from optcom.effects.self_steepening import SelfSteepening
from optcom.equations.abstract_nlse import AbstractNLSE


class ANLSE(AbstractNLSE):
    r"""Approximated non linear Schrodinger equations.

    Represent the different effects in the approximated NLSE.

    Notes
    -----

    NLSE approximation 1:

    .. math:: \frac{\partial A_j}{\partial t} = i\gamma \Bigg[ |A_j|^2
              + \sigma \sum_{k\neq j}|A_k|^2 + i S \frac{1}{A_j}
              \bigg(\frac{\partial |A_j|^2 A_j}{\partial t}
              + \eta \sum_{k\neq j}
              \frac{\partial |A_k|^2 A_j}{\partial t} \bigg)
              - T_R \bigg(\frac{\partial |A_j|^2}{\partial t}
              + \eta \sum_{k\neq j} \frac{\partial |A_k|^2}{\partial t}
              \bigg) \Bigg] A_j

    NLSE approximation 2:

    .. math:: \frac{\partial A_j}{\partial t} = i\gamma \Bigg[ |A_j|^2
              + \sigma \sum_{k\neq j}|A_k|^2 + i S \bigg(A^{*}_j
              + \frac{\eta}{A_j} \sum_{k\neq j}|A_k|^2\bigg)
              \frac{\partial A_j}{\partial t}
              + \big(i S - T_R\big) \frac{\partial |A_j|^2}{\partial t}
              + \eta \big(i S - T_R\big) \sum_{k\neq j}
              \frac{\partial |A_k|^2}{\partial t} \Bigg] A_j

    NLSE approximation 3:

    .. math:: \frac{\partial A_j}{\partial t} = i\gamma \Bigg[ |A_j|^2
              + \sigma \sum_{k\neq j}|A_k|^2
              + \bigg((2 i S - T_R) A^{*}_j
              + i S \frac{\eta}{A_j} \sum_{k\neq j} |A_k|^2\bigg)
              \frac{\partial A_j}{\partial t}
              + \big(i S - T_R\big) A_j
              \frac{\partial A^{*}_j}{\partial t}
              + \eta \big(i S - T_R\big) \sum_{k\neq j}
              \frac{\partial |A_k|^2}{\partial t} \Bigg] A_j

    NLSE approximation 1:

    .. math:: \begin{split}
              \hat{N}(A_j) = i\gamma \Bigg[ |A_j|^2
              + \sigma \sum_{k\neq j}|A_k|^2
              + i S \frac{1}{A_j}
              \mathcal{F}^{-1}\bigg\{ (-i \omega)
              \mathcal{F}\Big\{|A_j|^2 A_j
              + \eta \sum_{k\neq j}|A_k|^2 A_j\Big\}\bigg\} \\
              \quad\quad\quad
              - T_R \mathcal{F}^{-1}\bigg\{(-i\omega)
              \mathcal{F}\Big\{|A_j|^2+ \eta \sum_{k\neq j}|A_k|^2
              \Big\}\bigg\} \Bigg]
              \end{split}

    NLSE approximation 2:

    .. math:: \begin{split}
              \hat{N}(A_j) = i\gamma \Bigg[ |A_j|^2
              + \sigma\sum_{k\neq j}|A_k|^2
              + i S \Big(A_j^*
              + \eta \sum_{k\neq j}|A_k|^2\Big)
              \mathcal{F}^{-1}\big\{ (-i \omega)
              \mathcal{F}\{A_j\}\big\}\\
              \quad\quad\quad
              + \big(i S - T_R\big)\mathcal{F}^{-1}\bigg\{(-i\omega)
              \mathcal{F}\Big\{|A_j|^2
              + \eta \sum_{k\neq j}|A_k|^2\Big\}\bigg\} \Bigg]
              \end{split}

    NLSE approximation 3:

    .. math:: \begin{split}
              \hat{N}(A_j) = i\gamma \Bigg[ |A_j|^2
              + \sigma\sum_{k\neq j}|A_k|^2
              + \bigg((2 i S - T_R) A_j^*
              + i\eta S \sum_{k\neq j}|A_k|^2\bigg)
              \mathcal{F}^{-1}\big\{ (- i \omega)
              \mathcal{F}\{A_j\}\big\} \\
              \quad\quad\quad
              + \big(i S - T_R\big) A_j
              \mathcal{F}^{-1}\big\{(-i\omega)\mathcal{F}\{A_j^*\}\big\}
              + \eta\big(i S - T_R\big)
              \mathcal{F}^{-1}\bigg\{ (- i \omega)
              \mathcal{F}\Big\{\sum_{k\neq j}|A_k|^2\Big\}\bigg\}\Bigg]
              \end{split}

    """

    def __init__(self, alpha: Optional[Union[List[float], Callable]] = None,
                 alpha_order: int = 1,
                 beta: Optional[Union[List[float], Callable]] = None,
                 beta_order: int = 2,
                 gamma: Optional[Union[float, Callable]] = None,
                 sigma: float = cst.KERR_COEFF, eta: float = cst.XPM_COEFF,
                 T_R: float = cst.RAMAN_COEFF,
                 core_radius: float = cst.CORE_RADIUS,
                 NA: Union[float, Callable] = cst.NA, ATT: bool = True,
                 DISP: bool = True, SPM: bool = True, XPM: bool = False,
                 FWM: bool = False, SS: bool = False, RS: bool = False,
                 approx_type: int = cst.DEFAULT_APPROX_TYPE,
                 medium: str = cst.DEF_FIBER_MEDIUM) -> None:
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
        eta :
            Positive term multiplying the XPM in other non linear
            terms of the NLSE
        T_R :
            The raman coefficient. :math:`[]`
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
        SS :
            If True, trigger the self-steepening.
        RS :
            If True, trigger the Raman scattering.
        approx_type :
            The type of the NLSE approximation.
        medium :
            The main medium of the fiber.

        """
        super().__init__(alpha, alpha_order, beta, beta_order, gamma,
                         core_radius, NA, ATT, DISP, medium)
        if (XPM or SPM or FWM):
            self._effects_non_lin.append(Kerr(SPM, XPM, FWM, sigma))
        if (SS):
            self._effects_non_lin.append(SelfSteepening(eta, approx_type, XPM))
        if (RS):
            self._effects_non_lin.append(Raman(T_R, eta, approx_type, XPM))
        self._op_type = "_approx"
    # ==================================================================
    def op_non_lin(self, waves: Array[cst.NPFT], id: int,
                   corr_wave: Optional[Array[cst.NPFT]] = None
                   ) -> Array[cst.NPFT]:
        r"""Represent the non linear effects of the approximated NLSE.
        """

        return self._gamma[id] * super().op_non_lin(waves, id, corr_wave)
