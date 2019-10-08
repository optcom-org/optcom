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

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.effects.kerr import Kerr
from optcom.effects.raman import Raman
from optcom.effects.self_steepening import SelfSteepening
from optcom.equations.abstract_ampnlse import AbstractAmpNLSE
from optcom.equations.abstract_equation import AbstractEquation


class AmpANLSE(AbstractAmpNLSE):
    r"""Approximated non linear Schrodinger equations for fiber
    amplifier.

    Represent the different effects in the approximated NLSE for fiber
    amplifier.
    """

    def __init__(self, re_eq: AbstractEquation,
                 alpha: Optional[Union[List[float], Callable]] = None,
                 alpha_order: int = 1,
                 beta: Optional[Union[List[float], Callable]] = None,
                 beta_order: int = 2,
                 gamma: Optional[Union[float, Callable]] = None,
                 gain_order: int = 1,
                 sigma: float = cst.KERR_COEFF, eta: float = cst.XPM_COEFF,
                 T_R: float = cst.RAMAN_COEFF, R_0: float = cst.R_0,
                 R_L: float = cst.R_L,
                 nl_index: Union[float, Callable] = None,
                 ATT: bool = True, DISP: bool = True, SPM: bool = True,
                 XPM: bool = False, FWM: bool = False, SS: bool = False,
                 RS: bool = False, GS: bool = True,
                 approx_type: int = cst.DEFAULT_APPROX_TYPE,
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
        eta :
            Positive term multiplying the XPM in other non linear
            terms of the NLSE
        T_R :
            The raman coefficient. :math:`[]`
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
        SS :
            If True, trigger the self-steepening.
        RS :
            If True, trigger the Raman scattering.
        GS :
            If True, trigger the gain saturation.
        approx_type :
            The type of the NLSE approximation.
        medium :
            The main medium of the fiber amplifier.
        dopant :
            The doped medium of the fiber amplifier.

        """
        super().__init__(re_eq, alpha, alpha_order, beta, beta_order, gamma,
                         gain_order, nl_index, R_0, R_L, ATT, DISP, GS, medium,
                         dopant)
        if (XPM or SPM or FWM):
            self._effects_non_lin.append(Kerr(SPM, XPM, FWM, sigma))
        if (SS):
            self._effects_non_lin.append(SelfSteepening(eta, approx_type, XPM))
        if (RS):
            self._effects_non_lin.append(Raman(T_R, eta, approx_type, XPM))
        self._op_type = "_approx"
