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
from optcom.equations.abstract_cnlse import AbstractCNLSE
from optcom.equations.anlse import ANLSE


OPTIONAL_LIST_CALL_FLOAT = Optional[Union[List[List[float]], List[Callable]]]


class CANLSE(AbstractCNLSE):
    """Coupled non linear Schrodinger equations.

    Represent the different effects in the NLSE as well as the
    interaction of NLSEs propagating along each others. Note that
    automatic calculation of the coupling coefficients rely on a formula
    that is only correct for symmetric coupler.

    Attributes
    ----------
    nbr_eqs : int
        Number of NLSEs in the CNLSE.

    """
    def __init__(self, nbr_fibers: int = 2,
                 alpha: OPTIONAL_LIST_CALL_FLOAT = None,
                 alpha_order: int = 1,
                 beta: OPTIONAL_LIST_CALL_FLOAT  = None,
                 beta_order: int = 2,
                 gamma: Optional[Union[List[float], List[Callable]]] = None,
                 kappa: Optional[List[List[List[float]]]] = None,
                 sigma: List[float] = [cst.KERR_COEFF],
                 sigma_cross: List[List[float]] = [[cst.KERR_COEFF_CROSS]],
                 eta: float = cst.XPM_COEFF,
                 T_R: float = cst.RAMAN_COEFF,
                 NA: Union[List[float], List[Callable]] = [cst.NA],
                 ATT: bool = True, DISP: bool = True, SPM: bool = True,
                 XPM: bool = False, FWM: bool = False, SS: bool = False,
                 RS: bool = False, ASYM: bool = True, COUP: bool = True,
                 approx_type: int = cst.DEFAULT_APPROX_TYPE,
                 medium: str = cst.DEF_FIBER_MEDIUM,
                 c2c_spacing: List[List[float]] = [[cst.C2C_SPACING]],
                 core_radius: List[float] = [cst.CORE_RADIUS],
                 V: List[float] = [cst.V], n_0: List[float] = [cst.REF_INDEX]
                 ) -> None:
        r"""
        Parameters
        ----------
        nbr_fibers :
            The number of fibers in the coupler.
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
        kappa :
            The coupling coefficients. :math:`[km^{-1}]`
        sigma :
            Positive term multiplying the XPM term of the NLSE.
        sigma_cross :
            Positive term multiplying the XPM term of the NLSE inbetween
            the fibers.
        eta :
            Positive term multiplying the XPM in other non linear
            terms of the NLSE.
        T_R :
            The raman coefficient. :math:`[]`
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
        ASYM :
            If True, trigger the asymmetry effects between cores.
        COUP :
            If True, trigger the coupling effects between cores.
        approx_type :
            The type of the NLSE approximation.
        medium :
            The main medium of the fiber.
        c2c_spacing :
            The center to center distance between two cores.
            :math:`[\mu m]`
        core_radius :
            The core radius. :math:`[\mu m]`
        V :
            The fiber parameter.
        n_0 :
            The refractive index outside of the waveguides.

        """
        super().__init__(nbr_fibers, beta, kappa, sigma_cross, c2c_spacing,
                         core_radius, V, n_0, ASYM, COUP, XPM, medium)
        alpha_ = util.make_list(alpha, nbr_fibers)
        beta_ = util.make_list(beta, nbr_fibers)
        gamma_ = util.make_list(gamma, nbr_fibers)
        sigma_ = util.make_list(sigma, nbr_fibers)
        NA_ = util.make_list(NA, nbr_fibers)
        core_radius_ = util.make_list(core_radius, nbr_fibers)
        for i in range(nbr_fibers):
            self._eqs[i].append(ANLSE(alpha_[i], alpha_order, beta_[i],
                                      beta_order, gamma_[i], sigma_[i], eta,
                                      T_R, core_radius_[i], NA_[i], ATT, DISP,
                                      SPM, XPM, FWM, SS, RS, approx_type,
                                      medium))
