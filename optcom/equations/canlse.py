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

from typing import Callable, List, Optional, Union

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.effects.kerr import Kerr
from optcom.effects.raman import Raman
from optcom.effects.raman_approx import RamanApprox
from optcom.equations.abstract_cnlse import AbstractCNLSE
from optcom.equations.anlse import ANLSE
from optcom.parameters.refractive_index.sellmeier import Sellmeier


TAYLOR_COEFF_TYPE_OPTIONAL = List[Union[List[float], Callable, None]]
FLOAT_COEFF_TYPE_OPTIONAL = List[Union[float, Callable, None]]
TAYLOR_COUP_COEFF_OPTIONAL = List[List[Union[List[float], Callable, None]]]



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
                 alpha: TAYLOR_COEFF_TYPE_OPTIONAL = [None],
                 alpha_order: int = 0,
                 beta: TAYLOR_COEFF_TYPE_OPTIONAL = [None],
                 beta_order: int = 2,
                 gamma: FLOAT_COEFF_TYPE_OPTIONAL = [None],
                 kappa: TAYLOR_COUP_COEFF_OPTIONAL = [[None]],
                 sigma: List[float] = [cst.XPM_COEFF],
                 sigma_cross: List[List[float]] = [[cst.XPM_COEFF_CROSS]],
                 eta: List[float] = [cst.XNL_COEFF],
                 eta_cross: List[List[float]] = [[cst.XNL_COEFF_CROSS]],
                 T_R: List[float] = [cst.RAMAN_COEFF],
                 core_radius: List[float] = [cst.CORE_RADIUS],
                 clad_radius: float = cst.CLAD_RADIUS_COUP,
                 c2c_spacing: List[List[float]] = [[cst.C2C_SPACING]],
                 n_core: FLOAT_COEFF_TYPE_OPTIONAL = [None],
                 n_clad: Optional[Union[float, Callable]] = None,
                 NA: FLOAT_COEFF_TYPE_OPTIONAL = [None],
                 v_nbr: FLOAT_COEFF_TYPE_OPTIONAL = [None],
                 eff_area: FLOAT_COEFF_TYPE_OPTIONAL = [None],
                 nl_index: FLOAT_COEFF_TYPE_OPTIONAL = [None],
                 medium_core: List[str] = [cst.FIBER_MEDIUM_CORE],
                 medium_clad: str = cst.FIBER_MEDIUM_CLAD,
                 temperature: float = cst.TEMPERATURE,
                 ATT: bool = True, DISP: bool = True, SPM: bool = True,
                 XPM: bool = False, FWM: bool = False, SS: bool = False,
                 RS: bool = False, XNL: bool = False, ASYM: bool = True,
                 COUP: bool = True, NOISE: bool = True,
                 approx_type: int = cst.DEFAULT_APPROX_TYPE,
                 UNI_OMEGA: bool = True, STEP_UPDATE: bool = True,
                 INTRA_COMP_DELAY: bool = True, INTRA_PORT_DELAY: bool = True,
                 INTER_PORT_DELAY: bool = False) -> None:
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
            Positive term multiplying the XPM terms of the NLSE.
        sigma_cross :
            Positive term multiplying the XPM terms of the NLSE
            inbetween the fibers.
        eta :
            Positive term multiplying the cross-non-linear terms of the
            NLSE.
        eta_cross :
            Positive term multiplying the cross-non-linear terms of the
            NLSE inbetween the fibers.
        T_R :
            The raman coefficient. :math:`[]`
        core_radius :
            The core radius. :math:`[\mu m]`
        clad_radius :
            The radius of the cladding. :math:`[\mu m]`
        c2c_spacing :
            The center to center distance between two cores.
            :math:`[\mu m]`
        n_core :
            The refractive index of the core.  If a callable is
            provided, variable must be angular frequency.
            :math:`[ps^{-1}]`
        n_clad :
            The refractive index of the clading.  If a callable is
            provided, variable must be angular frequency.
            :math:`[ps^{-1}]`
        NA :
            The numerical aperture.  If a callable is provided, variable
            must be angular frequency. :math:`[ps^{-1}]`
        v_nbr :
            The V number.  If a callable is provided, variable must be
            angular frequency. :math:`[ps^{-1}]`
        eff_area :
            The effective area.  If a callable is provided, variable
            must be angular frequency. :math:`[ps^{-1}]`
        nl_index :
            The non-linear coefficient.  If a callable is provided,
            variable must be angular frequency. :math:`[ps^{-1}]`
        medium_core :
            The medium of the fiber core.
        medium_clad :
            The medium of the fiber cladding.
        temperature :
            The temperature of the fiber. :math:`[K]`
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
        SS : bool
            If True, trigger the self-steepening.
        RS :
            If True, trigger the Raman scattering.
        XNL :
            If True, trigger cross-non linear effects.
        ASYM :
            If True, trigger the asymmetry effects between cores.
        COUP :
            If True, trigger the coupling effects between cores.
        NOISE :
            If True, trigger the noise calculation.
        approx_type :
            The type of the NLSE approximation.
        UNI_OMEGA :
            If True, consider only the center omega for computation.
            Otherwise, considered omega discretization.
        STEP_UPDATE :
            If True, update fiber parameters at each spatial sub-step.
        INTRA_COMP_DELAY :
            If True, take into account the relative time difference,
            between all waves, that is acquired while propagating
            in the component.
        INTRA_PORT_DELAY :
            If True, take into account the initial relative time
            difference between channels of all fields but for each port.
        INTER_PORT_DELAY :
            If True, take into account the initial relative time
            difference between channels of all fields of all ports.

        """
        alpha_ = util.make_list(alpha, nbr_fibers)
        beta_ = util.make_list(beta, nbr_fibers)
        gamma_ = util.make_list(gamma, nbr_fibers)
        sigma_ = util.make_list(sigma, nbr_fibers)
        eta_ = util.make_list(eta, nbr_fibers)
        T_R_ = util.make_list(T_R, nbr_fibers)
        core_radius_ = util.make_list(core_radius, nbr_fibers)
        n_core_ = util.make_list(n_core, nbr_fibers)
        NA_ = util.make_list(NA, nbr_fibers)
        v_nbr_ = util.make_list(v_nbr, nbr_fibers)
        eff_area_ = util.make_list(eff_area, nbr_fibers)
        nl_index_ = util.make_list(nl_index, nbr_fibers)
        medium_core_ = util.make_list(medium_core, nbr_fibers)
        c2c_spacing_ = util.make_matrix(c2c_spacing, nbr_fibers, nbr_fibers,
                                        sym=True)
        clad_radius_ = [min(c2c_spacing_[i][:i] + c2c_spacing_[i][i+1:])
                        for i in range(len(c2c_spacing_))]
        n_clad_ = Sellmeier(medium=medium_clad) if n_clad is None else n_clad

        nlses: List[ANLSE] = []
        for i in range(nbr_fibers):
            nlses.append(ANLSE(alpha_[i], alpha_order, beta_[i], beta_order,
                               gamma_[i], sigma_[i], eta_[i], T_R_[i],
                               core_radius_[i], clad_radius_[i], n_core_[i],
                               n_clad_, NA_[i], v_nbr_[i], eff_area_[i],
                               nl_index_[i], medium_core_[i], medium_clad,
                               temperature, ATT, DISP, SPM, XPM, FWM, SS, RS,
                               XNL, NOISE, approx_type, UNI_OMEGA, STEP_UPDATE,
                               INTRA_COMP_DELAY, INTRA_PORT_DELAY,
                               INTER_PORT_DELAY))

        beta__: List[Union[List[float], Callable]] = \
            [nlse._beta for nlse in nlses]
        v_nbr__: List[Union[float, Callable]] = \
            [nlse._v_nbr for nlse in nlses]
        kappa_ = util.make_list(kappa, nbr_fibers)
        kappa_ = util.make_matrix(kappa_, nbr_fibers, nbr_fibers, sym=True)
        sigma_cross_ = util.make_matrix(sigma_cross, nbr_fibers, nbr_fibers,
                                        sym=True)
        eta_cross_ = util.make_matrix(eta_cross, nbr_fibers, nbr_fibers,
                                      sym=True)
        kerr_effects: List[Optional[Kerr]] = \
            [nlse._kerr for nlse in nlses]
        raman_effects: List[Optional[Union[Raman, RamanApprox]]] = \
            [nlse._raman for nlse in nlses]

        super().__init__(nbr_fibers, beta__, kappa_, sigma_cross_, eta_cross_,
                         core_radius_, clad_radius, c2c_spacing_, n_clad_,
                         v_nbr__, temperature, ASYM, COUP, XPM, FWM, XNL,
                         NOISE, STEP_UPDATE, INTRA_COMP_DELAY,
                         INTRA_PORT_DELAY, INTER_PORT_DELAY, kerr_effects,
                         raman_effects)

        for i in range(nbr_fibers):
            self._add_eq(nlses[i], i)
