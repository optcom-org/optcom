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
from optcom.equations.abstract_ampnlse import AbstractAmpNLSE
from optcom.equations.abstract_ampnlse import SEED_SPLIT
from optcom.equations.abstract_re_fiber import AbstractREFiber
from optcom.equations.gnlse import GNLSE
from optcom.equations.nlse import NLSE


TAYLOR_COEFF_TYPE_OPTIONAL = List[Union[List[float], Callable, None]]
FLOAT_COEFF_TYPE_OPTIONAL = List[Union[float, Callable, None]]


class AmpHNLSE(AbstractAmpNLSE):
    r"""Hybrid Non linear Schrodinger equations for fiber amplifier.

    Represent the different effects in the NLSE or the GNLSE
    for fiber amplifier. Each of the pump and seed pulse can be
    allocated to NLSE or GNLSE.

    """

    def __init__(self, re_fiber: AbstractREFiber,
                 alpha: TAYLOR_COEFF_TYPE_OPTIONAL = [None],
                 alpha_order: int = 0,
                 beta: TAYLOR_COEFF_TYPE_OPTIONAL = [None],
                 beta_order: int = 2, gain_order: int = 0,
                 gamma: FLOAT_COEFF_TYPE_OPTIONAL = [None],
                 sigma: float = cst.XPM_COEFF, eta: float = cst.XNL_COEFF,
                 h_R: Optional[Union[float, Callable]] = None,
                 f_R: float = cst.F_R,
                 n_core: FLOAT_COEFF_TYPE_OPTIONAL = [None],
                 n_clad: FLOAT_COEFF_TYPE_OPTIONAL = [None],
                 NA: FLOAT_COEFF_TYPE_OPTIONAL = [None],
                 v_nbr: FLOAT_COEFF_TYPE_OPTIONAL = [None],
                 eff_area: FLOAT_COEFF_TYPE_OPTIONAL = [None],
                 nl_index: FLOAT_COEFF_TYPE_OPTIONAL = [None],
                 overlap: FLOAT_COEFF_TYPE_OPTIONAL = [None],
                 sigma_a: FLOAT_COEFF_TYPE_OPTIONAL = [None],
                 sigma_e: FLOAT_COEFF_TYPE_OPTIONAL = [None],
                 en_sat: FLOAT_COEFF_TYPE_OPTIONAL = [None],
                 R_0: Union[float, Callable] = cst.R_0,
                 R_L: Union[float, Callable] = cst.R_L,
                 core_radius: float = cst.CORE_RADIUS,
                 clad_radius: float = cst.CLAD_RADIUS,
                 medium_core: str = cst.FIBER_MEDIUM_CORE,
                 medium_clad: str = cst.FIBER_MEDIUM_CLAD,
                 temperature: float = cst.TEMPERATURE,
                 ATT: List[bool] = [True, True],
                 DISP: List[bool] = [True, False],
                 SPM: List[bool] = [True, False],
                 XPM: List[bool] = [False, False],
                 FWM: List[bool] = [False, False],
                 SS: List[bool] = [False, False],
                 XNL: List[bool] = [False, False],
                 GAIN_SAT: bool = False, NOISE: bool = True,
                 split_noise_option: str = SEED_SPLIT,
                 UNI_OMEGA: List[bool] = [True, True],
                 STEP_UPDATE: bool = False, INTRA_COMP_DELAY: bool = True,
                 INTRA_PORT_DELAY: bool = True, INTER_PORT_DELAY: bool = False
                 ) -> None:
        r"""
        Parameters
        ----------
        re_fiber : AbstractREFiber
            The rate equations describing the fiber laser dynamics.
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
        gain_order :
            The order of the gain coefficients to take into account.
        gamma :
            The non linear coefficient.
            :math:`[rad\cdot W^{-1}\cdot km^{-1}]` If a callable is
            provided, variable must be angular frequency.
            :math:`[ps^{-1}]`
        sigma :
            Positive term multiplying the XPM terms of the NLSE.
        eta :
            Positive term multiplying the cross-non-linear terms of the
            NLSE.
        h_R :
            The Raman response function values.  If a callable is
            provided, variable must be time. :math:`[ps]`
        f_R :
            The fractional contribution of the delayed Raman response.
            :math:`[]`
        n_core :
            The refractive index of the core.  If a callable
            is provided, variable must be angular frequency.
            :math:`[ps^{-1}]` (1<=len(n_core)<=2 for signal and pump)
        n_clad :
            The refractive index of the cladding.  If a callable
            is provided, variable must be angular frequency.
            :math:`[ps^{-1}]` (1<=len(n_clad)<=2 for signal and pump)
        NA :
            The numerical aperture. :math:`[]`  If a callable
            is provided, variable must be angular frequency.
            :math:`[ps^{-1}]` (1<=len(NA)<=2 for signal and pump)
        v_nbr :
            The V number. :math:`[]`  If a callable
            is provided, variable must be angular frequency.
            :math:`[ps^{-1}]` (1<=len(v_nbr)<=2 for signal and pump)
        eff_area :
            The effective area. :math:`[\u m^2]` If a callable
            is provided, variable must be angular frequency.
            :math:`[ps^{-1}]` (1<=len(eff_area)<=2 for signal and pump)
        nl_index :
            The non-linear coefficient. :math:`[m^2\cdot W^{-1}]`  If a
            callable is provided, variable must be angular frequency.
            :math:`[ps^{-1}]` (1<=len(nl_index)<=2 for signal and pump)
        overlap :
            The overlap factor. :math:`[]`  If a callable
            is provided, variable must be angular frequency.
            :math:`[ps^{-1}]` (1<=len(overlap)<=2 for signal and pump)
        sigma_a :
            The absorption cross sections. :math:`[nm^2]`  If a callable
            is provided, variable must be angular frequency.
            :math:`[ps^{-1}]` (1<=len(sigma_a)<=2 for signal and pump)
        sigma_e :
            The emission cross sections. :math:`[nm^2]`  If a callable
            is provided, variable must be angular frequency.
            :math:`[ps^{-1}]` (1<=len(sigma_e)<=2 for signal and pump)
        en_sat :
            The saturation energy. :math:`[J]`  If a callable
            is provided, variable must be angular frequency.
            :math:`[ps^{-1}]` (1<=len(en_sat)<=2 for signal and pump)
        R_0 :
            The reflectivity at the fiber start.  If a callable
            is provided, variable must be angular frequency.
            :math:`[ps^{-1}]`
        R_L :
            The reflectivity at the fiber end.  If a callable
            is provided, variable must be angular frequency.
            :math:`[ps^{-1}]`
        core_radius :
            The radius of the core. :math:`[\mu m]`
        clad_radius :
            The radius of the cladding. :math:`[\mu m]`
        medium_core :
            The medium of the core.
        medium_clad :
            The medium of the cladding.
        temperature :
            The temperature of the medium. :math:`[K]`
        ATT :
            If True, trigger the attenuation. The first element is
            related to the seed and the second to the pump.
        DISP :
            If True, trigger the dispersion. The first element is
            related to the seed and the second to the pump.
        SPM :
            If True, trigger the self-phase modulation. The first
            element is related to the seed and the second to the pump.
        XPM :
            If True, trigger the cross-phase modulation. The first
            element is related to the seed and the second to the pump.
        FWM :
            If True, trigger the Four-Wave mixing. The first element is
            related to the seed and the second to the pump.
        SS :
            If True, trigger the self-steepening. The first element is
            related to the seed and the second to the pump.
        XNL :
            If True, trigger cross-non linear effects. The first element
            is related to the seed and the second to the pump.
        GAIN_SAT :
            If True, trigger the gain saturation.
        NOISE :
            If True, trigger the noise calculation.
        split_noise_option :
            The way the spontaneous emission power is split among the
            fields.
        UNI_OMEGA :
            If True, consider only the center omega for computation.
            Otherwise, considered omega discretization.  The first
            element is related to the seed and the second to the pump.
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
        super().__init__(re_fiber, gain_order, en_sat, R_0, R_L, temperature,
                         GAIN_SAT, NOISE, split_noise_option,
                         UNI_OMEGA, STEP_UPDATE)
        n_core_ = re_fiber.n_core if (n_core == [None]) else n_core
        n_clad_ = re_fiber.n_clad if (n_clad == [None]) else n_clad
        NA_ = re_fiber.NA if (NA == [None]) else NA
        v_nbr_ = re_fiber.v_nbr if (v_nbr == [None]) else v_nbr
        eff_area_ = re_fiber.eff_area if (eff_area == [None]) else eff_area
        alpha_ = util.make_list(alpha, 2)
        beta_ = util.make_list(beta, 2)
        gamma_ = util.make_list(gamma, 2)
        n_core_ = util.make_list(n_core_, 2)
        n_clad_ = util.make_list(n_clad_, 2)
        NA_ = util.make_list(NA_, 2)
        v_nbr_ = util.make_list(v_nbr_, 2)
        eff_area_ = util.make_list(eff_area_, 2)
        nl_index_ = util.make_list(nl_index, 2)
        ATT_ = util.make_list(ATT, 2)
        DISP_ = util.make_list(DISP, 2)
        SPM_ = util.make_list(SPM, 2)
        XPM_ = util.make_list(XPM, 2)
        FWM_ = util.make_list(FWM, 2)
        SS_ = util.make_list(SS, 2)
        XNL_ = util.make_list(XNL, 2)
        UNI_OMEGA_ = util.make_list(UNI_OMEGA, 2)
        nlse: List[Union[NLSE, GNLSE]] = []
        for k in range(4):
            i = k//2
            if (SS_[i]):
                nlse.append(GNLSE(alpha_[i], alpha_order, beta_[i], beta_order,
                                  gamma_[i], sigma, eta, h_R, f_R, core_radius,
                                  clad_radius, n_core_[i], n_clad_[i], NA_[i],
                                  v_nbr_[i], eff_area_[i], nl_index_[i],
                                  medium_core, medium_clad, temperature,
                                  ATT_[i], DISP_[i], SPM_[i], XPM_[i], FWM_[i],
                                  XNL_[i], NOISE, UNI_OMEGA_[i], STEP_UPDATE,
                                  INTRA_COMP_DELAY, INTRA_PORT_DELAY,
                                  INTER_PORT_DELAY))
            else:
                nlse.append(NLSE(alpha_[i], alpha_order, beta_[i], beta_order,
                                 gamma_[i], sigma, eta, h_R, f_R, core_radius,
                                 clad_radius, n_core_[i], n_clad_[i], NA_[i],
                                 v_nbr_[i], eff_area_[i], nl_index_[i],
                                 medium_core, medium_clad, temperature,
                                 ATT_[i], DISP_[i], SPM_[i], XPM_[i], FWM_[i],
                                 XNL_[i], NOISE, UNI_OMEGA_[i], STEP_UPDATE,
                                 INTRA_COMP_DELAY, INTRA_PORT_DELAY,
                                 INTER_PORT_DELAY))
        self._add_eq(nlse[0], 0)
        self._add_eq(nlse[1], 1)
        self._add_eq(nlse[2], 2)
        self._add_eq(nlse[3], 3)
