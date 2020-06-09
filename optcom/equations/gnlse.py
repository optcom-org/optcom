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

import copy
from typing import Callable, List, Optional, Union

import numpy as np

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.domain import Domain
from optcom.effects.abstract_effect import AbstractEffect
from optcom.effects.kerr import Kerr
from optcom.effects.raman import Raman
from optcom.effects.self_steepening import SelfSteepening
from optcom.equations.abstract_nlse import AbstractNLSE
from optcom.equations.abstract_nlse import sync_waves_decorator
from optcom.field import Field
from optcom.parameters.fiber.raman_response import RamanResponse


class GNLSE(AbstractNLSE):
    r"""General non linear Schrodinger equations.

    Represent the different effects in the GNLSE.

    Notes
    -----

    .. math:: \begin{split}
                \frac{\partial A_j}{\partial t}
                &= i\gamma \Big(1+\frac{i}{\omega_0}
                \frac{\partial}{\partial t}\Big)
                \Bigg[\bigg[(1-f_R)|A_j(z,t)|^2
                + f_R\int_{-\infty}^{+\infty}h_R(s)|A_j(z,t-s)|^2 ds
                \bigg] \\
                &\quad + \sum_{k\neq j} \bigg[\sigma (1-f_R)|A_k(z,t)|^2
                + \eta f_R\int_{-\infty}^{+\infty}h_R(s)|A_k(z,t-s)|^2
                ds\bigg] \Bigg] A_j
              \end{split}

    """

    def __init__(self, alpha: Optional[Union[List[float], Callable]] = None,
                 alpha_order: int = 0,
                 beta: Optional[Union[List[float], Callable]] = None,
                 beta_order: int = 2,
                 gamma: Optional[Union[float, Callable]] = None,
                 sigma: float = cst.XPM_COEFF, eta: float = cst.XNL_COEFF,
                 h_R: Optional[Union[float, Callable]] = None,
                 f_R: float = cst.F_R, core_radius: float = cst.CORE_RADIUS,
                 clad_radius: float = cst.CLAD_RADIUS,
                 n_core: Optional[Union[float, Callable]] = None,
                 n_clad: Optional[Union[float, Callable]] = None,
                 NA: Optional[Union[float, Callable]] = None,
                 v_nbr: Optional[Union[float, Callable]] = None,
                 eff_area: Optional[Union[float, Callable]] = None,
                 nl_index: Optional[Union[float, Callable]] = None,
                 medium_core: str = cst.FIBER_MEDIUM_CORE,
                 medium_clad: str = cst.FIBER_MEDIUM_CLAD,
                 temperature: float = cst.TEMPERATURE, ATT: bool = True,
                 DISP: bool = True, SPM: bool = True, XPM: bool = False,
                 FWM: bool = False, XNL: bool = False, NOISE: bool = True,
                 UNI_OMEGA: bool = True, STEP_UPDATE: bool = False,
                 INTRA_COMP_DELAY: bool = True, INTRA_PORT_DELAY: bool = True,
                 INTER_PORT_DELAY: bool = False) -> None:
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
        core_radius :
            The radius of the core. :math:`[\mu m]`
        clad_radius :
            The radius of the cladding. :math:`[\mu m]`
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
        XNL :
            If True, trigger cross-non linear effects.
        NOISE :
            If True, trigger the noise calculation.
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
        super().__init__(alpha, alpha_order, beta, beta_order, gamma,
                         core_radius, clad_radius, n_core, n_clad, NA, v_nbr,
                         eff_area, nl_index, medium_core, medium_clad,
                         temperature, ATT, DISP, NOISE, UNI_OMEGA, STEP_UPDATE,
                         INTRA_COMP_DELAY, INTRA_PORT_DELAY, INTER_PORT_DELAY)
        self._f_R = f_R
        self._kerr: Kerr = Kerr(SPM, XPM, FWM, sigma)
        self._add_non_lin_effect(self._kerr, 0, 0)
        if (h_R is None):
            h_R = RamanResponse()
        self._raman: Raman = Raman(h_R, True, XNL, eta)
        self._add_non_lin_effect(self._raman, 0, 0)
        self._self_steep: SelfSteepening = SelfSteepening()
        self._add_non_lin_effect(self._self_steep, 0, 0)
    # ==================================================================
    def open(self, domain: Domain, *fields: List[Field]) -> None:
        super().open(domain, *fields)
        self._raman.set()
        self._self_steep.set(self._center_omega)
    # ==================================================================
    @sync_waves_decorator
    def op_non_lin(self, waves: np.ndarray, id: int,
                   corr_wave: Optional[np.ndarray] = None
                   ) -> np.ndarray:
        r"""Represent the non linear effects of the NLSE.

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

        .. math:: \hat{N} = \mathcal{F}^{-1}\bigg\{i \gamma
                  \Big(1+\frac{\omega}{\omega_0}\Big)
                  \mathcal{F}\Big\{ (1-f_R) |A|^2
                  + f_R \mathcal{F}^{-1}\big\{\mathcal{F}\{h_R\}
                  \mathcal{F}\{|A|^2\}\big\}\Big\}\bigg\}

        """
        kerr = self._kerr.op(waves, id, corr_wave)
        raman = self._raman.op(waves, id, corr_wave)
        corr_wave = (self._gamma[id]
                     * (((1.0-self._f_R)*kerr) + (self._f_R*raman)))

        return self._self_steep.op(waves, id, corr_wave)
    # ==================================================================
    def term_non_lin(self, waves: np.ndarray, id: int, z: float,
                     corr_wave: Optional[np.ndarray] = None
                     ) -> np.ndarray:
        # No need to sync waves as only waves[id] is used in self._steep
        corr_wave = self.op_non_lin(waves, id, corr_wave)

        return self._self_steep.term(waves, id, corr_wave)
    # ==================================================================
    def term_rk4ip_non_lin(self, waves: np.ndarray, id: int, z: float,
                           corr_wave: Optional[np.ndarray] = None
                           ) -> np.ndarray:
        # No need to sync waves as only waves[id] is used in self._steep
        corr_wave = self.op_non_lin(waves, id, corr_wave)

        return self._self_steep.term_rk4ip(waves, id, corr_wave)
