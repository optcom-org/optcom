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

import numpy as np

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.domain import Domain
from optcom.effects.abstract_effect import AbstractEffect
from optcom.effects.kerr import Kerr
from optcom.effects.raman_approx import RamanApprox
from optcom.effects.self_steepening_approx import SelfSteepeningApprox
from optcom.equations.abstract_nlse import AbstractNLSE
from optcom.field import Field


class ANLSE(AbstractNLSE):
    r"""Approximated non linear Schrodinger equations.

    Represent the different effects in the approximated NLSE.

    Notes
    -----

    Equations :

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

    Implementation:

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
                 alpha_order: int = 0,
                 beta: Optional[Union[List[float], Callable]] = None,
                 beta_order: int = 2,
                 gamma: Optional[Union[float, Callable]] = None,
                 sigma: float = cst.XPM_COEFF, eta: float = cst.XNL_COEFF,
                 T_R: float = cst.RAMAN_COEFF,
                 core_radius: float = cst.CORE_RADIUS,
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
                 FWM: bool = False, SS: bool = False, RS: bool = False,
                 XNL: bool = False, NOISE: bool = True,
                 approx_type: int = cst.DEFAULT_APPROX_TYPE,
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
        T_R :
            The raman coefficient. :math:`[]`
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
        SS :
            If True, trigger the self-steepening.
        RS :
            If True, trigger the Raman scattering.
        XNL :
            If True, trigger cross-non linear effects.
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
        super().__init__(alpha, alpha_order, beta, beta_order, gamma,
                         core_radius, clad_radius, n_core, n_clad, NA, v_nbr,
                         eff_area, nl_index, medium_core, medium_clad,
                         temperature, ATT, DISP, NOISE, UNI_OMEGA, STEP_UPDATE,
                         INTRA_COMP_DELAY, INTRA_PORT_DELAY, INTER_PORT_DELAY)
        self._kerr: Optional[Kerr] = None
        if (XPM or SPM or FWM):
            self._kerr = Kerr(SPM, XPM, FWM, sigma)
            self._add_non_lin_effect(self._kerr, 0, 0)
        self._self_steep: Optional[SelfSteepeningApprox] = None
        if (SS):
            self._self_steep = SelfSteepeningApprox(True, XNL, eta,
                                                    approx_type)
            self._add_non_lin_effect(self._self_steep, 0, 0)
        self._raman: Optional[RamanApprox] = None
        if (RS):
            self._raman = RamanApprox(T_R, True, XNL, eta, approx_type)
            self._add_non_lin_effect(self._raman, 0, 0)
    # ==================================================================
    def open(self, domain: Domain, *fields: List[Field]) -> None:
        super().open(domain, *fields)
        if (self._self_steep is not None):
            self._self_steep.set(self._center_omega)
    # ==================================================================
    def op_non_lin(self, waves: np.ndarray, id: int,
                   corr_wave: Optional[np.ndarray] = None
                   ) -> np.ndarray:
        r"""Represent the non linear effects of the approximated NLSE.
        """

        return self._gamma[id] * super().op_non_lin(waves, id, corr_wave)
    # ==================================================================
    def term_non_lin(self, waves: np.ndarray, id: int, z: float,
                   corr_wave: Optional[np.ndarray] = None
                   ) -> np.ndarray:
        r"""Represent the non linear effects of the approximated NLSE.
        """

        return self._gamma[id] * super().term_non_lin(waves, id, z, corr_wave)
