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

from typing import Callable, List, Optional, overload, Union

import numpy as np

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.domain import Domain
from optcom.effects.abstract_effect import AbstractEffect
from optcom.effects.abstract_effect_taylor import AbstractEffectTaylor
from optcom.effects.attenuation import Attenuation
from optcom.effects.dispersion import Dispersion
from optcom.equations.abstract_field_equation import AbstractFieldEquation
from optcom.equations.abstract_field_equation import sync_waves_decorator
from optcom.field import Field
from optcom.parameters.dispersion.chromatic_disp import ChromaticDisp
from optcom.parameters.fiber.effective_area import EffectiveArea
from optcom.parameters.fiber.nl_coefficient import NLCoefficient
from optcom.parameters.fiber.numerical_aperture import NumericalAperture
from optcom.parameters.fiber.v_number import VNumber
from optcom.parameters.refractive_index.nl_index import NLIndex
from optcom.parameters.refractive_index.sellmeier import Sellmeier
from optcom.utils.callable_container import CallableContainer


class AbstractNLSE(AbstractFieldEquation):
    r"""Non linear Schrodinger equations.

    Represent the different effects in the NLSE as well as different
    types of NLSEs.

    """

    def __init__(self, alpha: Optional[Union[List[float], Callable]],
                 alpha_order: int,
                 beta: Optional[Union[List[float], Callable]],
                 beta_order: int, gamma: Optional[Union[float, Callable]],
                 core_radius: float, clad_radius: float,
                 n_core: Optional[Union[float, Callable]],
                 n_clad: Optional[Union[float, Callable]],
                 NA: Optional[Union[float, Callable]],
                 v_nbr: Optional[Union[float, Callable]],
                 eff_area: Optional[Union[float, Callable]],
                 nl_index: Optional[Union[float, Callable]],
                 medium_core: str, medium_clad: str, temperature: float,
                 ATT: bool, DISP: bool, NOISE: bool, UNI_OMEGA: bool,
                 STEP_UPDATE: bool, INTRA_COMP_DELAY: bool,
                 INTRA_PORT_DELAY: bool, INTER_PORT_DELAY: bool) -> None:
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
            The temperature of the medium. :math:`[K]`
        ATT :
            If True, trigger the attenuation.
        DISP :
            If True, trigger the dispersion.
        NOISE :
            If True, trigger the noise calculation.
        UNI_OMEGA :
            If True, consider only the center omega for computation.
            Otherwise, considered omega discretization.
        STEP_UPDATE :
            If True, update component parameters at each spatial
            space step by calling the _update_variables method.
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
        super().__init__(nbr_eqs=1, SHARE_WAVES=False, prop_dir=[True],
                         NOISE=NOISE, STEP_UPDATE=STEP_UPDATE,
                         INTRA_COMP_DELAY=INTRA_COMP_DELAY,
                         INTRA_PORT_DELAY=INTRA_PORT_DELAY,
                         INTER_PORT_DELAY=INTER_PORT_DELAY)
        # media --------------------------------------------------------
        self._medium_core: str = medium_core
        self._medium_clad: str = medium_clad
        # Refractive indices and numerical aperture --------------------
        self._n_core: Union[float, Callable]
        self._n_clad: Union[float, Callable]
        self._NA: Union[float, Callable]
        fct_calc_n_core = NumericalAperture.calc_n_core
        fct_calc_n_clad = NumericalAperture.calc_n_core
        if (n_core is not None and n_clad is not None and NA is not None):
            self._n_core = n_core
            self._n_clad = n_clad
            self._NA = NA
        elif (n_core is not None and n_clad is not None):
            self._n_core = n_core
            self._n_clad = n_clad
            self._NA = NumericalAperture(self._n_core, self._n_clad)
        elif (n_core is not None and NA is not None):
            self._n_core = n_core
            self._NA = NA
            self._n_clad = CallableContainer(fct_calc_n_clad,
                                             [self._NA, self._n_core])
        elif (n_clad is not None and NA is not None):
            self._n_clad = n_clad
            self._NA = NA
            self._n_core = CallableContainer(fct_calc_n_core,
                                             [self._NA, self._n_clad])
        elif (n_core is not None):
            self._n_core = n_core
            self._n_clad = Sellmeier(medium_clad, cst.FIBER_CLAD_DOPANT,
                                     cst.CLAD_DOPANT_CONCENT)
            self._NA = NumericalAperture(self._n_core, self._n_clad)
        elif (n_clad is not None):
            self._n_clad = n_clad
            self._n_core = Sellmeier(medium_core, cst.FIBER_CORE_DOPANT,
                                     cst.CORE_DOPANT_CONCENT)
            self._NA = NumericalAperture(self._n_core, self._n_clad)
        elif (NA is not None):
            self._NA = NA
            self._n_core = Sellmeier(medium_core, cst.FIBER_CORE_DOPANT,
                                     cst.CORE_DOPANT_CONCENT)
            self._n_clad = CallableContainer(fct_calc_n_clad,
                                             [self._NA, self._n_core])
        else:   # all None
            self._n_core = Sellmeier(medium_core, cst.FIBER_CORE_DOPANT,
                                     cst.CORE_DOPANT_CONCENT)
            self._n_clad = Sellmeier(medium_clad, cst.FIBER_CLAD_DOPANT,
                                     cst.CLAD_DOPANT_CONCENT)
            self._NA = NumericalAperture(self._n_core, self._n_clad)
        # V number -----------------------------------------------------
        self._v_nbr: Union[float, Callable]
        if (v_nbr is not None):
            self._v_nbr = v_nbr
        else:
            self._v_nbr = VNumber(self._NA, core_radius)
        # Effective Area -----------------------------------------------
        self._eff_area: Union[float, Callable]
        if (eff_area is not None):
            self._eff_area = eff_area
        else:
            self._eff_area = EffectiveArea(self._v_nbr, core_radius)
        # Non-linear index ---------------------------------------------
        self._nl_index: Union[float, Callable]
        if (nl_index is not None):
            self._nl_index = nl_index
        else:
            self._nl_index = NLIndex(medium_core)
        # Non-linear coefficient ---------------------------------------
        self._predict_gamma: Optional[Callable] = None
        self._gamma: np.ndarray
        if (gamma is not None):
            if (callable(gamma)):
                self._predict_gamma = gamma
            else:
                self._gamma = np.asarray(util.make_list(gamma))
        else:
            self._predict_gamma = NLCoefficient(self._nl_index, self._eff_area)
        # Effects ------------------------------------------------------
        self._att: Optional[AbstractEffectTaylor] = None
        self._disp: Optional[AbstractEffectTaylor] = None
        if (ATT):
            if (alpha is not None):
                self._att = Attenuation(alpha, alpha_order,
                                        UNI_OMEGA=UNI_OMEGA)
                self._add_lin_effect(self._att, 0, 0)
                self._add_delay_effect(self._att)
            else:
                util.warning_terminal("Currently no method to calculate "
                    "attenuation as a function of the medium. Must provide "
                    "attenuation coefficient alpha, otherwise attenuation "
                    "effect will be ignored.")
        # Even if DISP==False, need _beta in CNLSE
        self._beta: Union[List[float], Callable]
        if (beta is None):
            self._beta = ChromaticDisp(self._n_core)
        else:
            self._beta = beta
        if (DISP):
            self._disp = Dispersion(self._beta, beta_order, start_taylor=2,
                                    UNI_OMEGA=UNI_OMEGA)
            self._add_lin_effect(self._disp, 0, 0)
            self._add_delay_effect(self._disp)
    # ==================================================================
    @property
    def gamma(self):

        return self._gamma
    # ==================================================================
    def _update_variables(self):
        if (self._att is not None):
            self._att.set(self._center_omega, self._abs_omega)
        if (self._disp is not None):
            self._disp.set(self._center_omega, self._abs_omega)
        if (self._predict_gamma is not None):
            self._gamma = self._predict_gamma(self._center_omega)
        else:
            self._gamma = util.modify_length_ndarray(self._gamma,
                                                     len(self._center_omega))
    # ==================================================================
    def open(self, domain: Domain, *fields: List[Field]) -> None:
        super().open(domain, *fields)
        # Noise --------------------------------------------------------
        if (self._att is not None):
            att_coeff: np.ndarray = self._att.coeff(self._noise_omega, 0)
            noise_fct = lambda noises, z, h, ind: (-1*noises[ind]*att_coeff)
            for i in range(self._nbr_eqs):
                self._add_noise_effect(noise_fct, i)
