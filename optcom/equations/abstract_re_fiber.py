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

import math
import warnings
from typing import Callable, List, Optional, overload, Union, Tuple

import numpy as np
from scipy import interpolate

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.domain import Domain
from optcom.equations.abstract_field_equation import AbstractFieldEquation
from optcom.field import Field
from optcom.parameters.fiber.absorption_section import AbsorptionSection
from optcom.parameters.fiber.effective_area import EffectiveArea
from optcom.parameters.fiber.emission_section import EmissionSection
from optcom.parameters.fiber.numerical_aperture import NumericalAperture
from optcom.parameters.fiber.overlap_factor import OverlapFactor
from optcom.parameters.fiber.v_number import VNumber
from optcom.parameters.refractive_index.resonant_index import ResonantIndex
from optcom.parameters.refractive_index.sellmeier import Sellmeier
from optcom.utils.callable_container import CallableContainer
from optcom.utils.callable_litt_expr import CallableLittExpr


FLOAT_COEFF_TYPE_OPTIONAL = List[Union[float, Callable, None]]


# Exceptions
class AbstractREFiberWarning(UserWarning):
    pass

class PumpSchemeWarning(AbstractREFiberWarning):
    pass


class AbstractREFiber(AbstractFieldEquation):

    def __init__(self, N_T: float, doped_area: Optional[float],
                 n_core: FLOAT_COEFF_TYPE_OPTIONAL,
                 n_clad: FLOAT_COEFF_TYPE_OPTIONAL,
                 NA: FLOAT_COEFF_TYPE_OPTIONAL,
                 v_nbr: FLOAT_COEFF_TYPE_OPTIONAL,
                 eff_area: FLOAT_COEFF_TYPE_OPTIONAL,
                 overlap: FLOAT_COEFF_TYPE_OPTIONAL,
                 sigma_a: FLOAT_COEFF_TYPE_OPTIONAL,
                 sigma_e: FLOAT_COEFF_TYPE_OPTIONAL, core_radius: float,
                 clad_radius: float, medium_core: str, medium_clad: str,
                 dopant: str, temperature: float, RESO_INDEX: bool,
                 CORE_PUMPED: bool, CLAD_PUMPED: bool, NOISE: bool,
                 STEP_UPDATE: bool) -> None:
        r"""
        Parameters
        ----------
        N_T :
            The total doping concentration. :math:`[nm^{-3}]`
        doped_area :
            The doped area. :math:`[\mu m^2]`  If None, will be set to
            the core area.
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
        overlap :
            The overlap factors of the signal and the pump.
            (1<=len(overlap)<=2). [signal, pump]  If a
            callable is provided, the variable must be angular
            frequency. :math:`[ps^{-1}]`
        sigma_a :
            The absorption cross sections of the signal and the pump
            (1<=len(sigma_a)<=2). :math:`[nm^2]` [signal, pump]  If a
            callable is provided, the variable must be angular
            frequency. :math:`[ps^{-1}]`
        sigma_e :
            The emission cross sections of the signal and the pump
            (1<=len(sigma_e)<=2). :math:`[nm^2]` [signal, pump]  If a
            callable is provided, the variable must be angular
            frequency. :math:`[ps^{-1}]`
        core_radius :
            The radius of the core. :math:`[\mu m]`
        clad_radius :
            The radius of the cladding. :math:`[\mu m]`
        medium_core :
            The medium of the core.
        medium_clad :
            The medium of the cladding.
        dopant :
            The doping medium of the active fiber.
        temperature :
            The temperature of the fiber. :math:`[K]`
        RESO_INDEX :
            If True, trigger the resonant refractive index which will
            be added to the core refractive index. To see the effect of
            the resonant index, the flag STEP_UPDATE must be set to True
            in order to update the dispersion coefficient at each space
            step depending on the resonant index at each space step.
        CORE_PUMPED :
            If True, there is dopant in the core.
        CLAD_PUMPED :
            If True, there is dopant in the cladding.
        NOISE :
            If True, trigger the noise calculation.
        STEP_UPDATE :
            If True, update fiber parameters at each spatial sub-step.

        """
        nbr_eqs = 4
        super().__init__(nbr_eqs=nbr_eqs, prop_dir=[True, False, True, False],
                         SHARE_WAVES=True, NOISE=NOISE, STEP_UPDATE=True,
                         INTRA_COMP_DELAY=False, INTRA_PORT_DELAY=False,
                         INTER_PORT_DELAY=False)
        n_core = util.make_list(n_core, 2)
        n_clad = util.make_list(n_clad, 2)
        NA = util.make_list(NA, 2)
        v_nbr = util.make_list(v_nbr, 2)
        eff_area = util.make_list(eff_area, 2)
        overlap = util.make_list(overlap, 2)
        sigma_a = util.make_list(sigma_a, 2)
        sigma_e = util.make_list(sigma_e, 2)
        # Alias --------------------------------------------------------
        CLE = CallableLittExpr
        CC = CallableContainer
        # Doping concentration -----------------------------------------
        self._N_T = N_T
        # Media --------------------------------------------------------
        self._medium_core: str = medium_core
        self._medium_clad: str = medium_clad
        # Refractive indices and numerical aperture --------------------
        self._n_core: List[Union[float, Callable]] = []
        self._n_clad: List[Union[float, Callable]] = []
        self._NA: List[Union[float, Callable]] = []
        self._n_reso: List[ResonantIndex] = []
        fct_calc_n_core = NumericalAperture.calc_n_core
        fct_calc_n_clad = NumericalAperture.calc_n_core
        n_core_pur_: Union[CallableLittExpr, CallableContainer, Sellmeier]
        n_core_: Union[CallableLittExpr, CallableContainer, Sellmeier]
        n_clad_: Union[CallableLittExpr, CallableContainer, NumericalAperture,
                       Sellmeier]
        NA_: Callable
        n_reso_: ResonantIndex
        for i in range(2):
            crt_n_core = n_core[i]
            crt_n_clad = n_clad[i]
            crt_NA = NA[i]
            if (crt_n_core is not None):
                n_core_pur_ = CLE([np.ones_like, crt_n_core], ['*'])
                if (RESO_INDEX):
                    n_reso_ = ResonantIndex(dopant, n_core_pur_, 0.0)
                    n_core_ = CLE([n_core_pur_, n_reso_], ['+'])
                else:
                    n_core_ = n_core_pur_
                if (crt_n_clad is not None and crt_NA is not None):
                    n_clad_ = CLE([np.ones_like, crt_n_clad], ['*'])
                    NA_ = CLE([np.ones_like, crt_NA], ['*'])
                elif (crt_n_clad is not None):
                    n_clad_ = CLE([np.ones_like, crt_n_clad], ['*'])
                    NA_ = NumericalAperture(n_core_, n_clad_)
                elif (crt_NA is not None):
                    NA_ = CLE([np.ones_like, crt_NA], ['*'])
                    n_clad_ = CC(fct_calc_n_clad, [NA_, n_core_])
                else:    # n_clad and NA are None
                    n_clad_ = Sellmeier(self._medium_clad,
                                        cst.FIBER_CLAD_DOPANT,
                                        cst.CLAD_DOPANT_CONCENT)
                    NA_ = NumericalAperture(n_core_, n_clad_)
            else:
                if (crt_n_clad is not None and crt_NA is not None):
                    n_clad_ = CLE([np.ones_like, crt_n_clad], ['*'])
                    NA_ = CLE([np.ones_like, crt_NA], ['*'])
                    n_core_pur_ = CC(fct_calc_n_core, [NA_, n_clad_])
                    if (RESO_INDEX):
                        n_reso_ = ResonantIndex(dopant, n_core_pur_, 0.0)
                        n_core_ = CLE([n_core_pur_, n_reso_], ['+'])
                    else:
                        n_core_ = n_core_pur_
                else:
                    n_core_pur_ = Sellmeier(self._medium_core,
                                            cst.FIBER_CORE_DOPANT,
                                            cst.CORE_DOPANT_CONCENT)
                    if (RESO_INDEX):
                        n_reso_ = ResonantIndex(dopant, n_core_pur_, 0.0)
                        n_core_ = CLE([n_core_pur_, n_reso_], ['+'])
                    else:
                        n_core_ = n_core_pur_
                    if (crt_n_clad is not None):
                        n_clad_ = CLE([np.ones_like, crt_n_clad], ['*'])
                        NA_ = NumericalAperture(n_core_, n_clad_)
                    elif (crt_NA is not None):
                        NA_ = CLE([np.ones_like, crt_NA], ['*'])
                        n_clad_ = CC(fct_calc_n_clad, [NA_, n_core_])
                    else:     # all None
                        n_clad_ = Sellmeier(self._medium_clad,
                                            cst.FIBER_CLAD_DOPANT,
                                            cst.CLAD_DOPANT_CONCENT)
                        NA_ = NumericalAperture(n_core_, n_clad_)
            self._n_core.append(n_core_)
            self._n_clad.append(n_clad_)
            if (RESO_INDEX):
                self._n_reso.append(n_reso_)
            self._NA.append(NA_)
        # V number -----------------------------------------------------
        self._v_nbr: List[Union[float, Callable]] = []
        v_nbr_: Union[CallableLittExpr, VNumber]
        for i in range(2):
            crt_v_nbr = v_nbr[i]
            if (crt_v_nbr is None):
                v_nbr_ = VNumber(self._NA[i], core_radius)
            else:
                v_nbr_ = CLE([np.ones_like, crt_v_nbr], ['*'])
            self._v_nbr.append(v_nbr_)
        # Effective Area -----------------------------------------------
        self._eff_area: List[Union[float, Callable]] = []
        eff_area_: Union[CallableLittExpr, EffectiveArea]
        for i in range(2):
            crt_eff_area = eff_area[i]
            if (crt_eff_area is None):
                eff_area_ = EffectiveArea(self._v_nbr[i], core_radius)
            else:
                eff_area_ = CLE([np.ones_like, crt_eff_area], ['*'])
            self._eff_area.append(eff_area_)
        # Doped area ---------------------------------------------------
        core_area: float = cst.PI*(core_radius**2)
        clad_area: float = cst.PI*(clad_radius**2 - core_radius**2)
        self._doped_area: float
        if (doped_area is None):
            self._doped_area = core_area
        else:
            self._doped_area = doped_area
        # Overlap factor -----------------------------------------------
        if (not CORE_PUMPED and not CLAD_PUMPED):
            warning_message: str = ("CORE_PUMPED and CLAD_PUMPED are False, "
                "the fiber amplifier must be at least pumped in the "
                "core or in the cladding for lasing effect. CORE_PUMPED "
                "has been set to True.")
            warnings.warn(warning_message, PumpSchemeWarning)
            CORE_PUMPED = True
        self._overlap: List[Union[float, Callable]] = []
        overlap_: Union[CallableLittExpr, OverlapFactor]
        clad_pump_overlap: Union[float, CallableLittExpr, OverlapFactor]
        core_pump_overlap: Union[float, CallableLittExpr, OverlapFactor]
        for i in range(2):
            crt_overlap = overlap[i]
            if (crt_overlap is None):
                if (not i):
                    overlap_ = OverlapFactor(self._eff_area[i],
                                             self._doped_area)
                else:
                    clad_pump_overlap = self._doped_area / clad_area
                    core_pump_overlap = OverlapFactor(self._eff_area[i],
                                                      self._doped_area)
                    if (CORE_PUMPED and CLAD_PUMPED):
                        overlap_ = CLE([core_pump_overlap, clad_pump_overlap],
                                       ['+'])
                    elif (CLAD_PUMPED):
                        overlap_ = CLE([np.ones_like, clad_pump_overlap],
                                       ['*'])
                    else:   # CORE_PUMPED (forced beforehand if wasn't)
                        overlap_ = core_pump_overlap
            else:
                overlap_ = CLE([np.ones_like, crt_overlap], ['*'])
            self._overlap.append(overlap_)
        # Cross sections -----------------------------------------------
        self._sigma_a: List[Union[float, Callable]] = []
        self._sigma_e: List[Union[float, Callable]] = []
        sigma_a_: Union[CallableLittExpr, AbsorptionSection]
        sigma_e_: Union[CallableLittExpr, EmissionSection]
        for i in range(2):
            crt_sigma_a = sigma_a[i]
            crt_sigma_e = sigma_e[i]
            if (crt_sigma_a is not None and crt_sigma_e is not None):
                sigma_a_ = CLE([np.ones_like, crt_sigma_a], ['*'])
                sigma_e_ = CLE([np.ones_like, crt_sigma_e], ['*'])
            elif (crt_sigma_a is not None):
                sigma_a_ = CLE([np.ones_like, crt_sigma_a], ['*'])
                sigma_e_ = EmissionSection(dopant, medium_core, temperature,
                                           sigma_a_)
            elif (crt_sigma_e is not None):
                sigma_e_ = CLE([np.ones_like, crt_sigma_e], ['*'])
                sigma_a_ = AbsorptionSection(dopant, medium_core, temperature,
                                             sigma_e_)
            else:   # both None
                sigma_a_ = AbsorptionSection(dopant)
                sigma_e_ = EmissionSection(dopant, medium_core, temperature)
            self._sigma_a.append(sigma_a_)
            self._sigma_e.append(sigma_e_)
        # Population density -------------------------------------------
        self._pop = np.zeros(0, dtype=np.float64)
    # ==================================================================
    @property
    def population_levels(self):

        return self._pop.T  # (nbr_levels, steps)
    # ==================================================================
    @property
    def ground_pop(self):

        return self._pop.T[0]  # (nbr_levels, steps)
    # ==================================================================
    @property
    def meta_pop(self):

        return self._pop.T[1]  # (nbr_levels, steps)
    # ==================================================================
    @property
    def n_core(self) -> List[Union[float, Callable]]:

        return self._n_core
    # ==================================================================
    @property
    def n_clad(self) -> List[Union[float, Callable]]:

        return self._n_clad
    # ==================================================================
    @property
    def n_reso(self) -> List[ResonantIndex]:

        return self._n_reso
    # ==================================================================
    @property
    def NA(self) -> List[Union[float, Callable]]:

        return self._NA
    # ==================================================================
    @property
    def v_nbr(self) -> List[Union[float, Callable]]:

        return self._v_nbr
    # ==================================================================
    @property
    def eff_area(self) -> List[Union[float, Callable]]:

        return self._eff_area
    # ==================================================================
    @property
    def overlap(self) -> List[Union[float, Callable]]:

        return self._overlap
    # ==================================================================
    @property
    def sigma_a(self) -> List[Union[float, Callable]]:

        return self._sigma_a
    # ==================================================================
    @property
    def sigma_e(self) -> List[Union[float, Callable]]:

        return self._sigma_e
    # ==================================================================
    def get_population_levels(self):

        return self.population_levels
