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
from typing import Callable, List, Optional, overload, Union, Tuple

import numpy as np

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.domain import Domain
from optcom.equations.re_fiber_2levels import REFiber2Levels


FLOAT_COEFF_TYPE_OPTIONAL = List[Union[float, Callable, None]]


class REFiberYb(REFiber2Levels):

    def __init__(self, N_T: float = cst.N_T, tau: float = cst.TAU_META_YB,
                 doped_area: Optional[float] = None,
                 n_core: FLOAT_COEFF_TYPE_OPTIONAL = [None],
                 n_clad: FLOAT_COEFF_TYPE_OPTIONAL = [None],
                 NA: FLOAT_COEFF_TYPE_OPTIONAL = [None],
                 v_nbr: FLOAT_COEFF_TYPE_OPTIONAL = [None],
                 eff_area: FLOAT_COEFF_TYPE_OPTIONAL = [None],
                 overlap: FLOAT_COEFF_TYPE_OPTIONAL = [None],
                 sigma_a: FLOAT_COEFF_TYPE_OPTIONAL = [None],
                 sigma_e: FLOAT_COEFF_TYPE_OPTIONAL = [None],
                 core_radius: float = cst.CORE_RADIUS,
                 clad_radius: float = cst.CLAD_RADIUS,
                 medium_core: str = cst.FIBER_MEDIUM_CORE,
                 medium_clad: str = cst.FIBER_MEDIUM_CLAD,
                 temperature: float = cst.TEMPERATURE, RESO_INDEX: bool = True,
                 CORE_PUMPED: bool = True, CLAD_PUMPED: bool = False,
                 NOISE: bool = True, UNI_OMEGA: List[bool] = [True, True],
                 STEP_UPDATE: bool = False) -> None:
        r"""
        Parameters
        ----------
        N_T :
            The total doping concentration. :math:`[nm^{-3}]`
        tau :
            The lifetime of the metastable level. :math:`[\mu s]`
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
        UNI_OMEGA :
            If True, consider only the center omega for computation.
            Otherwise, considered omega discretization.  The first
            element is related to the seed and the second to the pump.
        STEP_UPDATE :
            If True, update fiber parameters at each spatial sub-step.

        """
        dopant = 'Yb'
        super().__init__(N_T, tau, doped_area, n_core, n_clad, NA, v_nbr,
                         eff_area, overlap, sigma_a, sigma_e, core_radius,
                         clad_radius, medium_core, medium_clad, dopant,
                         temperature, RESO_INDEX, CORE_PUMPED, CLAD_PUMPED,
                         NOISE, UNI_OMEGA, STEP_UPDATE)
