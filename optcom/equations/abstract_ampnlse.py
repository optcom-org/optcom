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
import math
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.domain import Domain
from optcom.effects.abstract_effect import AbstractEffect
from optcom.effects.attenuation import Attenuation
from optcom.effects.gain import Gain
from optcom.effects.gain_saturation import GainSaturation
from optcom.effects.kerr import Kerr
from optcom.equations.abstract_re_fiber import AbstractREFiber
from optcom.equations.abstract_field_equation import AbstractFieldEquation
from optcom.equations.abstract_field_equation import sync_waves_decorator
from optcom.equations.ase_noise import ASENoise
from optcom.field import Field
from optcom.parameters.fiber.absorption_section import AbsorptionSection
from optcom.parameters.fiber.energy_saturation import EnergySaturation
from optcom.parameters.fiber.doped_fiber_gain import DopedFiberGain
from optcom.parameters.fiber.emission_section import EmissionSection
from optcom.parameters.fiber.se_power import SEPower
from optcom.parameters.refractive_index.resonant_index import ResonantIndex
from optcom.solvers.ode_solver import ODESolver
from optcom.utils.callable_litt_expr import CallableLittExpr
from optcom.utils.fft import FFT


#
SEED_SPLIT: str = 'seed_split'
PUMP_SPLIT: str = 'pump_split'
ALL_SPLIT: str = 'all_split'
NO_SPLIT: str = 'no_split'
SPLIT_NOISE_OPTIONS: List[str] = [SEED_SPLIT, PUMP_SPLIT, ALL_SPLIT, NO_SPLIT]


# Exceptions
class AbstractAmpNLSEError(Exception):
    pass

class UnknownNoiseSplitOptionError(AbstractAmpNLSEError):
    pass


class AbstractAmpNLSE(AbstractFieldEquation):

    def __init__(self, re_fiber: AbstractREFiber, gain_order: int,
                 en_sat: List[Optional[Union[float, Callable]]],
                 R_0: Union[float, Callable], R_L: Union[float, Callable],
                 temperature: float, GAIN_SAT: bool, NOISE: bool,
                 split_noise_option: str, UNI_OMEGA: List[bool],
                 STEP_UPDATE: bool) -> None:
        r"""
        Parameters
        ----------
        re_fiber : AbstractREFiber
            The rate equations describing the fiber laser dynamics.
        gain_order :
            The order of the gain coefficients to take into account.
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
        temperature :
            The temperature of the medium. :math:`[K]`
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
            If True, update component parameters at each spatial
            space step by calling the _update_variables method.

        """
        # 4 equations, one for each port. (2 signals + 2 pumps)
        super().__init__(nbr_eqs=4, prop_dir=[True, False, True, False],
                         SHARE_WAVES=True, NOISE=NOISE,
                         STEP_UPDATE=STEP_UPDATE,
                         INTRA_COMP_DELAY=False, INTRA_PORT_DELAY=False,
                         INTER_PORT_DELAY=False)
        self._re = re_fiber
        self._UNI_OMEGA = UNI_OMEGA
        self._n_reso: List[ResonantIndex] = self._re.n_reso
        self._R_L: Union[float, Callable] = R_L
        self._R_0: Union[float, Callable] = R_0
        self._eff_area = self._re.eff_area
        self._overlap = self._re.overlap
        self._sigma_a = self._re.sigma_a
        self._sigma_e = self._re.sigma_e
        # Alias --------------------------------------------------------
        CLE = CallableLittExpr
        # Gain ---------------------------------------------------------
        start_taylor_gain = 1 if GAIN_SAT else 0
        self._gain_coeff: List[DopedFiberGain] = []
        self._absorp_coeff: List[DopedFiberGain] = []
        k: int = 0  # counter
        for i in range(self._nbr_eqs):
            k = i//2
            self._gain_coeff.append(DopedFiberGain(self._sigma_e[k],
                                                   self._overlap[k], 0.))
            self._absorp_coeff.append(DopedFiberGain(self._sigma_a[k],
                                                     self._overlap[k], 0.))
        self._gains: List[Gain] = []
        self._absorps: List[Attenuation] = []
        for i in range(self._nbr_eqs):
            k = i//2
            self._gains.append(Gain(self._gain_coeff[i], gain_order,
                                    start_taylor=start_taylor_gain,
                                    UNI_OMEGA=UNI_OMEGA[k]))
            self._add_lin_effect(self._gains[-1], i, i)
            self._absorps.append(Attenuation(self._absorp_coeff[i], gain_order,
                                             start_taylor=start_taylor_gain,
                                             UNI_OMEGA=UNI_OMEGA[k]))
            self._add_lin_effect(self._absorps[-1], i, i)
        # Gain saturation ----------------------------------------------
        en_sat = util.make_list(en_sat, 2)
        self._sat_gains: Optional[List[GainSaturation]] = None
        if (GAIN_SAT):
            self._sat_gains = []
            start_taylor_gain = 1
            en_sat_: Union[float, Callable]
            for i in range(self._nbr_eqs):
                k = i//2
                crt_en_sat = en_sat[k]
                if (crt_en_sat is None):
                    en_sat_ = EnergySaturation(self._eff_area[k],
                                               self._sigma_a[k],
                                               self._sigma_e[k],
                                               self._overlap[k])
                else:
                    en_sat_ = CLE([np.ones_like, crt_en_sat], ['*'])
                coeff = CLE([self._gain_coeff[i], self._absorp_coeff[i]],
                            ['-'])
                self._sat_gains.append(GainSaturation(coeff, en_sat_,
                                                      UNI_OMEGA=UNI_OMEGA[k]))
                self._add_lin_effect(self._sat_gains[-1], i, i)
        # Noise --------------------------------------------------------
        self._split_noise_option = util.check_attr_value(split_noise_option,
                                                         SPLIT_NOISE_OPTIONS,
                                                         SEED_SPLIT)
        # Reflection coeff ---------------------------------------------
        self._R_0_waves: np.ndarray = np.array([])
        self._R_L_waves: np.ndarray = np.array([])
        self._R_0_noises: np.ndarray = np.array([])
        self._R_L_noises: np.ndarray = np.array([])
    # ==================================================================
    @property
    def R_0_waves(self) -> np.ndarray:

        return self._R_0_waves
    # ==================================================================
    @property
    def R_L_waves(self) -> np.ndarray:

        return self._R_L_waves
    # ==================================================================
    @property
    def R_0_noises(self) -> np.ndarray:

        return self._R_0_noises
    # ==================================================================
    @property
    def R_L_noises(self) -> np.ndarray:

        return self._R_L_noises
    # ==================================================================
    def open(self, domain: Domain, *fields: List[Field]) -> None:
        super().open(domain, *fields)
        # Noise --------------------------------------------------------
        # Init splitting ratios depending on requirement
        split_ratios: List[float] = []
        split_ratio: float = 0.
        if (self._split_noise_option == SEED_SPLIT):
            split_ratio = (self._id_tracker.nbr_fields_in_eq(0)
                           + self._id_tracker.nbr_fields_in_eq(1))
            split_ratio = 1./split_ratio if split_ratio else 0.
            split_ratios = [split_ratio, split_ratio, 0., 0.]
        elif (self._split_noise_option == PUMP_SPLIT):
            split_ratio = (self._id_tracker.nbr_fields_in_eq(2)
                           + self._id_tracker.nbr_fields_in_eq(3))
            split_ratio = 1./split_ratio if split_ratio else 0.
            split_ratios = [0., 0., split_ratio, split_ratio]
        elif (self._split_noise_option == ALL_SPLIT):
            split_ratio = self._id_tracker.nbr_fields
            split_ratio = 1./split_ratio if split_ratio else 0.
            split_ratios = [split_ratio for i in range(self._nbr_eqs)]
        elif (self._split_noise_option == NO_SPLIT):
            split_ratios = [1. for i in range(self._nbr_eqs)]
        else:

            raise UnknownNoiseSplitOptionError("Unknown splitting noise "
                "option '{}'.".format(self._split_noise_option))
        # Split Spontaneous Emission power
        ase_noise_: ASENoise
        se_power = SEPower(self._noise_domega)
        for i in range(self._nbr_eqs):
            se_power_ = CallableLittExpr([se_power, split_ratios[i]], ['*'])
            ase_noise_ = ASENoise(se_power_, self._gain_coeff[0],
                                  self._absorp_coeff[0], self._noise_omega)
            self._add_noise_effect(ase_noise_, i)
        # Reflection coeffs --------------------------------------------
        if (callable(self._R_0)):
            if (self._UNI_OMEGA):
                self._R_0_waves = self._R_0(self._center_omega).reshape((-1,1))
            else:
                self._R_0_coeff = np.zeros_like(self._abs_omega)
                for i in range(len(self._center_omega)):
                    self._R_0_coeff[i] = self._R_0(self._abs_omega[i])
            self._R_0_noises = self._R_0(self._noise_omega)
        else:
            if (self._UNI_OMEGA):
                self._R_0_waves = np.ones((len(self._center_omega), 1))
            else:
                self._R_0_waves = np.ones_like(self._abs_omega)
            self._R_0_waves *= self._R_0
            self._R_0_noises = self._R_0 * np.ones_like(self._noise_omega)
        if (callable(self._R_L)):
            if (self._UNI_OMEGA):
                self._R_L_waves = self._R_L(self._center_omega).reshape((-1,1))
            else:
                self._R_L_coeff = np.zeros_like(self._abs_omega)
                for i in range(len(self._center_omega)):
                    self._R_L_coeff[i] = self._R_L(self._abs_omega[i])
            self._R_L_noises = self._R_L(self._noise_omega)
        else:
            if (self._UNI_OMEGA):
                self._R_L_waves = np.ones((len(self._center_omega), 1))
            else:
                self._R_L_waves = np.ones_like(self._abs_omega)
            self._R_L_waves *= self._R_L
            self._R_L_noises = self._R_L * np.ones_like(self._noise_omega)
    # ==================================================================
    def set(self, waves: np.ndarray, noises: np.ndarray, z: float, h: float
            ) -> None:
        for n_reso in self._n_reso:
            n_reso.N = self._re.meta_pop[-1]
        for i in range(self._nbr_eqs):
            # Check if channels propagating in eq. i
            if (self.id_tracker.nbr_waves_in_eq(i)):
                self._gain_coeff[i].N = self._re.meta_pop[-1]
                self._absorp_coeff[i].N = self._re.ground_pop[-1]
                center_omega = self.id_tracker.waves_in_eq_id(
                    self._center_omega, i)
                abs_omega = self.id_tracker.waves_in_eq_id(self._abs_omega, i)
                self._gains[i].set(center_omega, abs_omega)
                self._absorps[i].set(center_omega, abs_omega)
                if (self._sat_gains is not None):
                    self._sat_gains[i].set(center_omega, abs_omega)
        super().set(waves, noises, z, h)
    # ==================================================================
    def close(self, domain: Domain, *fields: List[Field]) -> None:
        super().close(domain, *fields)
        # Reset parameters for future computation
        for n_reso in self._n_reso:
            n_reso.N = 0.0
        for gain in self._gain_coeff:
            gain.N = 0.0
        for absorp in self._absorp_coeff:
            absorp.N = 0.0
