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
from optcom.effects.absorption import Absorption
from optcom.effects.emission import Emission
from optcom.effects.relaxation import Relaxation
from optcom.equations.abstract_re_fiber import AbstractREFiber
from optcom.field import Field
from optcom.utils.fft import FFT


FLOAT_COEFF_TYPE_OPTIONAL = List[Union[float, Callable, None]]


class REFiber2Levels(AbstractREFiber):

    def __init__(self, N_T: float = cst.N_T, tau: float = cst.TAU_META,
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
                 dopant: str = cst.DEF_FIBER_DOPANT,
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
        UNI_OMEGA :
            If True, consider only the center omega for computation.
            Otherwise, considered omega discretization.  The first
            element is related to the seed and the second to the pump.
        STEP_UPDATE :
            If True, update fiber parameters at each spatial sub-step.

        """
        super().__init__(N_T, doped_area, n_core, n_clad, NA, v_nbr, eff_area,
                         overlap, sigma_a, sigma_e, core_radius, clad_radius,
                         medium_core, medium_clad, dopant, temperature,
                         RESO_INDEX, CORE_PUMPED, CLAD_PUMPED, NOISE,
                         STEP_UPDATE)
        self._nbr_levels = 2
        # Rate equations definition ------------------------------------
        # Seed ---------------------------------------------------------
        self._absorption_s = Absorption(self._sigma_a[0], self._overlap[0],
                                        self._doped_area, UNI_OMEGA[0])
        self._emission_s = Emission(self._sigma_e[0], self._overlap[0],
                                    self._doped_area, UNI_OMEGA[0])
        self._add_ind_effect(self._emission_s, 0, 1, 1.0)
        self._add_ind_effect(self._emission_s, 1, 1, -1.0)
        self._add_ind_effect(self._absorption_s, 0, 0, -1.0)
        self._add_ind_effect(self._absorption_s, 1, 0, 1.0)
        # Pump ---------------------------------------------------------
        self._absorption_p = Absorption(self._sigma_a[1], self._overlap[1],
                                        self._doped_area, UNI_OMEGA[1])
        self._emission_p = Emission(self._sigma_e[1], self._overlap[1],
                                    self._doped_area, UNI_OMEGA[1])
        self._add_ind_effect(self._absorption_p, 0, 0, -1.0)
        self._add_ind_effect(self._absorption_p, 1, 0, 1.0)
        self._add_ind_effect(self._emission_p, 0, 1, 1.0)
        self._add_ind_effect(self._emission_p, 1, 1, -1.0)
        # Noise --------------------------------------------------------
        UNI_OMEGA_NOISE = False   # need to consider each omega
        self._absorption_n = Absorption(self._sigma_a[0], self._overlap[0],
                                        self._doped_area, UNI_OMEGA_NOISE)
        self._emission_n = Emission(self._sigma_e[0], self._overlap[0],
                                    self._doped_area, UNI_OMEGA_NOISE)
        self._add_ind_effect(self._absorption_n, 1, 0, -1.0)
        self._add_ind_effect(self._absorption_n, 0, 0, 1.0)
        self._add_ind_effect(self._emission_n, 0, 1, 1.0)
        self._add_ind_effect(self._emission_n, 1, 1, -1.0)
        # Relaxation ---------------------------------------------------
        self._relaxation = Relaxation(tau) # us -> ps
        self._add_ind_effect(self._relaxation, 0, 1, 1.0)
        self._add_ind_effect(self._relaxation, 1, 1, -1.0)
        # Signal frequency step size -----------------------------------
        # if time dependency would be : (to do)
        # self._prop: np.ndarray[float, steps, samples, nbr_levels]
        # self._pop: np.ndarray = np.zeros((0, 0, self._nbr_levels),
        #                                   dtype=np.float64)
        # if no time dependency
        # self._prop: np.ndarray[float, steps, nbr_levels]
        self._pop: np.ndarray = np.array([])
        self._noises_back_up: np.ndarray = np.array([])
    # ==================================================================
    @overload
    def __call__(self, waves: np.ndarray, z: float, h: float
                 )-> np.ndarray: ...
    # ------------------------------------------------------------------
    @overload
    def __call__(self, waves: np.ndarray, z: float, h: float, ind: int
                 ) -> np.ndarray: ...
    # ------------------------------------------------------------------
    def __call__(self, *args):
        r"""Calculate the upper level population density.

        Parameters
        ----------
        wave :
            The wave vector.
        z :
            The spatial variable.
        h :
            The spatial step.

        Returns
        -------
        :
            Value of the upper level population density.
            :math:`[nm^{-3}]`

        Notes
        -----

        .. math::
            N_1(z) =  \frac{\frac{2\pi N_{T}}{h A_c}\Big[
            \sum^K_{k=1} \frac{\Gamma_{s}(\omega_k)}{\omega_k}
            \sigma_{a}(\omega_k) P_{s,k}^{\pm}(z) + \sum^L_{l=1}
            \frac{\Gamma_{p}(\omega_l)}{\omega_l} \sigma_{a}(\omega_l)
            P_{p,l}^{\pm}(z)\Big]}{\frac{2\pi}{h A_c}\bigg[\sum^K_{k=1}
            \frac{\Gamma_{s}(\omega_k)}{\omega_k}\big[ \sigma_{a}
            (\omega_k) + \sigma_{e}(\omega_k)\big] P_{s,k}^{\pm}(z)
            + \sum^M_{m=1}\frac{\Gamma_{s}(\omega_m)}{\omega_m}
            \sigma_e(\omega_m) P_{ase,m}^{\pm}(z) + \sum^L_{l=1}
            \frac{\Gamma_{p}(\omega_l)}{\omega_l}\big[
            \sigma_{a}(\omega_l) + \sigma_{e}(\omega_l)\big]
            P_{p,l}^{\pm}(z)\bigg] + \gamma_{10}}

        """
        if (len(args) == 3):
            waves, z, h = args
            packet_s = []
            packet_p = []
            fst_ind = 0
            while (len(waves) > fst_ind):   # add counterprop waves
                packet_s.append(np.vstack((
                    self.id_tracker.waves_in_eq_id(waves[fst_ind:], 0),
                    self.id_tracker.waves_in_eq_id(waves[fst_ind:], 1))))
                packet_p.append(np.vstack((
                    self.id_tracker.waves_in_eq_id(waves[fst_ind:], 2),
                    self.id_tracker.waves_in_eq_id(waves[fst_ind:], 3))))
                fst_ind += len(packet_s[-1]) + len(packet_p[-1])
            # Emission and absorption of the seed and pump
            num = 0.0
            den = 0.0
            for waves_s in packet_s:
                for i in range(len(waves_s)):
                    num += self._absorption_s.term(waves_s, i)
                    den += self._emission_s.term(waves_s, i)
            for waves_p in packet_p:
                for i in range(len(waves_p)):
                    num += self._absorption_p.term(waves_p, i)
                    den += self._emission_p.term(waves_p, i)
            # Amplified spontaneous emission
            if (self._NOISE):
                num += np.real(np.sum(self._absorption_n.op(np.array([]), 0)
                                      * self._back_up_noise))
                den += np.real(np.sum(self._emission_n.op(np.array([]), 0)
                                      * self._back_up_noise))
            # Bringing all together
            den += num
            den += self._relaxation.term(waves, 0)
            N_1 = (num / den) * self._N_T
            if (N_1 < 0): # If pump power not enough to trigger inversion of pop.
                N_1 = 0.0
            self._pop[-1] = np.array([self._N_T - N_1, N_1])

            return waves
        else: # Can implemented time dependent equations here (len(args) == 4)

            raise NotImplementedError()
    # ==================================================================
    def set(self, waves: np.ndarray, noises: np.ndarray, z: float, h: float
            ) -> None:
        super().set(waves, noises, z, h)
        if (self._NOISE):
            self._back_up_noise = np.sum(noises, axis=0)
        if (not self._pop.size):
            self._pop = np.zeros((0, self._nbr_levels), dtype=np.float64)
        to_add: np.ndarray = np.zeros((1, self._nbr_levels), dtype=np.float64)
        self._pop = np.vstack((self._pop, to_add))
    # ==================================================================
    def _update_variables(self):
        start_s = self.id_tracker.wave_ids_in_eq_id(0)[0]
        end_s = self.id_tracker.wave_ids_in_eq_id(1)[1]
        start_p = self.id_tracker.wave_ids_in_eq_id(2)[0]
        end_p = self.id_tracker.wave_ids_in_eq_id(3)[1]
        center_omega_s = self._center_omega[start_s:end_s+1]
        abs_omega_s = self._abs_omega[start_s:end_s+1]
        center_omega_p = self._center_omega[start_p:end_p+1]
        abs_omega_p = self._abs_omega[start_p:end_p+1]
        self._absorption_s.set(center_omega_s, abs_omega_s)
        self._emission_s.set(center_omega_s, abs_omega_s)
        self._absorption_p.set(center_omega_p, abs_omega_p)
        self._emission_p.set(center_omega_p, abs_omega_p)
        self._absorption_n.set(center_omega_p, np.array([self._noise_omega]))
        self._emission_n.set(center_omega_p, np.array([self._noise_omega]))
        self._absorption_s.rep_freq = self._rep_freq[start_s:end_s+1]
        self._absorption_p.rep_freq = self._rep_freq[start_p:end_p+1]
        self._emission_s.rep_freq = self._rep_freq[start_s:end_s+1]
        self._emission_p.rep_freq = self._rep_freq[start_p:end_p+1]
    # ==================================================================
    def pre_pass(self) -> None:
        self._pop = np.array([])
        self._noises_back_up = np.array([])
        super().pre_pass()
