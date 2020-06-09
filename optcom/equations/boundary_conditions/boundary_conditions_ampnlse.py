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
from typing import Optional, Tuple

import numpy as np

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.equations.abstract_ampnlse import AbstractAmpNLSE
from optcom.equations.boundary_conditions.abstract_boundary_conditions import\
    AbstractBoundaryConditions, apply_cond_decorator
from optcom.field import Field


class BoundaryConditionsAmpNLSE(AbstractBoundaryConditions):
    """Boundary conditions for AmpNLSE equations.
    """

    def __init__(self, AmpNLSE: AbstractAmpNLSE,
                 REFL_SEED: bool = True, REFL_PUMP: bool = True,
                 PRE_PUMP_PROP: bool = False) -> None:
        """
        Parameters
        ----------
        AmpNLSE :
            The equation to which the boundary conditions are linked.
        REFL_SEED : bool
            If True, take into account the reflected seed waves for
            computation.
        REFL_PUMP : bool
            If True, take into account the reflected pump waves for
            computation.
        PRE_PUMP_PROP :
            If True, propagate only the pump the first iteration and
            then add the seed at the second iteration.  Otherwise,
            consider both pump and seed simultaneously.  Results will be
            consistent but setting this flag to True can have a
            significant impact on the convergence speed.  Without
            knowledge of the algorithm, it is adviced to let it to
            False.

        """
        super().__init__()
        self._eq: AbstractAmpNLSE = AmpNLSE
        self.REFL_SEED: bool = REFL_SEED
        self.REFL_PUMP: bool = REFL_PUMP
        self._waves_f_ref: np.ndarray = np.array([])
        self._waves_b_ref: np.ndarray = np.array([])
        self._noises_f_ref: np.ndarray = np.array([])
        self._noises_b_ref: np.ndarray = np.array([])
        self._id_tracker = self._eq._id_tracker
        self.PRE_PUMP_PROP: bool = PRE_PUMP_PROP
    # ==================================================================
    def reset(self):
        super().reset()
        self._waves_f_ref = np.array([])
        self._waves_b_ref = np.array([])
        self._noises_f_ref = np.array([])
        self._noises_b_ref = np.array([])
    # ==================================================================
    def _calc_power_for_ratio(self, array: np.ndarray):

        return np.sum(Field.temporal_power(array), axis=1).reshape((-1,1))
    # ==================================================================
    def get_input(self, waves: np.ndarray, noises: np.ndarray,
                  upper_bound: bool) -> Tuple[np.ndarray, np.ndarray]:
        """Save the initial waves and noises and return the first
        guess.  The first guess is the pump waves and noises only, the
        seeds are considered later by the boundary conditions.

        Parameters
        ----------
        waves :
            The initial values of the wave envelopes.
        noises :
            The initial values of the noise powers.
        upper_bound :
            If True, considered first guess at the end z=L.  Otherwise,
            considered at the end z=0.

        Returns
        -------
        :
            The first guesses of wave envelopes and noise powers.

        """
        self.reset()
        # Waves --------------------------------------------------------
        waves_init: np.ndarray = waves[:self._id_tracker.nbr_waves]
        waves_refl: np.ndarray = waves[self._id_tracker.nbr_waves:]
        waves_s_f: np.ndarray
        waves_s_b: np.ndarray
        waves_p_f: np.ndarray
        waves_p_b: np.ndarray
        waves_s_f_r: np.ndarray
        waves_s_b_r: np.ndarray
        waves_p_f_r: np.ndarray
        waves_p_b_r: np.ndarray
        waves_s_f = self._id_tracker.waves_in_eq_id(waves_init, 0)
        waves_s_b = self._id_tracker.waves_in_eq_id(waves_init, 1)
        waves_p_f = self._id_tracker.waves_in_eq_id(waves_init, 2)
        waves_p_b = self._id_tracker.waves_in_eq_id(waves_init, 3)
        waves_s_f_r = self._id_tracker.waves_in_eq_id(waves_refl, 0)
        waves_s_b_r = self._id_tracker.waves_in_eq_id(waves_refl, 1)
        waves_p_f_r = self._id_tracker.waves_in_eq_id(waves_refl, 2)
        waves_p_b_r = self._id_tracker.waves_in_eq_id(waves_refl, 3)
        # Forward : z=0 -> z=L
        self._waves_f_ref = np.vstack((waves_s_f,       # z=0 seed forward
                                       waves_s_b_r,     # z=L seed reflection
                                       waves_p_f,       # z=0 pump forward
                                       waves_p_b_r))    # z=L pump reflection
        # Backward : z=L -> z=0
        self._waves_b_ref = np.vstack((waves_s_f_r,     # z=0 seed reflection
                                       waves_s_b,       # z=L seed backward
                                       waves_p_f_r,     # z=0 pump reflection
                                       waves_p_b))      # z=L pump backward
        # Init power of waves
        self._waves_s_f_e = self._calc_power_for_ratio(waves_s_f)
        self._waves_s_b_e = self._calc_power_for_ratio(waves_s_b)
        self._waves_p_f_e = self._calc_power_for_ratio(waves_p_f)
        self._waves_p_b_e = self._calc_power_for_ratio(waves_p_b)
        # Noises -------------------------------------------------------
        noises_init: np.ndarray = noises[:self._id_tracker.nbr_fields]
        noises_refl: np.ndarray = noises[self._id_tracker.nbr_fields:]
        noises_s_f: np.ndarray
        noises_s_b: np.ndarray
        noises_p_f: np.ndarray
        noises_p_b: np.ndarray
        noises_s_f_r: np.ndarray
        noises_s_b_r: np.ndarray
        noises_p_f_r: np.ndarray
        noises_p_b_r: np.ndarray
        noises_s_f = self._id_tracker.fields_in_eq_id(noises_init, 0)
        noises_s_b = self._id_tracker.fields_in_eq_id(noises_init, 1)
        noises_p_f = self._id_tracker.fields_in_eq_id(noises_init, 2)
        noises_p_b = self._id_tracker.fields_in_eq_id(noises_init, 3)
        noises_s_f_r = self._id_tracker.fields_in_eq_id(noises_refl, 0)
        noises_s_b_r = self._id_tracker.fields_in_eq_id(noises_refl, 1)
        noises_p_f_r = self._id_tracker.fields_in_eq_id(noises_refl, 2)
        noises_p_b_r = self._id_tracker.fields_in_eq_id(noises_refl, 3)
        # Forward : z=0 -> z=L
        self._noises_f_ref = np.vstack((noises_s_f,     # z=0 seed forward
                                        noises_s_b_r,   # z=L seed reflection
                                        noises_p_f,     # z=0 pump forward
                                        noises_p_b_r))  # z=L pump reflection
        # Backward : z=L -> z=0
        self._noises_b_ref = np.vstack((noises_s_f_r,   # z=0 seed reflection
                                        noises_s_b,     # z=L seed backward
                                        noises_p_f_r,   # z=0 pump reflection
                                        noises_p_b))    # z=L pump backward
        if (upper_bound):

            return self.apply_cond(self._waves_b_ref, self._noises_b_ref,
                                   upper_bound)
        else:

            return self.apply_cond(self._waves_f_ref, self._noises_f_ref,
                                   upper_bound)
    # ==================================================================
    @apply_cond_decorator
    def apply_cond(self, waves: np.ndarray, noises: np.ndarray,
                   upper_bound: bool) -> Tuple[np.ndarray, np.ndarray]:
        r"""Return the boundary conditions of the fiber amplifier at
        both ends.

        Parameters
        ----------
        waves :
            The current values of the wave envelopes at the specified
            end.
        noises :
            The current values of the noise powers at the specified end.
        upper_bound :
            If True, considered the boundary conditions at the end z=L.
            Otherwise, considered at the end z=0.

        Returns
        -------
        :
            The new guesses of wave envelopes and noise powers.

        Notes
        -----
        Boundary conditions at z=0:

        .. math:: \begin{aligned}
                    & A_s^+(0) = \Bigg(1 + \sqrt{R_0 \frac{P_{avg}
                        \big(A_{s}^{r+}(0)\big)}{P_{avg}
                        \big(A_{s,ref}^+\big)}}\Bigg)A_{s,ref}^+\\
                    & A_{s}^{r-}(0) = A_{s,ref}^{r-}
                        + \sqrt{R_0}A_{s}^{-}(0)\\
                    & A_p^+(0) = \Bigg(1 + \sqrt{R_0\frac{P_{avg}
                        \big(A_{p}^{r+}(0)\big)}{P_{avg}
                        \big(A_{p,ref}^+\big)}}\Bigg)A_{p,ref}^+\\
                    & A_{p}^{r-}(0) = A_{p,ref}^{r-}
                        + \sqrt{R_0}A_{p}^{-}(0)
                   \end{aligned}

        Boundary conditions at z=L:

        .. math:: \begin{aligned}
                    & A_s^-(L) = \Bigg(1 + \sqrt{R_L \frac{P_{avg}
                        \big(A_s^{r-}(L)\big)}{P_{avg}
                        \big(A_{s,ref}^-\big)}}\Bigg)A_{s,ref}^-\\
                    & A_s^{r+}(L) = A_{s,ref}^{r+}
                        + \sqrt{R_L}A_{s}^{+}(L)\\
                    & A_p^-(L) = \Bigg(1 + \sqrt{R_L\frac{P_{avg}
                        \big(A_p^{r-}(L)\big)}{P_{avg}
                        \big(A_{p,ref}^-\big)}}\Bigg)A_{p,ref}^-\\
                    & A_p^{r+}(L) = A_{p,ref}^{r+}
                        + \sqrt{R_L}A_{p}^{+}(L)\\
                   \end{aligned}

        Where the subscript + denotes the pulses coming from z=0 to z=L,
        the subscript - denotes the pulses coming from z=L to z=0,
        the subscript s denontes the seed pulses, the subscript p
        denotes the pump pulses, the subscript r denotes the
        reflection and the subscript ref denotes the initial reference
        pulses.  The same conditions are applied for the noises but
        the noise powers are considered, not the envelopes.
        As the ratio in the square root is between average powers,
        ratio between temporal powers can be considered as the average
        power is the temporal power multiplied by the elementary time
        step and the repetition frequency and that those metrics are
        identical for a pulse and its reflection.

        """
        waves_: np.ndarray = np.zeros_like(waves, dtype=cst.NPFT)
        noises_ : np.ndarray = np.zeros_like(noises, dtype=float)
        R_0_waves: np.ndarray = self._eq.R_0_waves
        R_L_waves: np.ndarray = self._eq.R_L_waves
        R_0_noises: np.ndarray = self._eq.R_0_noises
        R_L_noises: np.ndarray = self._eq.R_L_noises
        if (self.PRE_PUMP_PROP and self.get_iter() < 2):
            consider_seed = False # propagate only pumps
        else:
            consider_seed = True
        if (upper_bound):
            waves_, noises_ = self._apply_cond_at_end(waves, noises,
                                                      R_L_waves, R_L_noises,
                                                      consider_seed, True)
        else:
            waves_, noises_ = self._apply_cond_at_start(waves, noises,
                                                        R_0_waves, R_0_noises,
                                                        consider_seed, True)

        return waves_, noises_
    # ==================================================================
    def _apply_cond_at_start(self, waves: np.ndarray,
                             noises: np.ndarray, R_0_waves: np.ndarray,
                             R_0_noises: np.ndarray,
                             consider_seed: bool = True,
                             consider_pump: bool = True
                             ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply the boundary conditions at z=0."""
        waves_: np.ndarray = np.zeros_like(waves, dtype=cst.NPFT)
        noises_ : np.ndarray = np.zeros_like(noises, dtype=float)
        if (consider_seed):
            # z=0 seed forward ------------------------------------------
            start, end = self._id_tracker.wave_ids_in_eq_id(0)
            inc_energy = self._calc_power_for_ratio(waves[start:end+1])
            ratio = np.zeros_like(self._waves_s_f_e)
            ratio = np.divide(inc_energy, self._waves_s_f_e, out=ratio,
                              where=self._waves_s_f_e!=0)
            waves_[start:end+1] = (self._waves_f_ref[start:end+1]
                                   + (np.sqrt(ratio*R_0_waves[start:end+1])
                                      * self._waves_f_ref[start:end+1]))
            start, end = self._id_tracker.field_ids_in_eq_id(0)
            noises_[start:end+1] = (self._noises_f_ref[start:end+1]
                                    + (R_0_noises * noises_[start:end+1]))
            # z=L seed reflection --------------------------------------
            if (self.REFL_SEED):
                start, end = self._id_tracker.wave_ids_in_eq_id(1)
                waves_[start:end+1] = (self._waves_f_ref[start:end+1]
                                       + (np.sqrt(R_0_waves[start:end+1])
                                          * waves[start:end+1]))
                start, end = self._id_tracker.field_ids_in_eq_id(1)
                noises_[start:end+1] = (self._noises_f_ref[start:end+1]
                                        + (R_0_noises * noises_[start:end+1]))
        if (consider_pump):
            # z=0 pump forward -----------------------------------------
            start, end = self._id_tracker.wave_ids_in_eq_id(2)
            inc_energy = self._calc_power_for_ratio(waves[start:end+1])
            ratio = np.zeros_like(self._waves_p_f_e)
            ratio = np.divide(inc_energy, self._waves_p_f_e, out=ratio,
                              where=self._waves_p_f_e!=0)
            waves_[start:end+1] = (self._waves_f_ref[start:end+1]
                                   + (np.sqrt(ratio*R_0_waves[start:end+1])
                                      * self._waves_f_ref[start:end+1]))
            start, end = self._id_tracker.field_ids_in_eq_id(2)
            noises_[start:end+1] = (self._noises_f_ref[start:end+1]
                                    + (R_0_noises * noises_[start:end+1]))
            # z=L pump reflection --------------------------------------
            if (self.REFL_PUMP):
                start, end = self._id_tracker.wave_ids_in_eq_id(3)
                waves_[start:end+1] = (self._waves_f_ref[start:end+1]
                                       + (np.sqrt(R_0_waves[start:end+1])
                                          * waves[start:end+1]))
                start, end = self._id_tracker.field_ids_in_eq_id(3)
                noises_[start:end+1] = (self._noises_f_ref[start:end+1]
                                        + (R_0_noises * noises_[start:end+1]))

        return waves_, noises_
    # ==================================================================
    def _apply_cond_at_end(self, waves: np.ndarray,
                           noises: np.ndarray, R_L_waves: np.ndarray,
                           R_L_noises: np.ndarray,
                           consider_seed: bool = True,
                           consider_pump: bool = True
                           ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply the boundary conditions at z=L."""
        waves_: np.ndarray = np.zeros_like(waves, dtype=cst.NPFT)
        noises_ : np.ndarray = np.zeros_like(noises, dtype=float)
        if (consider_seed):
            # z=0 seed reflection --------------------------------------
            if (self.REFL_SEED):
                start, end = self._id_tracker.wave_ids_in_eq_id(0)
                waves_[start:end+1] = (self._waves_b_ref[start:end+1]
                                       + (np.sqrt(R_L_waves[start:end+1])
                                          * waves[start:end+1]))
                start, end = self._id_tracker.field_ids_in_eq_id(0)
                noises_[start:end+1] = (self._noises_b_ref[start:end+1]
                                        + (R_L_noises * noises_[start:end+1]))
            # z=L seed backward ----------------------------------------
            start, end = self._id_tracker.wave_ids_in_eq_id(1)
            inc_energy = self._calc_power_for_ratio(waves[start:end+1])
            ratio = np.zeros_like(self._waves_s_b_e)
            ratio = np.divide(inc_energy, self._waves_s_b_e, out=ratio,
                              where=self._waves_s_b_e!=0)
            waves_[start:end+1] = (self._waves_b_ref[start:end+1]
                                   + (np.sqrt(ratio*R_L_waves[start:end+1])
                                      * self._waves_b_ref[start:end+1]))
            start, end = self._id_tracker.field_ids_in_eq_id(1)
            noises_[start:end+1] = (self._noises_b_ref[start:end+1]
                                    + (R_L_noises * noises_[start:end+1]))
        if (consider_pump):
            # z=0 pump reflection --------------------------------------
            if (self.REFL_PUMP):
                start, end = self._id_tracker.wave_ids_in_eq_id(2)
                waves_[start:end+1] = (self._waves_b_ref[start:end+1]
                                       + (np.sqrt(R_L_waves[start:end+1])
                                          * waves[start:end+1]))
                start, end = self._id_tracker.field_ids_in_eq_id(2)
                noises_[start:end+1] = (self._noises_b_ref[start:end+1]
                                       + (R_L_noises * noises_[start:end+1]))
            # z=L pump backward ----------------------------------------
            start, end = self._id_tracker.wave_ids_in_eq_id(3)
            inc_energy = self._calc_power_for_ratio(waves[start:end+1])
            ratio = np.zeros_like(self._waves_p_b_e)
            ratio = np.divide(inc_energy, self._waves_p_b_e, out=ratio,
                              where=self._waves_p_b_e!=0)
            waves_[start:end+1] = (self._waves_b_ref[start:end+1]
                                   + (np.sqrt(ratio*R_L_waves[start:end+1])
                                      * self._waves_b_ref[start:end+1]))
            start, end = self._id_tracker.field_ids_in_eq_id(3)
            noises_[start:end+1] = (self._noises_b_ref[start:end+1]
                                    + (R_L_noises * noises_[start:end+1]))

        return waves_, noises_
    # ==================================================================
    def get_output(self, waves_f: np.ndarray, waves_b: np.ndarray,
                   noises_f: np.ndarray, noises_b: np.ndarray,
                   ) -> Tuple[np.ndarray, np.ndarray]:
        """Return the ouput of the waves and noises depending on the
        initial waves and noises saved in get_input().

        Parameters
        ----------
        waves_f :
            The final values of the wave envelopes at z=0.
        waves_b :
            The final values of the wave envelopes at z=L.
        noises_f :
            The final values of the noise powers at z=0.
        noises_b :
            The final values of the noise powers at z=L.

        Returns
        -------
        :
            The output of wave envelopes and noise powers.

        """
        # Waves --------------------------------------------------------
        for i in range(len(waves_f)):
            waves_f[i] *= np.sqrt(1.0-self._eq._R_L_waves[i])
        for i in range(len(waves_b)):
            waves_b[i] *= np.sqrt(1.0-self._eq._R_0_waves[i])
        waves_s_f: np.ndarray = self._id_tracker.waves_in_eq_id(waves_f, 0)
        waves_s_b: np.ndarray = self._id_tracker.waves_in_eq_id(waves_b, 1)
        waves_p_f: np.ndarray = self._id_tracker.waves_in_eq_id(waves_f, 2)
        waves_p_b: np.ndarray = self._id_tracker.waves_in_eq_id(waves_b, 3)
        waves_s_f_r: np.ndarray = self._id_tracker.waves_in_eq_id(waves_b, 0)
        waves_s_b_r: np.ndarray = self._id_tracker.waves_in_eq_id(waves_f, 1)
        waves_p_f_r: np.ndarray = self._id_tracker.waves_in_eq_id(waves_b, 2)
        waves_p_b_r: np.ndarray = self._id_tracker.waves_in_eq_id(waves_f, 3)
        waves_: np.ndarray
        waves_ = np.concatenate((waves_s_f,       # z=0 seed forward
                                 waves_s_b,       # z=L seed backward
                                 waves_p_f,       # z=0 pump forward
                                 waves_p_b,       # z=L pump backward
                                 waves_s_f_r,     # z=0 seed reflected
                                 waves_s_b_r,     # z=L seed reflected
                                 waves_p_f_r,     # z=0 pump reflected
                                 waves_p_b_r))    # z=L pump reflected
        # Noises ------------------------------------------------------
        noises_f *= np.sqrt(1.0-self._eq._R_L_noises)
        noises_b *= np.sqrt(1.0-self._eq._R_0_noises)
        noises_s_f: np.ndarray = self._id_tracker.fields_in_eq_id(noises_f, 0)
        noises_s_b: np.ndarray = self._id_tracker.fields_in_eq_id(noises_b, 1)
        noises_p_f: np.ndarray = self._id_tracker.fields_in_eq_id(noises_f, 2)
        noises_p_b: np.ndarray = self._id_tracker.fields_in_eq_id(noises_b, 3)
        noises_s_f_r: np.ndarray = self._id_tracker.fields_in_eq_id(noises_b,0)
        noises_s_b_r: np.ndarray = self._id_tracker.fields_in_eq_id(noises_f,1)
        noises_p_f_r: np.ndarray = self._id_tracker.fields_in_eq_id(noises_b,2)
        noises_p_b_r: np.ndarray = self._id_tracker.fields_in_eq_id(noises_f,3)
        noises_: np.ndarray
        noises_ = np.concatenate((noises_s_f,       # z=0 seed forward
                                  noises_s_b,       # z=L seed backward
                                  noises_p_f,       # z=0 pump forward
                                  noises_p_b,       # z=L pump backward
                                  noises_s_f_r,     # z=0 seed reflected
                                  noises_s_b_r,     # z=L seed reflected
                                  noises_p_f_r,     # z=0 pump reflected
                                  noises_p_b_r))    # z=L pump reflected

        return waves_, noises_
