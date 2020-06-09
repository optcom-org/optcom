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
from typing import List

import numpy as np

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.utils.id_tracker import IdTracker


# Exceptions
class SynchroniserError(Exception):
    pass

class InitializationError(SynchroniserError):
    pass


class Synchroniser(object):

    def __init__(self, INTRA_COMP_DELAY: bool, INTRA_PORT_DELAY: bool,
                 INTER_PORT_DELAY: bool, REP_FREQ_DELAY: bool = False) -> None:
        """
        Parameters
        ----------
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
        REP_FREQ_DELAY :
            If True, take into account the pulses at regular interval
            specified by the repetition frequency.

        """
        self.INTRA_COMP_DELAY: bool = INTRA_COMP_DELAY
        self.INTRA_PORT_DELAY: bool = INTRA_PORT_DELAY
        self.INTER_PORT_DELAY: bool = INTER_PORT_DELAY
        # REP_FREQ_DELAY currently unused, future feature
        self.REP_FREQ_DELAY: bool = REP_FREQ_DELAY
        self._valid_init: bool = False  # Make sure valid init is done pre-sync
        self._init_delays: np.ndarray = np.array([])
        self._rep_freqs: List[float] = []
        self._id_tracker: IdTracker
        self._dtime: float = 0.0
    # ==================================================================
    def initialize(self, init_delays: np.ndarray, rep_freqs: List[float],
                   id_tracker: IdTracker, dtime: float) -> None:
        self._init_delays = init_delays
        self._rep_freqs = rep_freqs
        self._id_tracker = id_tracker
        self._dtime = dtime
        self._valid_init = True
    # ==================================================================
    def sync(self, waves: np.ndarray, delays: np.ndarray, id: int = 0
             ) -> np.ndarray:
        """Adjust the wave envelopes in waves depending on their delays
        compare to the waves at the reference position id.

        Parameters
        ----------
        waves :
            The envelope value of each wave at each time step.
        delays :
            The current delays of each wave.
        id :
            The reference position of the wave from which to
            synchronise the other waves.

        Returns
        -------
        :
            Array containing the waves synchronised with the wave at
            reference position id.

        """
        if (not self._valid_init):

            raise InitializationError("Please first initialize the "
                "synchroniser by calling the method initialize().")
        # Calculate relative delays ------------------------------------
        sync_waves = np.zeros_like(waves, dtype=cst.NPFT)
        abs_time = np.zeros(len(waves))
        if (self.INTRA_COMP_DELAY):
            abs_time += delays[:,-1]
        if (self.INTER_PORT_DELAY):
            # Here delay rel. to all waves, so can add to absolute time
            abs_time += self._init_delays
        if (self.INTRA_PORT_DELAY and not self.INTER_PORT_DELAY):
            # Need to add only relative delay between channels from
            # the same port.
            # interfer with other waves in other fields / eqs / ports
            start, end = self._id_tracker.wave_ids_in_eq_id(
                self._id_tracker.eq_id_of_wave_id(id))
            inter_channel_delays = copy.copy(self._init_delays[start:end+1])
            for i in range(len(inter_channel_delays)):
                inter_channel_delays[i] -= self._init_delays[id]
            abs_time[start:end+1] += inter_channel_delays
        # Shifting arrays to synchronise -------------------------------
        rel_delay: int = 0  # relative delay [number of indices] to id
        # N.B.: value is + if further than waves[id] and - if before waves[id]
        for i in range(len(waves)):
            if (i != id):
                if (self._id_tracker.are_wave_ids_co_prop(i, id)):
                    rel_delay = int((abs_time[i] - abs_time[id]) / self._dtime)
                    # np.roll(a, b): shift right if b is (+), left if b is (-)
                    sync_waves[i] = np.roll(waves[i], rel_delay)
                    if (rel_delay < 0): # shifted left
                        sync_waves[i][rel_delay:] = 0.0
                    if (rel_delay > 0): # shifted right
                        sync_waves[i][:rel_delay] = 0.0
                else: # For now the counter prop waves are ignored
                    sync_waves[i] = 0.0
            else:
                sync_waves[i] = waves[i]

        return sync_waves
