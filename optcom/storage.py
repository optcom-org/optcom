# This# This file is part of Optcom.
#
# Optcom is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Optcom is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Optcom.  If not, see <https://www.gnu.org/licenses/>.

""".. moduleauthor:: Sacha Medaer"""

from __future__ import annotations

from typing import List

import numpy as np
from nptyping import Array

import optcom.utils.constants as cst
import optcom.utils.utilities as util
#from optcom.utils.cont_array import ContArray


class Storage(object):
    """Store computation step parameters and waves.

    """

    def __init__(self):

        self._channels: Array[cst.NPFT, self._nbr_channels, ...,
                              self._samples] = np.array([], dtype=cst.NPFT)
        self._space: Array[float] = np.array([])
        self._time: Array[float, self._nbr_channels, ...,
                           self._samples] = np.array([])
        self._samples: int = 0
        self._nbr_channels: int = 0
    # ==================================================================
    def __len__(self) -> int:

        return self._nbr_channels
    # ==================================================================
    @property
    def nbr_channels(self) -> int:

        return self._nbr_channels
    # ------------------------------------------------------------------
    @nbr_channels.setter
    def nbr_channels(self, nbr_channels: int) -> None:
        self._nbr_channels = nbr_channels
    # ==================================================================
    @property
    def samples(self) -> int:

        return self._samples
    # ------------------------------------------------------------------
    @samples.setter
    def samples(self, samples: int) -> None:
        self._samples = samples
    # ==================================================================
    @property
    def channels(self) -> Array[cst.NPFT]:

        return self._channels
    # ------------------------------------------------------------------
    @channels.setter
    def channels(self, channels: Array[cst.NPFT]) -> None:
        self._channels = channels
    # ==================================================================
    @property
    def space(self) -> Array[float]:

        return np.ones((self._nbr_channels, self._space.shape[0]))*self._space
    # ------------------------------------------------------------------
    @space.setter
    def space(self, space: Array[float]) -> None:
        self._space = space[0]
    # ==================================================================
    @property
    def time(self) -> Array[float]:

        return self._time
    # ------------------------------------------------------------------
    @time.setter
    def time(self, time: Array[float]) -> None:
        self._time = time
    # ==================================================================
    def append(self, channels, space, time) -> None:
        self._channels = channels
        self._nbr_channels = channels.shape[0]
        self._space = space
        self._time = time
        self._samples = time.shape[-1]
    # ==================================================================
    def extend(self, storage: Storage) -> None:
        check_channels = (self._nbr_channels == storage.nbr_channels)
        check_samples = (self._samples == storage.samples)
        if (check_samples):
            channels = storage.channels
            time = storage.time
            if (not check_channels):
                diff = storage.nbr_channels-self._nbr_channels
                if (diff > 0):
                    to_add = np.zeros(((diff,) + self.channels[0].shape))
                    self.channels = np.vstack((self.channels, to_add))
                    to_add = np.zeros(((diff,) + self.time[0].shape))
                    self.time = np.vstack((self.time, to_add))
                    self._nbr_channels = storage.nbr_channels
                else:
                    to_add = np.zeros(((abs(diff),) + channels[0].shape))
                    channels = np.vstack((channels, to_add))
                    to_add = np.zeros(((abs(diff),) + time[0].shape))
                    time = np.vstack((time, to_add))
            self.channels = np.hstack((self.channels, channels))
            self.time = np.hstack((self.time, time))
            space_to_add = storage.space[0] + np.sum(self._space)
            self._space = np.hstack((self._space, space_to_add))
        else:
            util.warning_terminal("Storages extension aborted: same number of "
                "samples is needed for extension.")

        return self
