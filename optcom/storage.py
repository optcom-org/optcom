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
    # ==================================================================
    @property
    def samples(self) -> int:

        return self._samples
    # ==================================================================
    @property
    def channels(self) -> Array[cst.NPFT]:

        return self._channels
    # ==================================================================
    @property
    def space(self) -> Array[float]:

        return np.ones((self._nbr_channels, self._space.shape[0]))*self._space
    # ==================================================================
    @property
    def time(self) -> Array[float]:

        return self._time
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
        if (check_channels and check_samples):
            self._channels = np.hstack((self._channels, storage.channels))
            self._time = np.hstack((self._time, storage.time))
            space_to_add = storage.space + np.sum(self._space)
            self._space = np.hstack((self._space, space_to_add))
        else:
            util.warning_terminal("Storages extension aborted: same number of "
                "samples and same number of channels is needed for extension.")
