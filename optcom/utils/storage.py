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

from __future__ import annotations

import warnings
from typing import List, Optional, Union

import numpy as np

import optcom.utils.constants as cst
import optcom.utils.utilities as util

# Exceptions
class StorageWarning(UserWarning):
    pass

class DimWarning(StorageWarning):
    pass


class Storage(object):
    """Store computation step parameters and waves.
    """

    def __init__(self):

        # self._channels: Array[cst.NPFT, nbr_channels, ..., samples]
        self._channels: np.ndarray = np.array([], dtype=cst.NPFT)
        # self._center_omega: Array[float, nbr_channels]
        self._center_omega: np.ndarray = np.array([])
        # self._rep_freq: Array[float, nbr_channels]
        self._rep_freq: np.ndarray = np.array([])
        # self._time: Array[float, self.nbr_channels, ..., samples]
        self._time: np.ndarray = np.array([])
        # self._noises: Array[float, steps]
        self._noises: np.ndarray = np.array([])
        # self._noises: Array[float, steps]
        self._space: np.ndarray = np.array([])
    # ==================================================================
    def __len__(self) -> int:

        return self.nbr_channels
    # ==================================================================
    def __getitem__(self, key: Union[int, slice]) -> np.ndarray:

        return self._channels[key]
    # ==================================================================
    @property
    def nbr_channels(self) -> int:

        return self._channels.shape[0]
    # ==================================================================
    @property
    def samples(self) -> int:

        return self._channels.shape[1]
    # ==================================================================
    @property
    def steps(self) -> int:

        return len(self._space)
    # ==================================================================
    @property
    def channels(self) -> np.ndarray:

        return self._channels
    # ------------------------------------------------------------------
    @channels.setter
    def channels(self, channels: np.ndarray) -> None:
        self._channels = channels
    # ==================================================================
    @property
    def center_omega(self) -> np.ndarray:

        return self._center_omega
    # ------------------------------------------------------------------
    @center_omega.setter
    def center_omega(self, center_omega: np.ndarray) -> None:
        self._center_omega = center_omega
    # ==================================================================
    @property
    def rep_freq(self) -> np.ndarray:

        return self._rep_freq
    # ------------------------------------------------------------------
    @rep_freq.setter
    def rep_freq(self, rep_freq: np.ndarray) -> None:
        self._rep_freq = rep_freq
    # ==================================================================
    @property
    def noises(self) -> np.ndarray:

        return self._noises
    # ------------------------------------------------------------------
    @noises.setter
    def noises(self, noises: np.ndarray) -> None:
        self._noises = noises
    # ==================================================================
    @property
    def space(self) -> np.ndarray:

        return self._space
    # ------------------------------------------------------------------
    @space.setter
    def space(self, space: np.ndarray) -> None:
        self._space = space[0]
    # ==================================================================
    @property
    def time(self) -> np.ndarray:

        return self._time
    # ------------------------------------------------------------------
    @time.setter
    def time(self, time: np.ndarray) -> None:
        self._time = time
    # ==================================================================
    def append(self, channels: np.ndarray, noises: np.ndarray,
               space: np.ndarray, time: np.ndarray, center_omega: np.ndarray,
               rep_freq: np.ndarray) -> None:
        self._channels = channels
        self._noises = noises
        self._space = space
        self._time = time
        self._center_omega = center_omega
        self._rep_freq = rep_freq
    # ==================================================================
    def extend(self, storage: Storage) -> Storage:
        """Extend the current storage with the provided storage.

        Parameters
        ----------
        storage : optcom.utils.storage.Storage
            The storage from which to extend.

        """
        check_channels = (self.nbr_channels == storage.nbr_channels)
        check_samples = (self.samples == storage.samples)
        if (check_samples):
            channels = storage.channels
            noises = storage.noises
            time = storage.time
            if (not check_channels):
                diff = storage.nbr_channels-self.nbr_channels
                if (diff > 0):
                    to_add = np.zeros(((diff,) + self.channels[0].shape))
                    self.channels = np.vstack((self.channels, to_add))
                    to_add = np.zeros(((diff,) + self.time[0].shape))
                    self.time = np.vstack((self.time, to_add))
                else:
                    to_add = np.zeros(((abs(diff),) + channels[0].shape))
                    channels = np.vstack((channels, to_add))
                    to_add = np.zeros(((abs(diff),) + time[0].shape))
                    time = np.vstack((time, to_add))
            self.channels = np.hstack((self.channels, channels))
            self.noises = np.vstack((self.noises, noises))
            self.time = np.hstack((self.time, time))
            space_to_add = storage.space[0] + np.sum(self._space)
            self._space = np.hstack((self._space, space_to_add))
        else:
            warning_message: str = ("Storages extension aborted: same number "
                "of samples is needed for extension.")
            warnings.warn(warning_message, DimWarning)

        return self
