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

import copy
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

        # self._channels: Array[cst.NPFT, nbr_channels, steps, samples]
        self._channels: np.ndarray = np.array([], dtype=cst.NPFT)
        # self._center_omega: Array[float, nbr_channels]
        self._center_omega: np.ndarray = np.array([])
        # self._rep_freq: Array[float, nbr_channels]
        self._rep_freq: np.ndarray = np.array([])
        # self._time: Array[float, nbr_channels, steps, samples]
        self._time: np.ndarray = np.array([])
        # self._noises: Array[float, nbr_channels, steps, noise_samples]
        self._noises: np.ndarray = np.array([])
        # self._space: Array[float, steps]
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

        return self._channels.shape[2]
    # ==================================================================
    @property
    def steps(self) -> int:
        # Should be equal to self._channels[1]

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
        self._space = space
    # ==================================================================
    @property
    def time(self) -> np.ndarray:

        return self._time
    # ------------------------------------------------------------------
    @time.setter
    def time(self, time: np.ndarray) -> None:
        self._time = time
    # ==================================================================
    def get_copy(self, start_channel: int = 0, stop_channel: int = -1,
                 start_step: int = 0, stop_step: int = -1) -> Storage:
        new_storage = copy.deepcopy(self)
        indices_to_delete = [i for i in range(0, start_channel)]
        if (stop_channel != -1):
            indices_to_delete += [i for i in range(stop_channel,
                                                   self.nbr_channels)]
        new_storage.delete_channel(indices_to_delete)
        indices_to_delete = [i for i in range(0, start_step)]
        if (start_step):    # start new space array at zero
            new_storage.space -= self.space[start_step]
        if (stop_step != -1):
            indices_to_delete += [i for i in range(stop_step, self.steps)]
        new_storage.delete_space_step(indices_to_delete)

        return new_storage
    # ==================================================================
    def delete_channel(self, *indices: int) -> None:
        self._channels = np.delete(self._channels, list(indices), axis=0)
        self._center_omega = np.delete(self._center_omega, list(indices),
                                       axis=0)
        self._rep_freq = np.delete(self._rep_freq, list(indices), axis=0)
        self._time = np.delete(self._time, list(indices), axis=0)
    # ==================================================================
    def delete_space_step(self, *indices: int) -> None:
        self._channels = np.delete(self._channels, list(indices), axis=1)
        self._time = np.delete(self._time, list(indices), axis=1)
        self._noises = np.delete(self._noises, list(indices), axis=1)
        self._space = np.delete(self._space, list(indices), axis=0)
    # ==================================================================
    def append(self, channels: np.ndarray, noises: np.ndarray,
               space: np.ndarray, time: np.ndarray, center_omega: np.ndarray,
               rep_freq: np.ndarray) -> None:
        if (self._channels.size):
            if (len(self._space) == len(space)
                    and self._channels.shape[2] == channels.shape[2]):
                self._channels = np.vstack((self._channels, channels))
                self._noises = np.vstack((self._noises, noises))
                self._time = np.vstack((self._time, time))
                self._center_omega = np.hstack((self._center_omega,
                                                center_omega))
                self._rep_freq = np.hstack((self._rep_freq, rep_freq))
            else:
                warning_message: str = ("Storages append aborted: same number "
                    "of samples and steps are needed.")
                warnings.warn(warning_message, DimWarning)
        else:
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
                    to_add = np.zeros(((diff,) + self.noises[0].shape))
                    self.noises = np.vstack((self.noises, to_add))
                else:
                    to_add = np.zeros(((abs(diff),) + channels[0].shape))
                    channels = np.vstack((channels, to_add))
                    to_add = np.zeros(((abs(diff),) + time[0].shape))
                    time = np.vstack((time, to_add))
                    to_add = np.zeros(((abs(diff),) + noises[0].shape))
                    noises = np.vstack((noises, to_add))
            self.channels = np.hstack((self.channels, channels))
            self.noises = np.vstack((self.noises, noises))
            self.time = np.hstack((self.time, time))
            space_to_add = storage.space + self._space[-1]
            self._space = np.hstack((self._space, space_to_add))
        else:
            warning_message: str = ("Storages extension aborted: same number "
                "of samples is needed for extension.")
            warnings.warn(warning_message, DimWarning)

        return self
