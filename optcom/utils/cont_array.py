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

from typing import List, overload, Tuple

import numpy as np


class ContArray():
    """Container for array with identical step size."""

    def __init__(self, nbr_ports: int) -> None:

        self._nbr_ports = nbr_ports
        self._arrays: List[Tuple[float, float, int]] = []
    # ==================================================================
    @overload
    def __getitem__(self, key: int) -> np.ndarray: ...
    # ==================================================================
    @overload
    def __getitem__(self, key: slice) -> np.ndarray: ...
    # ==================================================================
    def __getitem__(self, key):

        if (isinstance(key, slice)):
            res = []
            step: int = 0
            step = 1 if key.step is None else key.step
            for i in range(key.start, key.stop, step):
                res.append(self.create_array(self._arrays[i]))

            return res
        else:

            return self.create_array(self._arays[key])
    # ==================================================================
    def __setitem__(self, key: int, array: np.ndarray) -> None:

        self._arrays[key] = (array[0], array[-1], len(array))
    # ==================================================================
    def append(self, array: np.ndarray) -> None:
        self._arrays.append((array[0], array[-1], len(array)))
    # ==================================================================
    def create_array(self, coord: Tuple[float, float, int])-> np.ndarray:

        return np.linspace(coord[0], coord[1], coord[2], True, False)
