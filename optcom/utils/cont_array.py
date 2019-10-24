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

from typing import List, Tuple

import numpy as np
from nptyping import Array


class ContArray():
    """Container for array with identical step size."""

    def __init__(self, nbr_ports: int) -> None:

        self._nbr_ports = nbr_ports
        self._arrays: List[Tuple[float, float, int]] = []
    # ==================================================================
    @overload
    def __getitem__(self, key: int) -> Array[float]: ...
    # ==================================================================
    @overload
    def __getitem__(self, key: slice) -> Array[float]: ...
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
    def __setitem__(self, key: int, array: Array[float]) -> None:

        self._arrays[key] = (array[0], array[-1], len(array))
    # ==================================================================
    def append(self, array: Array[float]) -> None:
        self._arrays.append((array[0], array[-1], len(array)))
    # ==================================================================
    def create_array(self, coord: Tuple[float, float, int])-> Array[float]:

        return np.linspace(coord[0], coord[1], coord[2], True, False)
