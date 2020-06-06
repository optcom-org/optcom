# This file is part of Optcom.
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

from abc import ABCMeta, abstractmethod
from typing import overload

import numpy as np

import optcom.utils.utilities as util


class AbstractParameter(metaclass=ABCMeta):

    def __init__(self):

        return None
    # ==================================================================
    @overload
    @abstractmethod
    def __call__(self, physic_variable: float) -> float: ...
    # ------------------------------------------------------------------
    @overload
    @abstractmethod
    def __call__(self, physic_variable: np.ndarray) -> np.ndarray: ...
    # ------------------------------------------------------------------
    @abstractmethod
    def __call__(self, physic_variable): pass
