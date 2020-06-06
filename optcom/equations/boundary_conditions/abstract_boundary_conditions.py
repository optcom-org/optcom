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

import numpy as np

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.domain import Domain


def apply_cond_decorator(apply_cond):
    def method_wrapper(self, *args, **kwargs):
        self.inc_iter()

        return apply_cond(self, *args, **kwargs)

    return method_wrapper


class AbstractBoundaryConditions(metaclass=ABCMeta):
    """Define a common framework for boundary conditions.
    """

    def __init__(self):
        self.__crt_iter: int = -1
    # ==================================================================
    @abstractmethod
    def get_input(self, waves: np.ndarray, noises: np.ndarray,
                  upper_bound: bool): pass
    # ==================================================================
    @abstractmethod
    def apply_cond(self, waves: np.ndarray, noises: np.ndarray,
                   upper_bound: bool): pass
    # ==================================================================
    @abstractmethod
    def get_output(self, waves_f: np.ndarray, waves_b: np.ndarray,
                   noises_f: np.ndarray, noises_b: np.ndarray): pass
    # ==================================================================
    def initialize(self, domain: Domain) -> None: ...
    # ==================================================================
    def get_iter(self):

        return self.__crt_iter
    # ==================================================================
    def inc_iter(self):
        self.__crt_iter += 1
    # ==================================================================
    def dec_iter(self):
        self.__crt_iter -= 1
    # ==================================================================
    def reset(self):
        self.__crt_iter = -1
