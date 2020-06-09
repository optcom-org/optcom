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
