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
