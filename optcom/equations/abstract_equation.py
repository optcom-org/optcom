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

from abc import ABCMeta
from typing import List, Optional, overload, Tuple

import numpy as np

import optcom.utils.constants as cst
import optcom.utils.utilities as util


class AbstractEquation(metaclass=ABCMeta):

    def __init__(self) -> None:

        return None
    # ==================================================================
    @overload
    def __call__(self, vectors: np.ndarray, z: float, h: float
                 )-> np.ndarray: ...
    # ------------------------------------------------------------------
    @overload
    def __call__(self, vectors: np.ndarray, z: float, h: float, ind: int
                 ) -> np.ndarray: ...
    # ------------------------------------------------------------------
    def __call__(self, *args):
        """
        Parameters
        ----------
        vectors :
            The value of the unknowns at the current step.
        z :
            The current value of the variable.
        h :
            The step size.
        ind :
            The index of the considered vector.

        """

        raise NotImplementedError()
