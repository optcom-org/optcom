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

import copy
from abc import ABCMeta
from typing import Callable, List, Optional

import numpy as np

import optcom.utils.constants as cst
import optcom.utils.utilities as util

# Typing variables
#SOLVER_CALLABLE_TYPE = \
#    Union[Callable[[np.ndarray, float, float], np.ndarray],
#          Callable[[np.ndarray, float, float, int], np.ndarray]]

SOLVER_CALLABLE_TYPE = Callable
METHOD_SOLVER_CALLABLE_TYPE = Callable[[SOLVER_CALLABLE_TYPE, np.ndarray,
                                        float, float], np.ndarray]


class AbstractSolver(metaclass=ABCMeta):

    _default_method = ''    # Must be overwritten in child

    def __init__(self, f: SOLVER_CALLABLE_TYPE, method: Optional[str]) -> None:
        """
        Parameters
        ----------
        f :
            The function to compute.
        method :
            The computation method. Call the __call__ function of the
            equation if None.

        """
        self.name: str
        self._method: METHOD_SOLVER_CALLABLE_TYPE
        if (method is None):    # analytical solution, no need numerical
            self.name = 'f_call'
            self._method = getattr(self, self.name)
        elif (hasattr(self, method.lower())):   # Force to be static method
            self.name = method.lower()
            self._method = getattr(self, self.name)
        else:
            util.warning_terminal("The solver method '{}' does not exist, "
                                  "default solver '{}' is set."
                                  .format(method,
                                          self.__class__._default_method))
            self.name = self.__class__._default_method
            self._method = getattr(self, self.__class__._default_method)
        self.f: SOLVER_CALLABLE_TYPE = f
    # ==================================================================
    def __call__(self, vectors: np.ndarray, z: float, h: float) -> np.ndarray:
        """
        Parameters
        ----------
        vectors :
            The value of the variables at the considered time/
            space step.
        h :
            The step size.
        z :
            The variable value. (time, space, ...)

        """

        return self._method(self.f, vectors, z, h)
    # ==================================================================
    @staticmethod
    def f_call(f: SOLVER_CALLABLE_TYPE, vectors: np.ndarray, z: float, h: float
               ) -> np.ndarray:
        """Call the __call__ method of the equation f."""

        return f(vectors, z, h)
