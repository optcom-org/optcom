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

import copy
from abc import ABCMeta
from typing import Callable, List, Optional

import numpy as np

import optcom.utils.constants as cst
import optcom.utils.utilities as util

# Typing variables
SOLVER_CALLABLE_TYPE = Callable[[np.ndarray, float, float], np.ndarray]
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
