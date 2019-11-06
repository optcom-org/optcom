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
from typing import Callable, List, Optional

from nptyping import Array

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.equations.abstract_equation import AbstractEquation


class AbstractSolver(object):

    def __init__(self, f: AbstractEquation, method: Optional[str] = None):
        """
        Parameters
        ----------
        f : AbstractEquation
            The function to compute.
        method :
            The computation method.
        """
        self.name: str
        self._method: Callable
        if (method is None):    # analytical solution, no need numerical
            self.name = 'f_call'
            self._method = getattr(self, self.name)
        elif (hasattr(self, method.lower())):   # Force to be static method
            self.name = method.lower()
            self._method = getattr(self, self.name)
        else:
            print(self.__class__)
            print(self.__class__._default_method)
            util.warning_terminal("The solver method '{}' does not exist, "
                                  "default solver '{}' is set."
                                  .format(method,
                                          self.__class__._default_method))
            self.name = self.__class__._default_method
            self._method = getattr(self, self.__class__._default_method)
        self.f: AbstractEquation = f
    # ==================================================================
    def __call__(self, vectors: Array, h: float, z: float) -> Array:
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
        self.f.set(vectors, h, z)
        res = self._method(self.f, vectors, h, z)
        self.f.update(vectors, h, z)

        return res
    # ==================================================================
    @staticmethod
    def f_call(f: AbstractEquation, vectors: Array, h: float, z: float
               ) -> Array:
        """Call the __call__ method of the equation f."""

        return f(vectors, h, z)
