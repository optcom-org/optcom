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

import operator
from typing import Any, Callable, List, Sequence, Union

import optcom.utils.utilities as util


class CallableContainer(object):
    """This class host a function which have argument that can be
    functions themselves (of the same arguments). The call to __call__
    allows to pass this common arguments to the function arguments of
    the main function and pass the results to the main function.
    """

    def __init__(self, function: Callable,
                 variables: Sequence[Union[float, Callable]] = []) -> None:
        r"""
        Parameters
        ----------
        function :
            The main callable which will be call in the __call__.
        variables :
            Variables of the function.

        """
        self._function: Callable = function
        self._variables: Sequence[Union[float, Callable]] = variables
    # ==================================================================
    def __call__(self, *vars: Any) -> Any:
        if (self._variables):
            evals: List[Any] = []
            fct_or_nbr: Union[float, Callable]
            for i in range(len(self._variables)):
                fct_or_nbr = self._variables[i]
                if (callable(fct_or_nbr)):
                    evals.append(fct_or_nbr(*vars))
                else:
                    evals.append(fct_or_nbr)

            return self._function(*evals)
        else:

            return self._function(*vars)
