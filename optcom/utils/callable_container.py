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
