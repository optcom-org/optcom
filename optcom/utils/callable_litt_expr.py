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


# Exceptions
class CallableLittExprError(Exception):
    pass

class InitError(CallableLittExprError):
    pass


class CallableLittExpr(object):
    """This class allow to perform operations on functions of the same
    arguments.
    """

    def __init__(self, fcts: Sequence[Union[float, Callable]],
                 operators: List[str] = []) -> None:
        r"""
        Parameters
        ----------
        fcts :
            A series of callables or constants which composed the
            litteral expression to evaluate.
        operators :
            A list of str which correspond to the operator to apply
            between two elements of the litteral expression.  The length
            should be len(fcts) - 1.

        """
        self._fcts: Sequence[Union[float, Callable]] = fcts
        self._operators: List[str] = operators
        self._pre_and_suffix: bool
        if (len(self._fcts) == (len(self._operators)+1)):
            self._pre_and_suffix = False
        elif (len(self._fcts) == (len(self._operators)-1)):
            self._pre_and_suffix = True
        else:

            raise InitError("Length of functions in litteral expression "
                "should be equal to the length of operators minus one"
                "in case of pre- and suffix, or the length of operators added "
                "to one if no pre- or suffix. But here the operator list has "
                "length {} and the function list has length {}."
                .format(len(operators), len(fcts)))
    # ==================================================================
    def __call__(self, *vars: Any) -> Any:
        evals: List[Any] = []
        fct_or_nbr: Union[float, Callable]
        for i in range(len(self._fcts)):
            fct_or_nbr = self._fcts[i]
            if (callable(fct_or_nbr)):
                evals.append(fct_or_nbr(*vars))
            else:
                evals.append(fct_or_nbr)
        if (evals):
            res: Any
            dict_eval = {}
            if (self._operators):
                str_to_eval: str = ''
                if (self._pre_and_suffix):
                    str_to_eval += self._operators[0]
                for i in range(len(self._fcts)):
                    j = (i+1) if self._pre_and_suffix else i
                    str_to_add = "evals{}".format(i)
                    dict_eval[str_to_add] = evals[i]
                    str_to_eval += str_to_add
                    if ((i != len(self._fcts)-1) or self._pre_and_suffix):
                        str_to_eval += self._operators[j]
                res = eval(str_to_eval, {}, dict_eval)
            else:
                res = evals[0]

            return res

        return None


if __name__ == "__main__":

    a: Callable = lambda x : x**2
    b: Callable = lambda y : 1 / y
    d: Callable = lambda x : x
    CLE: CallableLittExpr = CallableLittExpr([a, b, a, d, b],
                                             ['+', '/', '-', '*'])
    print(CLE(3.))
