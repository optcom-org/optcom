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
from typing import Callable, List, Optional, overload, Union

import numpy as np

import optcom.utils.constants as cst
import optcom.utils.utilities as util


METHODS = ["diff"]


class Gradient(object):

    def __init__(self, func: Callable, method: str = cst.DFT_GRADIENTMETHOD
                 ) -> None:
        self._func: Callable = func
        self._method: str = util.check_attr_value(method, METHODS,
                                                  cst.DFT_GRADIENTMETHOD)
    # ==================================================================
    @overload
    def __call__(self, z: float, h: float, ind: int) -> float: ...
    # ------------------------------------------------------------------
    @overload
    def __call__(self, z: np.ndarray, h: np.ndarray, ind: int
                 ) -> np.ndarray: ...
    # ------------------------------------------------------------------
    def __call__(self, z, h, ind):

        return getattr(Gradient, self._method)(self._func, z, h, ind)
    # ==================================================================
    @overload
    @staticmethod
    def diff(func: Callable, z: float, h: float, ind: int) -> float: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def diff(func: Callable, z: np.ndarray, h: np.ndarray, ind: int
             ) -> np.ndarray: ...
    # ------------------------------------------------------------------
    @staticmethod
    def diff(func, z, h, ind=0):

        # ind is index of variable for multi var function

        eval_0 = func(z)
        if (isinstance(eval_0, np.ndarray)):
            diff = np.zeros((len(eval_0), 1))
            for i in range(len(eval_0)):
                if (not isinstance(z, np.ndarray)):
                    z_1 = z + h
                    eval_1 = func(z_1)
                    diff[i][0] = (eval_1[i] - eval_0[i]) / h
                else:
                    z_1 = copy.copy(z)
                    z_1[ind] += h[ind]
                    eval_1 = func(z_1)
                    diff[i][0] = (eval_1[i] - eval_0[i]) / h[ind]

            return diff
        else:
            if (not isinstance(z, np.ndarray)):
                z_1 = z + h
                eval_1 = func(z_1)
                diff = (eval_1 - eval_0) / h
            else:
                z_1 = copy.copy(z)
                z_1[ind] += h[ind]
                eval_1 = func(z_1)
                diff = (eval_1 - eval_0) / h[ind]

            return diff


if __name__ == "__main__":

    import numpy as np

    func_uni_in_uni_out: Callable = lambda x: x**2
    func_uni_in_multi_out: Callable = lambda x: np.asarray([x, x**2, x**3])
    func_multi_in_uni_out: Callable = lambda x: x[0]**2 + x[1]**2
    func_multi_in_multi_out: Callable = lambda x: np.asarray([x[0], x[0]**2,
                                                              x[1]**3])

    nbr_var: int = 2
    nbr_func: int = 3

    var_uni: float = 3.
    #var_multi = np.random.rand(nbr_var)
    var_multi = np.asarray([3., 3.])

    h_uni: float = float(1e-6)
    h_multi: np.ndarray = np.ones(nbr_var, dtype=float) * h_uni

    print('################################ Functions test')
    print('func_uni_in_uni_out: ', func_uni_in_uni_out(var_uni))
    print('func_uni_in_multi_out: ', func_uni_in_multi_out(var_uni))
    print('func_multi_in_uni_out: ', func_multi_in_uni_out(var_multi))
    print('func_multi_in_multi_out: ', func_multi_in_multi_out(var_multi))

    print('################################ Gradient test with var 0')
    print('gradient func_uni_in_uni_out: ',
          Gradient.diff(func_uni_in_uni_out, var_uni, h_uni, 0))
    print('gradient func_uni_in_multi_out: ',
          Gradient.diff(func_uni_in_multi_out, var_uni, h_uni, 0))
    print('gradient func_multi_in_uni_out: ',
          Gradient.diff(func_multi_in_uni_out, var_multi, h_multi, 0))
    print('gradient func_multi_in_multi_out: ',
          Gradient.diff(func_multi_in_multi_out, var_multi, h_multi, 0))

    print('################################ Gradient test with var 1')
    print('gradient func_uni_in_uni_out: ',
          Gradient.diff(func_uni_in_uni_out, var_uni, h_uni, 1))
    print('gradient func_uni_in_multi_out: ',
          Gradient.diff(func_uni_in_multi_out, var_uni, h_uni, 1))
    print('gradient func_multi_in_uni_out: ',
          Gradient.diff(func_multi_in_uni_out, var_multi, h_multi, 1))
    print('gradient func_multi_in_multi_out: ',
          Gradient.diff(func_multi_in_multi_out, var_multi, h_multi, 1))
