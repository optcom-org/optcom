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
from optcom.solvers.gradient import Gradient


class Jacobian(object):

    def __init__(self) -> None:

        return None
    # ==================================================================
    @overload
    @staticmethod
    def calc_jacobian(func: Callable[[float], float], z: float, h: float
                      ) -> float: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def calc_jacobian(func: Callable[[np.ndarray], np.ndarray], z: np.ndarray,
                      h: np.ndarray) -> np.ndarray: ...
    # ------------------------------------------------------------------
    @staticmethod
    def calc_jacobian(func, z, h):

        if (not isinstance(z, np.ndarray)):
            jacobian = Gradient.diff(func, z, h)

        else:
            if (len(z) != len(h)):
                util.warning_terminal("The variable and step vectors must "
                    "have same dimension, return zeros.")
                jacobian = np.zeros_like(z)
            else:
                jacobian = np.zeros(0)
                for i in range(len(z)):
                    to_add  = Gradient.diff(func, z, h, i)
                    if (not i):
                        jacobian = to_add
                    else:
                        jacobian = np.hstack((jacobian, to_add))

        return jacobian


if __name__ == "__main__":

    import math

    func_uni_in_uni_out = lambda x: x**2
    func_uni_in_multi_out = lambda x: np.asarray([x, x**2, x**3])
    func_multi_in_uni_out = lambda x: x[0]**2 + x[1]**2
    func_multi_in_multi_out = lambda x: np.asarray([x[0], x[0]**2, x[1]**3])

    nbr_var = 2
    nbr_func = 3

    var_uni = 3
    #var_multi = np.random.rand(nbr_var)
    var_multi = np.asarray([3., 3.])

    h_uni = 1e-6
    h_multi = np.ones(nbr_var, dtype=float) * h_uni

    print('################################ Functions test')
    print('func_uni_in_uni_out: ', func_uni_in_uni_out(var_uni))
    print('func_uni_in_multi_out: ', func_uni_in_multi_out(var_uni))
    print('func_multi_in_uni_out: ', func_multi_in_uni_out(var_multi))
    print('func_multi_in_multi_out: ', func_multi_in_multi_out(var_multi))

    print('################################ Jacobian test')
    print('jacobian func_uni_in_uni_out: ',
          Jacobian.calc_jacobian(func_uni_in_uni_out, var_uni, h_uni))
    print('jacobian func_uni_in_multi_out: ',
          Jacobian.calc_jacobian(func_uni_in_multi_out, var_uni, h_uni))
    print('jacobian func_multi_in_uni_out: ',
          Jacobian.calc_jacobian(func_multi_in_uni_out, var_multi, h_multi))
    print('jacobian func_multi_in_multi_out: ',
          Jacobian.calc_jacobian(func_multi_in_multi_out, var_multi, h_multi))

    print('################################ Error test')
    print('jacobian func_multi_in_multi_out: ',
          Jacobian.calc_jacobian(func_multi_in_multi_out, var_multi,
                                 h_multi[:nbr_var-2]))
