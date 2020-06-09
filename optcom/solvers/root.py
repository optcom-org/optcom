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
from optcom.solvers.jacobian import Jacobian


METHODS = ["newton_raphson"]


class Root(object):

    def __init__(self, func: Callable, method: str = cst.DFT_ROOTMETHOD,
                 error: float = 1e-6) -> None:
        self._func: Callable = func
        self._method: str = util.check_attr_value(method, METHODS,
                                                  cst.DFT_ROOTMETHOD)
        self._error: float = error
    # ==================================================================
    @overload
    def __call__(self, z: float, h: float) -> float: ...
    # ------------------------------------------------------------------
    @overload
    def __call__(self, z: np.ndarray, h: np.ndarray) -> np.ndarray: ...
    # ------------------------------------------------------------------
    def __call__(self, z, h):

        return getattr(Root, self._method)(self._func, z, h, self._error)
    # ==================================================================
    @overload
    @staticmethod
    def newton_raphson(func: Callable[[float], float], z: float, h: float,
                       error: float, max_iter: int) -> float: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def newton_raphson(func: Callable[[np.ndarray], np.ndarray], z: np.ndarray,
                        h: np.ndarray, error: float, max_iter: int
                        ) -> np.ndarray: ...
    # ------------------------------------------------------------------
    @staticmethod
    def newton_raphson(func, z, h, error=1e-6, max_iter=1e6):
        eval = func(z)
        if (not isinstance(z, np.ndarray)):
            root_0 = 0.
            root_1 = z
            i = 0
            while (not i or abs(root_1 - root_0) > error and i < max_iter):
                eval = func(root_1)
                jacob = Jacobian.calc_jacobian(func, root_1, h)
                jacob_inv = 1/jacob
                root_0 = root_1
                root_1 = root_1 - jacob_inv*eval
                i += 1
        else:
            if (len(z) != len(eval)):
                util.warning_terminal("The number of variables and vector "
                    "components must be equal in order to calculate "
                    "determinant, return zeros.")

                return np.zeros_like(z)
            else:
                root_0 = np.zeros_like(z)
                root_1 = z
                i = 0
                while (not i or np.linalg.norm(root_1-root_0) > error
                       and i < max_iter):
                    eval = func(root_1)
                    jacob = Jacobian.calc_jacobian(func, z, h)
                    jacob_inv = np.linalg.inv(jacob)
                    root_0 = root_1
                    root_1 = root_1 - np.matmul(jacob_inv, eval)
                    i += 1

        return root_1
    # ==================================================================
    @overload
    @staticmethod
    def modified_newton(func: Callable[[float], float], z: float, h: float,
                        error: float, max_iter: int) -> float: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def modified_newton(func: Callable[[np.ndarray], np.ndarray],
                        z: np.ndarray, h: np.ndarray, error: float,
                        max_iter: int) -> np.ndarray: ...
    # ------------------------------------------------------------------
    @staticmethod
    def modified_newton(func, z, h, error=1e-6, max_iter=1e6):

        def seq_fct(x):
            if (not isinstance(x, np.ndarray)):

                return x**2
            else:

                return np.matmul(x.T, x)

        first_iter = True
        eval = func(z)
        if (not isinstance(z, np.ndarray)):
            root_0 = 0.
            root_1 = z
            while (not j or abs(root_1 - root_0) > error and i < max_iter):
                eval = func(root_1)
                jacob = Jacobian.calc_jacobian(func, root_1, h)
                jacob_inv = 1/jacob
                root_0 = root_1
                j = 0
                seq = 0.
                conv = 0.
                while (not j or seq < conv):
                    lambda_ = 2**(-j)
                    root_0 = root_1
                    root_1 = root_1 - lambda_*jacob_inv*eval
                    seq = seq_fct(root_1)
                    conv = seq_fct(root_0)

                root_1 = root_1 - jacob_inv*eval
                first_iter = False

        else:
            if (len(z) != len(eval)):
                util.warning_terminal("The number of variables and vector "
                    "components must be equal in order to calculate "
                    "determinant, return zeros.")

                return np.zeros_like(z)
            else:
                root_0 = np.zeros_like(z)
                root_1 = z
                i = 0
                while (not i or np.linalg.norm(root_1-root_0) > error
                       and i < max_iter):
                    eval = func(root_1)
                    jacob = Jacobian.calc_jacobian(func, z, h)
                    jacob_inv = np.linalg.inv(jacob)

                    j = 0
                    seq = 0.
                    conv = 0.
                    root_0_temp = copy.copy(root_1)
                    while (not j or seq < conv):
                        lambda_ = 2**(-j)
                        gamma = 1. / np.linalg.cond(eval)
                        root_0_temp = root_1_temp
                        direction = np.matmul(jacob_inv, eval)
                        root_1_temp = root_0_temp - lambda_*direction
                        seq = seq_fct(root_0_temp)
                        seq_prec = seq_fct(root_0_temp)
                        seq_gradient = Jacobian.calc_jacobian(seq_fct,
                                                              root_0_temp, h)
                        conv = seq_prec - (np.linalg.norm(root_0_temp)
                                           * np.linalg.norm(seq_gradient)
                                           * gamma * lambda_ / 4.)

                    i += 1

        return root_1



if __name__ == "__main__":

    import math

    func_uni_in_uni_out = lambda x: x**2 - 1
    func_multi_in_multi_out = lambda x: np.asarray([x[0], x[1]**2 - 2,
                                                    x[2]**3 - 1])

    nbr_var: int = 3
    nbr_func: int = 3
    error: float = float(1e-3)

    var_uni: float = 3.
    #var_multi = np.random.rand(nbr_var)
    var_multi: np.ndarray = np.asarray([3., 3., 3.])

    h_uni: float = float(1e-10)
    h_multi: np.ndarray = np.ones(nbr_var, dtype=float) * h_uni

    print('################################ Functions test')
    print('func_uni_in_uni_out: ', func_uni_in_uni_out(var_uni))
    print('func_multi_in_multi_out: ', func_multi_in_multi_out(var_multi))

    print('################################ Newton Raphson test')
    print('newton raphson func_uni_in_uni_out: ',
          Root.newton_raphson(func_uni_in_uni_out, var_uni, h_uni, error,
                              int(1e6)))
    print('newton raphson func_multi_in_multi_out: ',
          Root.newton_raphson(func_multi_in_multi_out, var_multi, h_multi,
                              error, int(1e6)))

    print('################################ Modified Newton test')
    #print('modified newton func_uni_in_uni_out: ',
    #      Root.modified_newton(func_uni_in_uni_out, var_uni, h_uni))
    #print('modified newton func_multi_in_multi_out: ',
    #      Root.modified_newton(func_multi_in_multi_out, var_multi, h_multi))
