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

import os
from typing import Callable, List, Optional, overload

import numpy as np
from scipy import interpolate
from scipy.misc import derivative

import optcom.utils.utilities as util


class CSVFit(object):

    def __init__(self, file_name: str, delimiter: str = ',',
                 conv_func: List[Callable] = [],
                 conv_factor: List[float] = [1e0, 1e0],
                 order: Optional[int] = None, ext: int = 1,
                 root_dir: str = '.'):
        self._full_path_to_file = os.path.join(root_dir, file_name)
        self.delimiter = delimiter
        self.conv_func = conv_func
        self.conv_factor = conv_factor
        self.order = order
        self.ext = ext  # from scipy.interpolate.splev
        self._func: List[Callable]
        self._func = self._fit(self._full_path_to_file, delimiter, conv_func,
                               conv_factor)
    # ==================================================================
    @overload
    def __call__(self, var: float, order: Optional[int]) -> float: ...
    # ------------------------------------------------------------------
    @overload
    def __call__(self, var: np.ndarray, order: Optional[int]
                 ) -> np.ndarray: ...
    # ------------------------------------------------------------------
    def __call__(self, var, order=None):
        order_: Optional[int] = self.order if order is None else order
        # order can still be None if no order specified at constructor.
        # This allows one to have an 1-D array.
        if (order is None):
            if (isinstance(var, float)):
                res =  0.0
            else:
                res = np.zeros(len(var))
            res = interpolate.splev(var, self._func[0], ext=self.ext)
        else:
            if (isinstance(var, float)):
                res =  np.zeros((order+1, 1))
            else:
                res = np.zeros((order+1, len(var)))
            for i in range(order+1):
                if (len(self._func) > i):
                    res[i] = interpolate.splev(var, self._func[i],
                                               ext=self.ext)

        return res
    # ==================================================================
    def update(self):

        self._func = self._fit(self._full_path_to_file, self.delimiter,
                               self.conv_func, self.conv_factor)
    # ==================================================================
    def _fit(self, file_path: str, delimiter: str, conv_func: List[Callable],
             conv_factor: List[float]) -> List[Callable]:
        # conv factor first before conv func
        func = []
        data, names = util.read_csv(file_path, delimiter)
        if (not len(data)):
            util.warning_terminal("The csv file provided is either empty or "
                "do not comply with correct synthax.")
        else:
            x = np.asarray(data[0])
            if (len(conv_factor)):
                x *= conv_factor[0]
            if (len(conv_func)):
                x = conv_func[0](x)
            if (len(data) < 1):
                util.warning_terminal("The csv file provided contains only "
                    "one column, must provid at least two for fitting.")
            else:
                for i in range(1, len(data)):
                    y = np.asarray(data[i])
                    if (len(conv_factor) > i):
                        y *= conv_factor[i]
                    if (len(conv_func) > i):
                        y = conv_func[i](y)
                    # Make sure x data are increasing (needed for spl)
                    if (x[0] > x[-1]):
                        func.append(interpolate.splrep(x[::-1], y[::-1]))
                    else:
                        func.append(interpolate.splrep(x, y))

        return func

if __name__ == "__main__":

    import optcom.domain as domain
    import optcom.utils.plot as plot

    root_dir = './data/fiber_amp/cross_section/absorption/'
    file_name = ('yb.txt')
    csv = CSVFit(file_name, ',', conv_factor=[1e9, 1e18],
                 conv_func=[domain.Domain.lambda_to_omega],
                 root_dir=root_dir)
    Lambda = 976.0
    omega = domain.Domain.lambda_to_omega(Lambda)
    print(csv(omega, 1))

    Lambda = np.arange(300) + 850.
    omega = domain.Domain.lambda_to_omega(Lambda)
    res_1 = csv(omega, 0)

    root_dir = "./data/fiber_amp/cross_section/emission/"
    file_name = ('yb.txt')
    csv = CSVFit(file_name, ',', conv_factor=[1e9, 1e18],
                 conv_func=[domain.Domain.lambda_to_omega],
                 root_dir=root_dir)
    res_2 = csv(omega, 0)
    print('1010 : ', csv(domain.Domain.lambda_to_omega(1010.), 0))
    print('1015 : ', csv(domain.Domain.lambda_to_omega(1015.), 0))
    print('1020 : ', csv(domain.Domain.lambda_to_omega(1020.), 0))
    print('1025 : ', csv(domain.Domain.lambda_to_omega(1025.), 0))
    print('1030 : ', csv(domain.Domain.lambda_to_omega(1030.), 0))
    Lambda_temp = np.arange(150) + 1000.
    omega_temp = domain.Domain.lambda_to_omega(Lambda_temp)
    res_temp = csv(omega_temp, 0)
    print('max at ',
          domain.Domain.omega_to_lambda(omega_temp[np.argmax(res_temp)]),
          ' : ', np.amax(res_temp))

    plot.plot2d([Lambda], [res_1, res_2], x_labels=['Lambda'],
                y_labels=['sigma_a'], line_labels=['absorption', 'emission'])
