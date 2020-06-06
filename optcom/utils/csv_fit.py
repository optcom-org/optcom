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

from typing import Callable, List, Optional, overload

import numpy as np
from scipy import interpolate
from scipy.misc import derivative

import optcom.utils.utilities as util


class CSVFit(object):

    def __init__(self, file_name: str, delimiter: str = ',',
                 conv_func: List[Callable] = [],
                 conv_factor: List[float] = [1e0, 1e0],
                 order: Optional[int] = None, ext: int = 1):
        self.file_name = file_name
        self.delimiter = delimiter
        self.conv_func = conv_func
        self.conv_factor = conv_factor
        self.order = order
        self.ext = ext  # from scipy.interpolate.splev
        self._func: List[Callable]
        self._func = self._fit(file_name, delimiter, conv_func, conv_factor)
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

        self._func = self._fit(self.file_name, self.delimiter, self.conv_func,
                               self.conv_factor)
    # ==================================================================
    def _fit(self, file_name: str, delimiter: str, conv_func: List[Callable],
             conv_factor: List[float]) -> List[Callable]:
        # conv factor first before conv func
        func = []
        data, names = util.read_csv(file_name, delimiter)
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

    file_name = ('./data/fiber_amp/cross_section/absorption/yb.txt')
    csv = CSVFit(file_name, ',', conv_factor=[1e9, 1e18],
                 conv_func=[domain.Domain.lambda_to_omega])
    Lambda = 976.0
    omega = domain.Domain.lambda_to_omega(Lambda)
    print(csv(omega, 1))

    Lambda = np.arange(300) + 850.
    omega = domain.Domain.lambda_to_omega(Lambda)
    res_1 = csv(omega, 0)

    file_name = ('./data/fiber_amp/cross_section/emission/yb.txt')
    csv = CSVFit(file_name, ',', conv_factor=[1e9, 1e18],
                 conv_func=[domain.Domain.lambda_to_omega])
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
                y_labels=['sigma_a'], plot_labels=['absorption', 'emission'])
