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
from nptyping import Array
from scipy import interpolate
from scipy.misc import derivative

import optcom.utils.utilities as util


class CSVFit(object):

    def __init__(self, file_name: str, delimiter: str = ',',
                 conv_func: List[Callable] = [],
                 conv_factor: List[float] = [1e0, 1e0]):
        self.file_name = file_name
        self.delimiter = delimiter
        self.conv_func = conv_func
        self.conv_factor = conv_factor
        self._func: List[Callable]
        self._func = self._fit(file_name, delimiter, conv_func, conv_factor)
    # ==================================================================
    @overload
    def __call__(self, var: float, order: int) -> float: ...
    # ------------------------------------------------------------------
    @overload
    def __call__(self, var: Array[float], order: int) -> Array[float]: ...
    # ------------------------------------------------------------------
    def __call__(self, var, order=0):

        if (isinstance(var, float)):
            res =  [0.0 for i in range(order+1)]
        else:
            res = np.zeros((order+1, len(var)))
        for i in range(order+1):
            if (len(self._func) > i):
                res[i] = interpolate.splev(var, self._func[i])

        return res
    # ==================================================================
    def update(self):

        self._func = self._fit(self.file_name, self.delimiter, self.conv_func,
                               self.conv_factor)
    # ==================================================================
    def _fit(self, file_name: str, delimiter: str, conv_func: List[Callable],
             conv_factor: List[float]) -> List[Callable]:

        func = []
        data, names = util.read_csv(file_name, delimiter)
        if (not len(data)):
            util.warning_terminal("The csv file provided is either empty or "
                "do not comply with correct synthax.")
        else:
            x = np.asarray(data[0])
            if (len(conv_func)):
                x = conv_func[0](x)
            if (len(conv_factor)):
                x *= conv_factor[0]
            if (len(data) < 1):
                util.warning_terminal("The csv file provided contains only "
                    "one column, must provid at least two for fitting.")
            else:
                for i in range(1, len(data)):
                    y = np.asarray(data[i])
                    if (len(conv_func) > i):
                        y = conv_func[i](y)
                    if (len(conv_factor) > i):
                        y *= conv_factor[i]
                    # Make sure x data are increasing (needed for spl)
                    if (x[0] > x[-1]):
                        func.append(interpolate.splrep(x[::-1], y[::-1]))
                    else:
                        func.append(interpolate.splrep(x, y))

        return func

if __name__ == "__main__":

    file_name = ('./data/fiber_amp/cross_section/absorption/yb.txt')
    csv = CSVFit(file_name, ',', conv_factor=[1e0, 1e0, 2.0])
    print(csv(1.092e-06, 1))
