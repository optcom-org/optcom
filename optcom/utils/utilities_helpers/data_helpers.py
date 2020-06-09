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

from typing import Any, Callable, List, Optional, overload, Set, Tuple, Union

import csv

import numpy as np
from scipy import interpolate
from scipy.misc import derivative


def read_csv(file_name: str, delimiter: str = ','
             ) -> Tuple[List[List[Any]], List[str]]:
    """Read a csv file. Evaluate empty cell to zero."""
    with open(file_name, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=delimiter)
        res: List[Any] = []
        names: List[str] = []
        line_count: int = 0
        nbr_col: int = 0
        for row in csv_reader:
            if (line_count == 0):
                names = row
                nbr_col = len(names)
                res = [[] for i in range(nbr_col)]
            else:
                if (row): #empty line
                    for i in range(nbr_col):
                        if (not row[i]):
                            res[i].append(0.0)
                        else:
                            res[i].append(float(row[i]))
            line_count += 1

        return res, names


def write_csv(data: List[List[Any]], names: List[str], file_name: str,
              delimiter: str = ',') -> None:
    """Write data as a csv file file_name."""
    with open(file_name, mode='w', newline='') as csv_file:
        wr = csv.writer(csv_file, delimiter=delimiter)
        wr.writerow(names)
        nbr_col = len(names)
        for i in range(len(data[0])):
            row = []
            for j in range(nbr_col):
                row.append(data[j][i])
            wr.writerow(row)


def fit_data(x: Union[List, np.ndarray], y: Union[List, np.ndarray],
             fill_down: float = 0.0, fill_up: float = 0.0,
             extrapolate: bool = False) -> Callable:
    """Return a callable object which extrapolate the data x and y.
    For more, see documentation of scipy.interpolate.
    """
    if (extrapolate):
        inter = interpolate.interp1d(x, y, fill_value='extrapolate',
                                     bounds_error=False)
    else:
        inter = interpolate.interp1d(x, y, fill_value=(fill_down, fill_up),
                                     bounds_error=False)

    return inter


@overload
def deriv(fct: Callable, x: float, order_deriv: int) -> float: ...
@overload
def deriv(fct: Callable, x: np.ndarray, order_deriv: int
          ) -> np.ndarray: ...
def deriv(fct, x, order_deriv=0):
    r"""Compute the n^{th} derivative of the function fct.

    Parameters
    ----------
    fct :
        Function to call to calculate derivatives.
    x :
        The parameters.
    order_deriv :
        The order of the derivative. (0 <= order <= 5)

    Returns
    -------
    :
        The derivative of the function fct.

    """
    if (not order_deriv):

        res = fct(x)
    else:
        order = max(3, order_deriv+1+(order_deriv%2))
        if (isinstance(x, float)):
            res = 0.0
            res = derivative(fct, x, n=order_deriv, order=order)
        else:
            res = np.zeros_like(x)
            for i in range(len(x)):
                res[i] = derivative(fct, x[i], n=order_deriv, order=order)

    return res
