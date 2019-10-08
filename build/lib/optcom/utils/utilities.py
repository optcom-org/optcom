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

import copy
import operator
import csv
import os
from typing import Any, List, Optional, Tuple, Union

import numpy as np
from scipy import interpolate
from nptyping import Array

import optcom.utils.constants as cst

def get_max_len_list(x: List[List[Any]]) -> Tuple[int, int]:
    max_len = 0
    max_len_ind = None
    for i in range(len(x)):
        if (len(x[i]) > max_len):
            max_len = len(x[i])
            max_len_ind = i

    return max_len, max_len_ind

def equalize_len_list_elem(x: Optional[List[List[Any]]],
                           elem: Optional[Any] = None
                           ) -> Optional[List[List[Any]]]:
    """Make all elements of a list of same length and pad the
    elements needed according to the longest element if elem is None,
    with the value of elem otherwise.
    """
    res = copy.deepcopy(x)
    if (x is not None):
        max_len, max_len_ind = get_max_len_list(x)
        for i in range(len(x)):
            if (max_len_ind is not None and len(x[i]) < max_len):
                if (elem is not None):
                    res[i].extend([elem for j in range(max_len-len(x[i]))])
                else:
                    res[i].extend(x[max_len_ind][len(x[i]):max_len])

    return res

def pad_with_last_elem(x: List[Any], length: int, elem: Optional[Any] = None
                       ) -> List[Any]:
    """Pad list x up to length with its last element if elem is None,
    with elem otherwise.
    """
    res = copy.deepcopy(x)
    if (x is not None):
        if (len(x) != length):
            diff = len(x) - length
            if (diff < 0):
                if (elem is None):
                    last_elem = x[-1]
                else:
                    last_elem = elem
                for i in range(-diff):
                    res.append(last_elem)

    return res

def pad_list_with_last_elem(x, y, sym=False):
    """Pad list x with its last elem according of y list length, if sym
    is True, then pad also y.
    """
    if (x is not None):
        if (len(x) != len(y)):
            diff = len(x) - len(y)
            if (diff > 0 and sym):
                for i in range(diff):
                    y.append(y[-1])
            if (diff < 0):
                for i in range(-diff):
                    x.append(x[-1])
    else:
        x = [None for i in range(len(y))]

    return x, y


def pad_np_ndarray_with_last_elem(x, y, sym=False):

    if (x is not None):
        if (x.shape != y.shape):
            diff = x.shape[0] - y.shape[0]
            if (diff > 0 and sym):
                y = np.pad(y, (0,diff), 'constant',
                           constant_values=(0,y[-1]))
            if (diff < 0):
                x = np.pad(x, (0,diff), 'constant',
                           constant_values=(0,x[-1]))
    else:
        x = np.zeros(y.shape[0])

    return x, y


def make_list(x: Optional[Any], length: int = 1, elem: Optional[any] = None
              ) -> List[Optional[Any]]:
    """Make a list of length length out of x from a number by padding
    with this number if elem is None, elem otherwise.
    """
    res = copy.deepcopy(x)
    if (x is None):
        res = [None for i in range(length)]
    elif (isinstance(x, list)):
        res = pad_with_last_elem(x, length, elem)
    else:   # float
        res = pad_with_last_elem([x], length, elem)

    return res


def make_matrix(x: Optional[Any], nbr_lin: int = 1, nbr_col: int = 1,
                elem: Optional[Any] = None, sym: bool = False
                ) -> Optional[Any]:
    """Make a matrix nbr_lin \times nbr_col from a number or a list,
    make a symmetric matrix if sym is True, pad with last element
    otherwise where the last element is from x if elem is None, elem
    otherwise.
    """
    res = copy.deepcopy(x)
    if (x is not None):
        if (elem is not None):
            elem_lin = [elem for i in range(nbr_col)]
        else:
            elem_lin = None
        if (isinstance(x, list)):   # list
            if(len(x) and isinstance(x[0], list)):  # list of list
                if (sym):
                    max_len, max_len_ind = get_max_len_list(x)
                    for i in range(nbr_lin):
                        if (len(x) <= i):
                            res.append([])
                        for j in range(len(res[i]), nbr_col):
                            # if the res[j][i] elem exists
                            if (len(res) > j and len(res[j]) > i):
                                res[i].append(res[j][i])
                            else:
                                # Take already existing elem in the jth col.
                                if (max_len > j):
                                    temp_elem = x[max_len_ind][j]
                                else:
                                    temp_elem = elem
                                # Add one more elem on the ith line
                                res[i] = pad_with_last_elem(res[i], j+1,
                                                            temp_elem)
                else:
                    res = equalize_len_list_elem(res, elem)
                    res[0] = pad_with_last_elem(res[0], nbr_col, elem)
                    res = equalize_len_list_elem(res)
                    res = pad_with_last_elem(res, nbr_lin, elem_lin)
            else:
                res = pad_with_last_elem(res, nbr_col, elem)
                res = pad_with_last_elem([res], nbr_lin, elem_lin)
        else:   # float
            res = pad_with_last_elem([res], nbr_col, elem)
            res = pad_with_last_elem([res], nbr_lin, elem_lin)
    return res


def make_tensor(x: Optional[Any], nbr_lin: int, nbr_col: int,
                depth: int = 0, elem: float = None) -> Optional[Any]:
    """Make a tensor out of x and pad with elem if not None, with last
    item otherwise. Tensor size is nbr_lin \times nbr_col \times depth.
    """
    res = copy.deepcopy(x)
    if (x is not None):
        if (isinstance(res, list)):   # Is a list
            for i in range(len(x)):
                res[i] = make_matrix(x[i], nbr_col, depth, elem)
            for i in range(len(x), nbr_lin):
                if (elem is None):
                    res.append(res[-1])
                else:
                    res.append(make_matrix(elem, nbr_col, depth))
        else:   # Is a number
            if (not depth):
                depth = 1
            if (elem is None):
                elem = res
            res = [[[elem for k in range(depth)] for j in range(nbr_col)]
                   for i in range(nbr_lin)]
            res[0][0][0] = x

    return res


def is_all_elem_identical(x: List[Any]) -> bool:
    res = True
    if (x is not None):
        if (isinstance(x, list) and len(x)):
            i = 1
            while (res and i < len(x)):
                if (x[0] != x[i]):
                    res = False
                i += 1

    return res


def sum_elem_list(x):

    if (x is not None):
        if (x):
            sum = x[0]
            for i in range(1,len(x)):
                sum += x[i]

            return sum

    return None


def lower_str_list(x):

    for i in range(len(x)):
        x[i] = x[i].lower()

    return x


def print_terminal(to_print=None, sep_type=cst.STR_SEPARATOR_TERMINAL):

    print(sep_type, end='')
    if (to_print is not None):
        print(to_print)


def warning_terminal(to_print, sep_type=cst.STR_SEPARATOR_TERMINAL):

    print(sep_type, end='')
    print("!WARNING!: ", to_print)


def check_attr_type(attr, attr_name, *types):
    res = False
    for type in types:
        #print('typ', type)
        if (type is None):    # Allow attr to be None
            if (attr is None):
                res = True
        else:
            res = res or isinstance(attr, type)   # Keep the True
    if (not res):
        types_name = locals()['types']
        raise TypeError("{} must be one of the following type: {}"
                        .format(attr_name, types_name))


def check_attr_range(attr, attr_name, min_attr, max_attr, strict_left=False,
                     strict_right=False):

    if (strict_left):
        op_left = operator.lt
    else:
        op_left = operator.le
    if (strict_right):
        op_right = operator.gt
    else:
        op_right = operator.ge

    if (not(op_left(min_attr, attr) and op_right(max_attr, attr))):
        raise IndexError("Attribute {} out of range, must be in interval "
                         "[{},{}]"
                         .format(attr_name, min_attr, max_attr))


def permutations(x: List[Any]) -> List[List[Any]]:
    if (not x):
        return [[]]
    res = []
    for elem in x:
        temp = x[:]
        temp.remove(elem)
        res.extend([[elem] + perm for perm in permutations(temp)])

    return res


def unique(x: List[Any]) -> List[Any]:
    res = []
    for elem in x:
        if (elem not in res):
            res.append(elem)

    return res


def mean_list(x: List[Union[int, float, complex]]
              ) -> Union[int, float, complex]:

    if (x):
        mean = 0.0
        for i in range(len(x)):
            mean += x[i]

        return (mean / len(x))
    else:

        return 0.0


def read_csv(file_name: str, delimiter: str = ','
             ) -> Tuple[List[List[Any]], List[str]]:
    """Read a csv file. Evaluate empty cell to zero."""

    with open(file_name, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=delimiter)
        res = []
        names = []
        line_count = 0
        nbr_col = 0
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

    with open(file_name, mode='w', newline='') as csv_file:
        wr = csv.writer(csv_file, delimiter=delimiter)
        wr.writerow(names)
        nbr_col = len(names)
        for i in range(len(data[0])):
            row = []
            for j in range(nbr_col):
                row.append(data[j][i])
            wr.writerow(row)


def combine_to_one_array(x_data):

    if (x_data.ndim > 1):  # more than one x array
        min_value = x_data[0][0]
        max_value = x_data[0][-1]
        for i in range(1, x_data.shape[0]):
            if (min_value > x_data[i][0]):
                min_value = x_data[i][0]
            if (max_value < x_data[i][-1]):
                max_value = x_data[i][-1]
        dx = x_data[0][1] - x_data[0][0]
        x_samples = int(round(((max_value - min_value) / dx) + 1))
        x_data = np.linspace(min_value, max_value, x_samples, True)

    return x_data


def auto_pad(x_data, y_data, value_left=0.0, value_right=0.0):
    # consider x_data and y_data have same dim
    # and all arrays of x_data or y_data have same step size.

    if (x_data.ndim > 1):  # more than one x array
        if (x_data.shape[1] != y_data.shape[1]):
            warning_terminal("auto_pad utilities function is not made to "
                "work with different x and y data size")
        min_value = x_data[0][0]
        max_value = x_data[0][-1]
        for i in range(1, x_data.shape[0]):
            if (min_value > x_data[i][0]):
                min_value = x_data[i][0]
            if (max_value < x_data[i][-1]):
                max_value = x_data[i][-1]
        dx = x_data[0][1] - x_data[0][0]
        values = (value_left, value_right)
        x_samples = int(round(((max_value - min_value) / dx) + 1))
        for i in range(x_data.shape[0]):
            pad_left = int(round((x_data[i][0] - min_value) / dx))
            pad_right = int(round((max_value - x_data[i][-1]) / dx))
            # Arbitrarely changing padding if rounding effect didn't
            # work out. (should make that more clear)
            y_samples = pad_left + pad_right + y_data.shape[1]
            while (y_samples != x_samples):
                if (y_samples > x_samples):
                    pad_left -= 1
                else:
                    pad_left += 1
                y_samples = pad_left + pad_right + y_data.shape[1]
            array_to_add = np.pad(y_data[i], (pad_left, pad_right),
                                  'constant', constant_values=values)
            if (not i):
                y_data_new = array_to_add.reshape((1,-1))
            else:
                y_data_new = np.vstack((y_data_new, array_to_add))
        x_data = np.linspace(min_value, max_value, x_samples, True)
    else:
        y_data_new = y_data.reshape(1,-1)

    return x_data, y_data_new


def fit_data(x, y, fill_down=0.0, fill_up=0.0, extrapolate=False):

    if (extrapolate):
        inter = interpolate.interp1d(x, y, fill_value='extrapolate',
                                     bounds_error=False)
    else:
        inter = interpolate.interp1d(x, y, fill_value=(fill_down, fill_up),
                                     bounds_error=False)

    return inter


def is_float_in_list(var: float, list: List[float]) -> bool:
    " Avoid rounding error"
    res = False
    nbr_decimal_var = str(var)[::-1].find('.')
    for elem in list:
        nbr_decimal_elem = str(elem)[::-1].find('.')
        if (nbr_decimal_elem < nbr_decimal_var):
            if (elem == round(var, nbr_decimal_elem)):
                res = True
        else:
            if (var == round(elem, nbr_decimal_var)):
                res = True

    return res
