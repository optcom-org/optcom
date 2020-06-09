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


def get_max_len_list(x: List[List[Any]]) -> Tuple[int, int]:
    """Return the longest list and its index in a list of list.
    """
    max_len: int = 0
    max_len_ind: int = -1
    for i in range(len(x)):
        if (len(x[i]) > max_len):
            max_len = len(x[i])
            max_len_ind = i

    return max_len, max_len_ind


def equalize_len_list_elem(x: List[List[Any]], elem: Optional[Any] = None
                           ) -> List[List[Any]]:
    """Make all elements of a list of same length and pad the
    elements needed according to the longest element if elem is None,
    with the value of elem otherwise.
    """
    res: List[List[Any]] = []
    for item in x:
        res.append(item)
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
    res = []
    if (x is not None):
        for item in x:
            res.append(item)
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


def pad_list_with_last_elem(x: List[Any], y: List[Any], sym: bool = False
                            ) -> Tuple[List[Any], List[Any]]:
    """Pad list x with its last elem according of y list length. If sym
    is True, then pad also y depending on x.
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


def make_list(x: Any, length: int=1, elem: Any=None) -> List[Any]:
    """Make a list of length length out of x by padding x with last
    element of x if elem is None, elem otherwise. If x is None, return
    a list of length length where each element is None.
    """
    res = []
    if (x is None):
        res = [None for i in range(length)]
    elif (isinstance(x, list)):
        res = pad_with_last_elem(x, length, elem)
    else:   # other types
        res = pad_with_last_elem([x], length, elem)

    return res


def make_matrix(x: Any, nbr_lin: int = 1, nbr_col: int = 1,
                elem: Optional[Any] = None, sym: bool = False
                ) -> List[List[Any]]:
    r"""Make a matrix nbr_lin \times nbr_col out of x.
    Make a symmetric matrix if sym is True, pad with last element
    otherwise where the last element is the last element of x if elem is
    None, elem otherwise.
    """
    if (x is not None):
        res = []
        for item in x:
            res.append(item)
        elem_lin: Optional[List[Any]]
        temp_elem: Optional[Any]
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

    return [[None for i in range(nbr_col)] for i in range(nbr_lin)]


def make_tensor(x: Any, nbr_lin: int, nbr_col: int, depth: int = 0,
                elem: float = None) -> List[List[List[Any]]]:
    """Make a tensor out of x and pad with elem if not None, with last
    item otherwise. Tensor size is nbr_lin \times nbr_col \times depth.
    """
    res = []
    for item in x:
        res.append(item)
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


def is_all_elem_list_identical(x: List[Any]) -> bool:
    """Return True if all elemens of a list are identical, else False.
    """
    res: bool = True
    if (x is not None):
        if (isinstance(x, list) and len(x)):
            i = 1
            while (res and i < len(x)):
                if (x[0] != x[i]):
                    res = False
                i += 1

    return res


def sum_elem_list(x: Optional[List[Any]]) -> Optional[List[Any]]:
    """Sum all the elements of a list."""
    if (x is not None):
        if (x):
            sum = x[0]
            for i in range(1,len(x)):
                sum += x[i]

            return sum

    return None


def lower_str_list(x: List[str]) -> List[str]:
    """Lower all elements of a list of string."""
    for i in range(len(x)):
        x[i] = x[i].lower()

    return x


def permutations(x: List[Any]) -> List[List[Any]]:
    """Return all possible permutations of a list."""
    if (not x):
        return [[]]
    res = []
    for elem in x:
        temp = x[:]
        temp.remove(elem)
        res.extend([[elem] + perm for perm in permutations(temp)])

    return res


def unique(x: List[Any]) -> List[Any]:
    """Return a list with no repeated elements from a provided list."""
    res: List[Any] = []
    for elem in x:
        if (elem not in res):
            res.append(elem)

    return res


def unique_list(x: List[Any]) -> Tuple[List[Any], List[List[int]]]:
    """Return a list with no repeated elemens from a provided list as
    well as a list with repeated elements index in the initial list.
    """
    res: List[Any] = []
    ind: List[List[int]] = []
    for i, elem in enumerate(x):
        if (elem not in res):
            res.append(elem)
            ind.append([i])
        else:
            ind[res.index(elem)].append(i)

    return res, ind


def list_repetition_indices(x: List[Any]) -> List[List[int]]:
    """Return a list with repeated elements index in the initial list.
    """
    unique_elems: List[Any] = []
    inds: List[List[int]] = []
    for i, elem in enumerate(x):
        if (elem not in unique_elems):
            unique_elems.append(elem)
            inds.append([i])
        else:
            inds[unique_elems.index(elem)].append(i)

    return inds


def list_repetition_bool_map(x: List[Any], highlight_first_elem: bool = True
                             ) -> List[bool]:
    """Return a bool list of same dimension as the input list where the
    repeated element are set to False, else True.

    Parameters
    ----------
    x :
        List to consider.
    highlight_first_elem :
        If True, set to True the first occurrence of a repeated element,
        else set to True the last occurence of a repeated element.

    """
    res: List[bool] = [False for i in range(len(x))]
    ind_map = list_repetition_indices(x)
    for inds in ind_map:
        if (highlight_first_elem):
            res[inds[0]] = True
        else:
            res[inds[-1]] = True

    return res


@overload
def mean_list(x: List[int]) -> int: ...
@overload
def mean_list(x: List[float]) -> float: ...
@overload
def mean_list(x: List[complex]) -> complex: ...
def mean_list(x):
    """Return the mean of a list."""
    if (isinstance(x, int)):
        mean = 0
    elif (isinstance(x, complex)):
        mean = 0.0 + 0.0j
    else:
        mean = 0.0
    if (x):
        for i in range(len(x)):
            mean += x[i]

        return (mean / len(x))
    else:

        return mean


def is_float_in_list(var: float, list: List[float]) -> bool:
    """Return True if the float var is in list list and avoid rounding
    errors.
    """
    res: bool = False
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
