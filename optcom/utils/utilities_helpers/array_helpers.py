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

import numpy as np

from optcom.utils.utilities_helpers.terminal_display_helpers \
    import warning_terminal


def pad_array_with_last_elem(x: np.ndarray, y: np.ndarray, sym: bool = False
                             ) -> np.ndarray:
    """Pad a numpy ndarray x with the last elemen according to array y.
    """
    if (x is not None):
        if (x.shape != y.shape):
            diff = x.shape[0] - y.shape[0]
            if (diff > 0 and sym):
                y = np.pad(y, (0,diff), 'constant',
                           constant_values=(0,y[-1]))
            if (diff < 0):
                x = np.pad(x, (0,abs(diff)), 'constant',
                           constant_values=(0,x[-1]))
    else:
        x = np.zeros(y.shape[0])

    return x, y

def combine_to_one_array(x_data: np.ndarray) -> np.ndarray:
    """Combine a multidimension to one array respecting spacing in
    original array. Original arrays must have same spacing between
    elements.
    """
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


def auto_pad(x_data: np.ndarray, y_data: np.ndarray, value_left: float = 0.0,
             value_right: float = 0.0):
    """Pad x_data according to y_data. x_data and y_data must have same
    dimension and space between elemens of y_data or x_data must be
    equal. The padding on left is fill with value_left as well as the
    padding on the right with value_right.
    """
    x_data_new: np.ndarray = np.array([])
    y_data_new: np.ndarray = np.array([])
    if (x_data.ndim > 1):  # more than one x array
        if (x_data.shape[1] != y_data.shape[1]):
            warning_terminal("auto_pad utilities function is not made to "
                "work with different x and y data size.")
        if (x_data.ndim == 2):
            min_value = x_data[0][0]
            max_value = x_data[0][-1]
            nbr_dec_first = str(x_data[0][0])[::-1].find('.')
            nbr_dec_last = str(x_data[0][-1])[::-1].find('.')
            flag = True
            for i in range(1, x_data.shape[0]):
                flag = (x_data[0][0] == round(x_data[i][0], nbr_dec_first)
                        and x_data[0][-1] == round(x_data[i][-1], nbr_dec_last)
                        and flag)
                if (min_value > x_data[i][0]):
                    min_value = x_data[i][0]
                if (max_value < x_data[i][-1]):
                    max_value = x_data[i][-1]
            if (not flag):  # Avoid computation if no need
                dx = x_data[0][1] - x_data[0][0]
                values = (value_left, value_right)
                x_samples = int(round(((max_value - min_value) / dx) + 1))
                y_data_new = np.zeros((y_data.shape[0], x_samples))
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
                    y_data_new[i] = np.pad(y_data[i], (pad_left, pad_right),
                                           'constant', constant_values=values)
                x_data_new = np.linspace(min_value, max_value, x_samples, True)
            else:
                y_data_new = y_data
                x_data_new = x_data[0]
        elif (x_data.ndim == 3):
            x_shape = x_data.shape
            y_shape = y_data.shape
            x_data_shaped = x_data.reshape((x_shape[0]*x_shape[1], x_shape[2]))
            y_data_shaped = y_data.reshape((y_shape[0]*y_shape[1], y_shape[2]))
            x_data_new, y_data_new = auto_pad(x_data_shaped, y_data_shaped,
                                              value_left, value_right)
            y_data_new = y_data_new.reshape((y_shape[0], y_shape[1],
                                             y_data_new.shape[1]))
        else:
            warning_terminal("auto_pad does not accept tensor with more than "
                "3 dimensions.")
    else:
        x_data_new = x_data
        y_data_new = y_data.reshape(1,-1)

    return x_data_new, y_data_new


def modify_length_ndarray(ndarray: np.ndarray, length: int):
    """Modify the length of the array ndarray according to the provided
    length.
    """
    if (length < len(ndarray)):
        ndarray = ndarray[:length]
    else:
        for i in range(len(ndarray), length):
            ndarray = np.vstack((ndarray, ndarray[-1]))

    return ndarray


def vstack_ndarray(ndarray: np.ndarray, length: int) -> np.ndarray:
    """Stack vertically the array ndarray length times."""

    last = np.array([ndarray[-1]])
    while (ndarray.shape[0] < length):
        ndarray = np.vstack((ndarray, last))

    return ndarray


def hstack_ndarray(ndarray: np.ndarray, length: int) -> np.ndarray:
    """Stack horizontally the array ndarray length times."""

    last = np.array([ndarray[-1]])
    while (ndarray.shape[0] < length):
        ndarray = np.hstack((ndarray, last))

    return ndarray


def hstack_like(ndarray: np.ndarray, ndarray_ref: np.ndarray) -> np.ndarray:
    """Stack horizontally the array ndarray depending on length of
    ndarray_ref.
    """

    return ndarray.reshape((-1,1)) * np.ones(ndarray.shape + ndarray_ref.shape)


def array_equal_map(array_ref: np.ndarray, array_new: np.ndarray
                    ) -> np.ndarray:
    """Return an indices map for the second array where the first
    provided array is equal to the second provided array. Process the
    arrays row wise (along first dimension). Assume that both arrays
    contains unique elements.
    """
    map: np.ndarray = np.array([], dtype=np.int)
    for i in range(array_ref.shape[0]):
        map = np.append(map, np.argwhere(array_new==array_ref[i]))

    return map


def array_equal_duo_map(array_ref: np.ndarray, array_new: np.ndarray
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """Return an indices map for the first and second array where the
    first provided array is equal to the second provided array.
    Process the arrays row wise (along first dimension). Assume that
    both arrays contains unique elements.
    """
    map: np.ndarray = np.array([], dtype=np.int)
    map_: np.ndarray = np.array([], dtype=np.int)
    for i in range(array_ref.shape[0]):
        if (np.argwhere(array_new==array_ref[i]).size):
            map = np.append(map, np.argwhere(array_ref==array_ref[i]))
            map_ = np.append(map_, np.argwhere(array_new==array_ref[i]))

    return map, map_


def crop_array_from_ranges(x_data: np.ndarray, y_data: np.ndarray,
                           z_data: np.ndarray,
                           x_range: Optional[Tuple[float, float]],
                           y_range: Optional[Tuple[float, float]],
                           z_range: Optional[Tuple[float, float]],
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Helper for 3d graph data, crop the x, y, z data depending on the
    provided ranges.
    """
    if (x_range is not None):
        x_to_keep = np.where((x_data>x_range[0]) & (x_data<x_range[1]))
        x_data = x_data[x_to_keep[0]]
        z_data = z_data[:,:,x_to_keep[0]]
    if (y_range is not None):
        y_to_keep = np.where((y_data>y_range[0]) & (y_data<y_range[1]))
        y_data = y_data[y_to_keep[0]]
        z_data = z_data[:,y_to_keep[0]]
    if (z_range is not None):
        z_data = z_data[np.where((z_data>z_range[0]) & z_data<z_range[1])]

    return x_data, y_data, z_data
