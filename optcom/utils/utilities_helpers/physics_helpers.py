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

import math
from typing import overload, Optional

import numpy as np


@overload
def db_to_linear(db: float, ref: Optional[float]) -> float: ...
@overload
def db_to_linear(db: np.ndarray, ref: Optional[float]) -> np.ndarray: ...
@overload
def db_to_linear(db: np.ndarray, ref: Optional[np.ndarray]
                 ) -> np.ndarray: ...

def db_to_linear(db, ref=None):
    r"""Convert from decibel to linear form.

    Parameters
    ----------
    db :
        The decibel values to convert. :math:`[dB]`
    ref :
        A value by which to multiply the result.

    Returns
    -------
    :
        The converted linear form. :math:`[]`

    Notes
    -----

    .. math:: P_{lin} = 10^{\frac{P_{dB}}{10}}*P_0

    """
    if (isinstance(db, float)):
        if (ref is None):
            ref = 1.0

        return (10.0**(0.1*db)) * ref

    else:
        if (ref is None):
            ref = np.ones_like(db)

        return np.power(10.0, 0.1*db) * ref


@overload
def linear_to_db(linear: float, ref: Optional[float]) -> float: ...
@overload
def linear_to_db(linear: np.ndarray, ref: Optional[float]) -> np.ndarray: ...
@overload
def linear_to_db(linear: np.ndarray, ref: Optional[np.ndarray]
                 ) -> np.ndarray: ...
def linear_to_db(linear, ref=None):
    r"""Convert from linear form to decibel.

    Parameters
    ----------
    linear :
        The linear form values to convert.  :math:`[]`
    ref :
        A value by which to divide the provided linear values.

    Returns
    -------
    :
        The converted decibel form.  :math:`[dB]`

    Notes
    -----

    .. math:: P_{db} = 10log_{10}\Big(\frac{P_{lin}}{P_0}\Big)

    """
    if (isinstance(linear, float)):
        if (ref is None):
            ref = 1.0

        return 10.0*math.log10(linear/ref)

    else:
        if (ref is None):
            ref = np.ones_like(linear)
        a = linear/ref
        print(a[a==0])
        return 10.0*np.log10(linear/ref)
