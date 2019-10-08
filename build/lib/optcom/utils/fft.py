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

from typing import Optional

import numpy as np
from nptyping import Array

import optcom.utils.constants as cst

# Typehints variables
NpArray = Array[cst.NPFT]


class FFT(object):

    _fft_counter = 0

    def __init__(self):

        return None
    # ==================================================================
    @classmethod
    def inc_fft_counter(cls):

        cls._fft_counter += 1
    # ==================================================================
    # numpy.fft methods binding ----------------------------------------
    @staticmethod
    def fft(A: NpArray, count: bool = True) -> NpArray:

        if (count):
            FFT.inc_fft_counter()

        return np.fft.ifft(A)
    # ==================================================================
    @staticmethod
    def ifft(A: NpArray, count: bool = True) -> NpArray:

        if (count):
            FFT.inc_fft_counter()

        return np.fft.fft(A)
    # ==================================================================
    @staticmethod
    def fftshift(A: NpArray) -> NpArray:

        return np.fft.ifftshift(A)
    # ==================================================================
    @staticmethod
    def ifftshift(A: NpArray) -> NpArray:

        return np.fft.fftshift(A)
    # ==================================================================
    # fourier transform properties -------------------------------------
    @staticmethod
    def dt_fft(A: NpArray, omega: NpArray, order: int,
               fft_A: Optional[NpArray] = None) -> NpArray:

        if (fft_A is None):
            fft_A = FFT.fft(A)

        return np.power(-1j*omega, order) * fft_A
    # ==================================================================
    @staticmethod
    def dt_to_fft(A: NpArray, omega: NpArray, order: int,
                  fft_A: Optional[NpArray] = None) -> NpArray:

        return FFT.ifft(FFT.dt_fft(A, omega, order, fft_A))
    # ==================================================================
    @staticmethod
    def conv_fft(A: NpArray, B: NpArray, fft_A: Optional[NpArray] = None,
                 fft_B: Optional[NpArray] = None) -> NpArray:

        if (fft_A is None):
            fft_A = FFT.fft(A)
        if (fft_B is None):
            fft_B = FFT.fft(B)

        return (fft_A * fft_B)
    # ==================================================================
    @staticmethod
    def conv_to_fft(A: NpArray, B: NpArray, fft_A: Optional[NpArray] = None,
                    fft_B: Optional[NpArray] = None) -> NpArray:

        return FFT.ifft(FFT.conv_fft(A, B, fft_A, fft_B))
