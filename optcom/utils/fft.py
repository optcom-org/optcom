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
import pyfftw

import optcom.utils.constants as cst


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
    def fft(A: np.ndarray, count: bool = True) -> np.ndarray:

        if (count):
            FFT.inc_fft_counter()

        return pyfftw.interfaces.numpy_fft.ifft(A)
    # ==================================================================
    @staticmethod
    def ifft(A: np.ndarray, count: bool = True) -> np.ndarray:

        if (count):
            FFT.inc_fft_counter()

        return pyfftw.interfaces.numpy_fft.fft(A)
    # ==================================================================
    @staticmethod
    def fftshift(A: np.ndarray) -> np.ndarray:
        if (A.ndim < 2):

            return np.fft.ifftshift(A)
        else:
            res = np.zeros_like(A)
            for i in range(len(A)):
                res[i] = np.fft.ifftshift(A[i])

            return res
    # ==================================================================
    @staticmethod
    def ifftshift(A: np.ndarray) -> np.ndarray:
        if (A.ndim < 2):

            return np.fft.fftshift(A)
        else:
            res = np.zeros_like(A)
            for i in range(len(A)):
                res[i] = np.fft.fftshift(A[i])

            return res
    # ==================================================================
    # fourier transform error management -------------------------------
    @staticmethod
    def fft_mult_ifft(A: np.ndarray, B: np.ndarray) -> np.ndarray:

        if (np.sum(B) == B.size):   # B is identity vectors

            return A
        else:

            return FFT.fft(B * FFT.ifft(A))
    # ==================================================================
    @staticmethod
    def ifft_mult_fft(A: np.ndarray, B: np.ndarray) -> np.ndarray:

        if (np.sum(B) == B.size):   # B is identity vectors

            return A
        else:

            return FFT.ifft(B * FFT.fft(A))
    # ==================================================================
    # fourier transform properties -------------------------------------
    @staticmethod
    def dt_fft(A: np.ndarray, omega: np.ndarray, order: int,
               fft_A: Optional[np.ndarray] = None) -> np.ndarray:

        if (fft_A is None):
            fft_A = FFT.fft(A)

        return np.power(-1j*omega, order) * fft_A
    # ==================================================================
    @staticmethod
    def dt_to_fft(A: np.ndarray, omega: np.ndarray, order: int,
                  fft_A: Optional[np.ndarray] = None) -> np.ndarray:

        return FFT.ifft(FFT.dt_fft(A, omega, order, fft_A))
    # ==================================================================
    @staticmethod
    def conv_fft(A: np.ndarray, B: np.ndarray,
                 fft_A: Optional[np.ndarray] = None,
                 fft_B: Optional[np.ndarray] = None) -> np.ndarray:

        if (fft_A is None):
            fft_A = FFT.fft(A)
        if (fft_B is None):
            fft_B = FFT.fft(B)

        return (fft_A * fft_B)
    # ==================================================================
    @staticmethod
    def conv_to_fft(A: np.ndarray, B: np.ndarray,
                    fft_A: Optional[np.ndarray] = None,
                    fft_B: Optional[np.ndarray] = None) -> np.ndarray:

        return FFT.ifft(FFT.conv_fft(A, B, fft_A, fft_B))
