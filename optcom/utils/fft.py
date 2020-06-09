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
