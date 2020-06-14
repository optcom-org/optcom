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

from typing import Callable, Optional, overload

import numpy as np

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.field import Field
from optcom.equations.abstract_equation import AbstractEquation
from optcom.parameters.fiber.doped_fiber_gain import DopedFiberGain


class ASENoise(AbstractEquation):
    r"""Calculate the noise propagation in fiber amplifier.


    Notes
    -----

    .. math:: \begin{split}
                 \frac{\partial P_{ase}^{\pm}(z)}{\partial z} =
                 &\pm \Big[\Gamma_{s}(\omega)\big[\sigma_{e}(\omega)
                 N_i(z) - \sigma_{a}(\omega)N_j(z)\big]
                 - \eta(\omega)\Big] P_{ase}^\pm (z)
                 & \pm \Gamma_{s}(\omega)\sigma_{e}(\omega) N_i(z)
                 P_{0}^\pm(z)
               \end{split}

    """

    def __init__(self, se_power: Callable, gain_coeff: DopedFiberGain,
                 absorp_coeff: DopedFiberGain, noise_omega: np.ndarray
                 ) -> None:
        r"""
        Parameters
        ----------
        se_power :
            The spontaneous emission power. :math:`[W]`
        gain_coeff : optcom.parameters.fiber.doped_fiber_gain.DopedFiberGain
            The gain coefficient of the noise.
        absorp_coeff : optcom.parameters.fiber.doped_fiber_gain.DopedFiberGain
            The absorption coefficient of the noise.
        noise_omega :
            The angular frequencies composing the noise array.
            :math:`[ps^{-1}]`

        """
        self._noise_omega: np.ndarray = noise_omega
        self._se_power: np.ndarray = se_power(noise_omega)
        self._gain_coeff: DopedFiberGain = gain_coeff
        self._absorp_coeff: DopedFiberGain = absorp_coeff
    # ==================================================================
    @overload
    def __call__(self, noises: np.ndarray, z: float, h: float
                 )-> np.ndarray: ...
    # ------------------------------------------------------------------
    @overload
    def __call__(self, noises: np.ndarray, z: float, h: float, ind: int
                 ) -> np.ndarray: ...
    # ------------------------------------------------------------------
    def __call__(self, *args):
        if (len(args) == 4):
            noises, z, h, ind = args
            arg = np.zeros_like(self._noise_omega)
            gain_ase = np.zeros_like(self._noise_omega)
            absorp_ase = np.zeros_like(self._noise_omega)
            gain_ase = self._gain_coeff(self._noise_omega)
            absorp_ase = self._absorp_coeff(self._noise_omega)
            arg = gain_ase - absorp_ase

            return (noises[ind]*arg) + (self._se_power*gain_ase)
        else:

            raise NotImplementedError()
