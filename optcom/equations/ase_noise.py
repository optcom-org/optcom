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

from typing import Callable, Optional

import numpy as np

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.equations.abstract_equation import AbstractEquation
from optcom.parameters.fiber.doped_fiber_gain import DopedFiberGain
from optcom.parameters.fiber.se_power import SEPower
from optcom.field import Field


class ASENoise(AbstractEquation):

    def __init__(self, se_power: Callable,
                 gain_coeff: DopedFiberGain,
                 absorp_coeff: DopedFiberGain,
                 noise_omega: np.ndarray) -> None:
        r"""
        Parameters
        ----------
        se_power :
            The spontaneous emission power. :math:`[W]`
        gain_coeff : DopedFiberGain
            The gain coefficient of the noise.
        absorp_coeff :
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
    def __call__(self, noises: np.ndarray, z: float, h: float):
        arg = np.zeros_like(self._noise_omega)
        gain_ase = np.zeros_like(self._noise_omega)
        absorp_ase = np.zeros_like(self._noise_omega)
        gain_ase = self._gain_coeff(self._noise_omega)
        absorp_ase = self._absorp_coeff(self._noise_omega)
        arg = gain_ase - absorp_ase

        return (noises*arg) + (self._se_power*gain_ase)
