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

from typing import Callable, List, Optional, overload, Union

import numpy as np

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.effects.abstract_effect_taylor import AbstractEffectTaylor


class Dispersion(AbstractEffectTaylor):
    r"""The dispersion effect.

    Attributes
    ----------
    omega : numpy.ndarray of float
        The angular frequency array. :math:`[ps^{-1}]`
    time : numpy.ndarray of float
        The time array. :math:`[ps]`
    domega : float
        The angular frequency step. :math:`[ps^{-1}]`
    dtime : float
        The time step. :math:`[ps]`
    order_taylor :
        The order of coeff coefficients Taylor series expansion to
        take into account.

    """

    def __init__(self, beta: Union[List[float], Callable],
                 order_taylor: int = 2, start_taylor: int = 0,
                 skip_taylor: List[int] = [], UNI_OMEGA: bool = True) -> None:
        r"""
        Parameters
        ----------
        beta :
            The derivatives of the propagation constant.
            :math:`[km^{-1}, ps\cdot km^{-1}, ps^2\cdot km^{-1},
            ps^3\cdot km^{-1}, \ldots]` If a callable is provided,
            variables must be (:math:`\omega`, order) where
            :math:`\omega` angular frequency. :math:`[ps^{-1}]`
        order_taylor :
            The order of beta coefficients Taylor series expansion to
            take into account. (will be set to the length of the beta
            array if one is provided)
        start_taylor :
            The order of the derivative from which to start the Taylor
            series expansion.
        skip_taylor :
            The order_taylors of the derivative to not consider.
        UNI_OMEGA :
            If True, consider only the center omega for computation.
            Otherwise, considered omega discretization.

        """
        super().__init__(beta, order_taylor, start_taylor, skip_taylor,
                         UNI_OMEGA)
    # ==================================================================
    def op(self, waves: np.ndarray, id: int,
           corr_wave: Optional[np.ndarray] = None) -> np.ndarray:
        """The operator of the dispersion effect."""

        return 1j * self._op[id]
