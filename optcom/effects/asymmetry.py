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

from typing import Callable, List, Optional, overload, Union

import numpy as np
from nptyping import Array

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.effects.abstract_effect import AbstractEffect
from optcom.effects.dispersion import Dispersion
from optcom.equations.sellmeier import Sellmeier
from optcom.utils.taylor import Taylor


class Asymmetry(AbstractEffect):
    r"""Generic class for effect object.

    Attributes
    ----------
    omega : numpy.ndarray of float
        The angular frequency array. :math:`[ps^{-1}]`
    time : numpy.ndarray of float
        The time array. :math:`[ps]`
    center_omega : numpy.ndarray of float
        The center angular frequency. :math:`[ps^{-1}]`

    """

    def __init__(self, delta: Optional[float] = None,
                 beta_01: Optional[Union[float, Callable]] = None,
                 beta_02: Optional[Union[float, Callable]] = None,
                 medium: str = 'SiO2') -> None:
        r"""
        Parameters
        ----------
        delta :
            The asymmetry measure coefficient. If a callable is provided,
            variable must be angular frequency. :math:`[ps^{-1}]`
        beta_01 :
            The zeroth order Taylor series dispersion coefficient of
            first waveguide. :math:`[km^{-1}]`
        beta_02 :
            The zeroth order Taylor series dispersion coefficient of
            second waveguide. :math:`[km^{-1}]`
        medium :
            The medium in which the dispersion is considered.

        """
        super().__init__()
        self._medium: str = medium
        self._delta: float
        self._predict: Optional[Callable] = None
        if ((delta is None) and (beta_01 is not None)
                and (beta_02 is not None)):
            if (not callable(beta_01) and not callable(beta_02)):
                self._delta = Asymmetry.calc_delta(beta_01, beta_02)
            #else:
            #    self._predict = lambda om1, om2: Asymmetry.calc_beta(
            #        beta_01(om1, 0)[0], beta_02(om2, 0)[0])
        elif (delta is None):
            beta = lambda omega: Dispersion.calc_beta_coeffs(
                omega, 0, Sellmeier(medium))[0]
            self._predict = lambda om1, om2: Asymmetry.calc_delta(beta(om1),
                                                                  beta(om2))
        else:
            self._delta = delta
    # ==================================================================
    @property
    def delta(self) -> float:

        return self._delta
    # ==================================================================
    @delta.setter
    def delta(self, delta: float) -> None:
        self._delta = delta
    # ==================================================================
    def update(self, center_omega: Optional[Array[float]] = None,
               id: int = 0):
        # Assume second omega as mean of all omega so far.  Find way to
        # charact. the delta if multiple channel in other waveguides.
        if (center_omega is None):
            center_omega = self._center_omega
        if (self._predict is not None):
            mean_center_omega = np.mean(center_omega)
            curr_center_omega = center_omega[id]
            self._delta = self._predict(curr_center_omega, mean_center_omega)
    # ==================================================================
    def op(self, waves: Array[cst.NPFT], id: int,
           corr_wave: Optional[Array[cst.NPFT]] = None) -> Array[cst.NPFT]:
        """The operator of the asymmetry effect."""
        self.update(id=id)

        return 1j * self._delta
    # ==================================================================
    @overload
    @staticmethod
    def calc_delta(beta_01: float, beta_02: float) -> float: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def calc_delta(beta_01: Array[float], beta_02: Array[float]
                   ) -> Array[float]: ...
    # ------------------------------------------------------------------
    @staticmethod
    def calc_delta(beta_01, beta_02):
        r"""Calculate the measure of asymmetry for the parameters given
        for two waveguides. [8]_

        Parameters
        ----------
        beta_01 :
            The first term of the propagation constant of the first
            waveguide. :math:`[km^{-1}]`
        beta_02 :
            The first term of the propagation constant of the second
            waveguide. :math:`[km^{-1}]`

        Returns
        -------
        :
            Value of the asymmetry measure. :math:`[km^{-1}]`

        Notes
        -----

        .. math:: \delta_{a12} = \frac{1}{2} (\beta_{01} - \beta_{02})

        References
        ----------
        .. [8] Govind Agrawal, Chapter 2: Fibers Couplers,
           Applications of Nonlinear Fiber Optics (Second Edition),
           Academic Press, 2008, Page 57.

        """

        return 0.5 * (beta_01 - beta_02)
