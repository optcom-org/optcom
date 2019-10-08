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

from typing import List, Optional

import numpy as np
from nptyping import Array

import optcom.utils.constants as cst


class AbstractEffect(object):
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

    def __init__(self, omega: Optional[Array[float]] = None,
                 time: Optional[Array[float]] = None,
                 center_omega: Optional[Array[float]] = None) -> None:

        self._omega: Optional[Array[float]] = omega
        self._time: Optional[Array[float]] = time
        self._center_omega: Optional[Array[float]] = center_omega
    # ==================================================================
    @property
    def omega(self) -> Array[float]:

        return self._omega
    # ==================================================================
    @omega.setter
    def omega(self, omega: Optional[Array[float]]) -> None:
        self._omega = omega
    # ==================================================================
    @property
    def time(self) -> Array[float]:

        return self._time
    # ==================================================================
    @time.setter
    def time(self, time: Optional[Array[float]]) -> None:
        self._time = time
    # ==================================================================
    @property
    def center_omega(self) -> Optional[Array[float]]:

        return self._center_omega
    # ==================================================================
    @center_omega.setter
    def center_omega(self, center_omega: Array[float]) -> None:
        self._center_omega = center_omega

    # ==================================================================
    def op(self, waves: Array[cst.NPFT], id: int,
           corr_wave: Optional[Array[cst.NPFT]] = None) -> Array[cst.NPFT]:
        """The operator of the effect.

        Parameters
        ----------
        waves :
            The wave packet.
        id :
            The ID of the considered wave in the wave packet.
        corr_wave :
            Corrective wave, optional, used if needed.

        Returns
        -------
        :
            The operator of the effect.

        """

        return np.zeros(waves[id].shape, dtype=cst.NPFT)
    # ==================================================================
    def op_approx(self, waves: Array[cst.NPFT], id: int,
                  corr_wave: Optional[Array[cst.NPFT]] = None
                  ) -> Array[cst.NPFT]:
        """The approximated operator of the effect.

        Parameters
        ----------
        waves :
            The wave packet.
        id :
            The ID of the considered wave in the wave packet.
        corr_wave :
            Corrective wave, optional, used if needed.

        Returns
        -------
        :
            The approximated operator of the effect.

        """

        return self.op(waves, id, corr_wave)
    # ==================================================================
    def term(self, waves: Array[cst.NPFT], id: int,
             corr_wave: Optional[Array[cst.NPFT]] = None) -> Array[cst.NPFT]:
        """The term of the effect.

        Parameters
        ----------
        waves :
            The wave packet.
        id :
            The ID of the considered wave in the wave packet.
        corr_wave :
            Corrective wave, optional, used if needed.

        Returns
        -------
        :
            The term of the effect.

        """
        if (corr_wave is None):
            corr_wave = waves[id]

        return self.op(waves, id, corr_wave) * corr_wave
    # ==================================================================
    def term_approx(self, waves: Array[cst.NPFT], id: int,
                    corr_wave: Optional[Array[cst.NPFT]] = None
                    ) -> Array[cst.NPFT]:
        """The approximated term of the effect.

        Parameters
        ----------
        waves :
            The wave packet.
        id :
            The ID of the considered wave in the wave packet.
        corr_wave :
            Corrective wave, optional, used if needed.

        Returns
        -------
        :
            The approximated term of the effect.

        """
        if (corr_wave is None):
            corr_wave = waves[id]

        return self.op_approx(waves, id, corr_wave) * corr_wave
