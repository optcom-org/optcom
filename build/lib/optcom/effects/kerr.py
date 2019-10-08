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
import optcom.utils.utilities as util
from optcom.effects.abstract_effect import AbstractEffect


class Kerr(AbstractEffect):
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

    def __init__(self, SPM: bool = False, XPM: bool = False, FWM: bool = False,
                 sigma: float = cst.XPM_COEFF) -> None:
        r"""
        Parameters
        ----------
        SPM :
            If True, trigger the self-phase modulation.
        XPM :
            If True, trigger the cross-phase modulation.
        FWM :
            If True, trigger the Four-Wave mixing.
        sigma :
            Positive term multiplying the XPM term.

        """
        super().__init__()
        self._SPM: bool = SPM
        self._XPM: bool = XPM
        self._FWM: bool = FWM
        if (self._FWM):
            util.warning_terminal("FWM effect currently not taken into"
                                   "account.")
        self._sigma: float = sigma
    # ==================================================================
    def op(self, waves: Array[cst.NPFT], id: int,
               corr_wave: Optional[Array[cst.NPFT]] = None) -> Array[cst.NPFT]:
        res = np.zeros(waves[id].shape, dtype=cst.NPFT)
        if (self._SPM):
            res += self.op_spm(waves, id, corr_wave)
        if (self._XPM):
            res += self.op_xpm(waves, id, corr_wave)
        #if (FWM):

        return res
    # ==================================================================
    def op_spm(self, waves: Array[cst.NPFT], id: int,
               corr_wave: Optional[Array[cst.NPFT]] = None) -> Array[cst.NPFT]:
        """The operator of the self-phase modulation effect."""

        return 1j * waves[id] * np.conj(waves[id])
    # ==================================================================
    def op_xpm(self, waves: Array[cst.NPFT], id: int,
               corr_wave: Optional[Array[cst.NPFT]] = None) -> Array[cst.NPFT]:
        """The operator of the cross-phase modulation effect."""
        res = np.zeros(waves[id].shape, dtype=cst.NPFT)
        for i in range(len(waves)):
            if (i != id):
                res += waves[id] * np.conj(waves[id])

        return 1j * self._sigma * res
    # ==================================================================
    def op_fwm(self, waves: Array[cst.NPFT], id: int,
               corr_wave: Optional[Array[cst.NPFT]] = None) -> Array[cst.NPFT]:
        pass
