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

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.effects.abstract_effect import AbstractEffect


class Kerr(AbstractEffect):
    r"""The Kerr effect.

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
    SPM :
        If True, trigger the self-phase modulation.
    XPM :
        If True, trigger the cross-phase modulation.
    FWM :
        If True, trigger the Four-Wave mixing.
    sigma :
        Positive term multiplying the XPM term.

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
        self.SPM: bool = SPM
        self.XPM: bool = XPM
        self.FWM: bool = FWM
        if (self.FWM):
            util.warning_terminal("FWM effect currently not taken into"
                                   "account.")
        self.sigma: float = sigma
    # ==================================================================
    def set(self, center_omega: np.ndarray = np.array([]),
            abs_omega: np.ndarray = np.array([])) -> None:

        return None
    # ==================================================================
    def op(self, waves: np.ndarray, id: int,
               corr_wave: Optional[np.ndarray] = None) -> np.ndarray:
        res = np.zeros(waves[id].shape, dtype=cst.NPFT)
        if (self.SPM):
            res += self.op_spm(waves, id, corr_wave)
        if (self.XPM):
            res += self.op_xpm(waves, id, corr_wave)
        #if (FWM):  # only if len(waves) >= 3

        return res
    # ==================================================================
    def op_spm(self, waves: np.ndarray, id: int,
               corr_wave: Optional[np.ndarray] = None) -> np.ndarray:
        """The operator of the self-phase modulation effect."""

        return 1j * waves[id] * np.conj(waves[id])
    # ==================================================================
    def op_xpm(self, waves: np.ndarray, id: int,
               corr_wave: Optional[np.ndarray] = None) -> np.ndarray:
        """The operator of the cross-phase modulation effect."""
        res = np.zeros(waves[id].shape, dtype=cst.NPFT)
        for i in range(len(waves)):
            if (i != id):
                res += waves[i] * np.conj(waves[i])

        return 1j * self.sigma * res
    # ==================================================================
    def op_fwm(self, waves: np.ndarray, id: int,
               corr_wave: Optional[np.ndarray] = None) -> np.ndarray:
        pass
