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

from typing import List, Optional, Tuple

import numpy as np
from nptyping import Array

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.domain import Domain
from optcom.effects.abstract_effect import AbstractEffect
from optcom.field import Field
from optcom.utils.fft import FFT


class AbstractEquation(object):

    def __init__(self):
        self._effects_lin: List[AbstractEffect] = []
        self._effects_non_lin: List[AbstractEffect] = []
        self._effects_all: List[AbstractEffect] = []
        self._op_type: str = ""
        self._center_omega: Array[float]
        self._omega: Array[float]
    # ==================================================================
    def __call__(self, vectors: Array, h: float, z: float) -> Array: ...
    # ==================================================================
    def open(self, domain: Domain, *fields: List[Field]) -> None:
        """This function is called once before a Stepper began the
        computation. Pass the time, wavelength and center wavelength
        to all the effects in the equation.
        """

        self._init_eq_ids(list(fields))

        self._center_omega = np.array([])
        for field_list in fields:
            for field in field_list:
                self._center_omega = np.hstack((self._center_omega,
                                                field.center_omega))

        self._omega = FFT.fftshift(domain.omega)

        effects = ["lin", "non_lin", "all"]
        for effect in effects:
            effect_list = getattr(self, "_effects_{}".format(effect))
            for i in range(len(effect_list)):
                effect_list[i].omega = self._omega
                effect_list[i].time = domain.time
                effect_list[i].center_omega = self._center_omega
    # ==================================================================
    def close(self, domain: Domain, *fields: List[Field]) -> None:
        """This function is called once after a Stepper ended the
        computation.
        """
        # Reset main array for further pass
        self._center_omega = np.array([])
        self._omega = np.array([])

        return None
    # ==================================================================
    def set(self, waves: Array[cst.NPFT], h: float, x: float) -> None:
        """This function is called before each step of the computation.
        """

        return None
    # ==================================================================
    def update(self, waves: Array[cst.NPFT], h: float, x: float) -> None:
        """This function is called after each step of the computation.
        """

        return None
    # ==================================================================
    # Fields and waves management for coupled equations ================
    # ==================================================================
    def _init_eq_ids(self, fields_per_eq: List[List[Field]]) -> None:
        """Get positions of the waves according to their related
        equation. Each element of eq_ids contains the last index of the
        wave contained in the equation, which id is the index in
        eq_ids, considering that all waves of all fields form one
        array.
        """
        self._eq_ids: List[int] = []
        ind = -1
        for fields in fields_per_eq:
            for field in fields:
                ind += len(field)   # Number of waves in field
            self._eq_ids.append(ind)

        return None
    # ==================================================================
    def _eq_id(self, wave_id: int) -> Optional[int]:
        """Return the equation id to which wave_id belongs."""
        for i, ind in enumerate(self._eq_ids):
            if (wave_id <= ind):

                return i

        return None
    # ==================================================================
    def _rel_wave_id(self, wave_id: int) -> int:
        """Return the wave id if there were only waves from the equation
        to which wave_id belongs to."""
        eq_id = self._eq_id(wave_id)
        if (eq_id): # Not first eq

            return wave_id - (self._eq_ids[eq_id-1] + 1)
        else:

            return wave_id
    # ==================================================================
    def _len_eq(self, eq_id: int) -> int:
        """Return the number of waves in the equation with id eq_id."""
        if (not eq_id):  # First eq

            return self._eq_ids[eq_id] + 1
        else:

            return (self._eq_ids[eq_id]
                    - self._eq_ids[eq_id-1])
    # ==================================================================
    def _eq_wave_ids(self, eq_id: int) -> Tuple[int, int]:
        """Return the start and end wave_ids in the equation with id
        eq_id.
        """
        if (not eq_id):  # First eq
            start = 0    # Beginning of the list
        else:
            start = self._eq_ids[eq_id-1] + 1
        end = self._eq_ids[eq_id]

        return start, end
    # ==================================================================
    def _in_eq_waves(self, waves: Array, eq_id: int) -> Array:
        """Return all waves from waves in the equation with id eq_id."""
        start, end = self._eq_wave_ids(eq_id)

        return waves[start:end+1]
    # ==================================================================
    def _out_eq_waves(self, waves: Array, eq_id: int) -> Array:
        """Return all waves from waves out of the equation with id
        eq_id."""
        start, end = self._eq_wave_ids(eq_id)

        return waves[0:start] + waves[end+1:len(waves)+1]
    # ==================================================================
    # Effects variables management =====================================
    # ==================================================================
    def _clear_var_lin(self):
        return None
    def _clear_var_non_lin(self):
        return None
    def _clear_var_all(self):
        return None
    # ==================================================================
    @staticmethod
    def exp(wave: Array[cst.NPFT], h: float) -> Array[cst.NPFT]:
        """Exponential function.

        Parameters
        ----------
        wave :
            The wave vector
        h :
            The spatial step

        Returns
        -------
        :
            Exponential of the wave multiplied by the spatial step

        Notes
        -----

        .. math:: \exp(h\hat{D})

        """
        return np.exp(h*wave, dtype=cst.NPFT)
    # ==================================================================
    # Operators ========================================================
    # ==================================================================
    def _call(self, op_or_term: str, effect_type: str, waves: Array[cst.NPFT],
              id: int, corr_wave: Array[cst.NPFT]) -> Array[cst.NPFT]:
        """Parse the operator/term (op_or_term) of the specified effects
        (effect_type) with specified operator type (self._op_type).
        """
        res = np.zeros(waves[id].shape, dtype=cst.NPFT)
        for effect in getattr(self, "_effects_{}".format(effect_type)):
            res += getattr(effect, "{}{}".format(op_or_term, self._op_type)
                           )(waves, id, corr_wave)
        getattr(self, "_clear_var_{}".format(effect_type))

        return res
    # ==================================================================
    def op_lin(self, waves: Array[cst.NPFT], id: int,
               corr_wave: Optional[Array[cst.NPFT]] = None) -> Array[cst.NPFT]:
        """Linear operator of the equation."""

        return self._call("op", "lin", waves, id, corr_wave)
    # ==================================================================
    def op_non_lin(self, waves: Array[cst.NPFT], id: int,
                   corr_wave: Optional[Array[cst.NPFT]] = None
                   ) -> Array[cst.NPFT]:
        """Non linear operator of the equation."""

        return self._call("op", "non_lin", waves, id, corr_wave)
    # ==================================================================
    def op_all(self, waves: Array[cst.NPFT], id: int,
               corr_wave: Optional[Array[cst.NPFT]] = None
               ) -> Array[cst.NPFT]:
        """General operator of the equation."""

        return self._call("op", "all", waves, id, corr_wave)
    # ==================================================================
    def exp_op_lin(self, waves: Array[cst.NPFT], id: int, h: float,
                   corr_wave: Optional[Array[cst.NPFT]] = None
                   ) -> Array[cst.NPFT]:

        return self.exp(self.op_lin(waves, id, corr_wave), h)
    # ==================================================================
    def exp_op_non_lin(self, waves: Array[cst.NPFT], id: int, h: float,
                       corr_wave: Optional[Array[cst.NPFT]] = None
                       ) -> Array[cst.NPFT]:

        return self.exp(self.op_non_lin(waves, id, corr_wave), h)
    # ==================================================================
    def exp_op_all(self, waves: Array[cst.NPFT], id: int, h: float,
                   corr_wave: Optional[Array[cst.NPFT]] = None
                   ) -> Array[cst.NPFT]:

        return self.exp(self.op_all(waves, id, corr_wave), h)
    # ==================================================================
    # Terms ============================================================
    # ==================================================================
    def term_lin(self, waves: Array[cst.NPFT], id: int,
                 corr_wave: Optional[Array[cst.NPFT]] = None
                 ) -> Array[cst.NPFT]:
        """Linear term of the equation."""

        return self._call("term", "lin", waves, id, corr_wave)
    # ==================================================================
    def term_non_lin(self, waves: Array[cst.NPFT], id: int,
                     corr_wave: Optional[Array[cst.NPFT]] = None
                     ) -> Array[cst.NPFT]:
        """Non linear term of the equation."""
        if (corr_wave is None):
            corr_wave = waves[id]

        return self._call("term", "non_lin", waves, id, corr_wave)
    # ==================================================================
    def term_all(self, waves: Array[cst.NPFT], id: int,
                 corr_wave: Optional[Array[cst.NPFT]] = None
                 ) -> Array[cst.NPFT]:
        """General term of the equation."""

        return self._call("term", "all", waves, id, corr_wave)
    # ==================================================================
    def exp_term_lin(self, waves: Array[cst.NPFT], id: int, h: float,
                     corr_wave: Optional[Array[cst.NPFT]] = None
                     ) -> Array[cst.NPFT]:
        if (corr_wave is None):
            corr_wave = waves[id]

        return FFT.ifft(self.exp_op_lin(waves, id, h, corr_wave)
                        * FFT.fft(corr_wave))
    # ==================================================================
    def exp_term_non_lin(self, waves: Array[cst.NPFT], id: int, h: float,
                         corr_wave: Optional[Array[cst.NPFT]] = None
                         ) -> Array[cst.NPFT]:
        if (corr_wave is None):
            corr_wave = waves[id]

        return  self.exp_op_non_lin(waves, id, h, corr_wave) * corr_wave
    # ==================================================================
    def exp_term_all(self, waves: Array[cst.NPFT], id: int, h: float,
                     corr_wave: Optional[Array[cst.NPFT]] = None
                     ) -> Array[cst.NPFT]:

        return np.zeros(waves[id].shape, dtype=cst.NPFT)
    # ==================================================================
