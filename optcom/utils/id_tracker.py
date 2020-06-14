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

from typing import List, Tuple

import numpy as np

from optcom.field import Field


# Exceptions
class IdTrackerError(Exception):
    pass

class InitError(IdTrackerError):
    pass


class IdTracker(object):
    """Save the position of each field and waves.
    """

    def __init__(self, nbr_eqs, prop_dir) -> None:
        r"""
        Parameters
        ----------
        nbr_eqs :
            The number of equations.
        prop_dir :
            If [i] is True, waves in equation [i] are co-propagating
            relatively to equation [0]. If [i] is False, waves in
            equations [i] are counter-propagating relatively to
            equation [0]. prop_dir[0] must be True. len(prop_dir) must
            be equal to nbr_eqs.

        """
        self._nbr_eqs: int = nbr_eqs
        self._prop_dir: List[bool] = prop_dir
        if (len(self._prop_dir) != self._nbr_eqs):

            raise InitError("The number of equation must be equal to the "
                "length of the provided propagation map.")
        if (self._nbr_eqs and not self._prop_dir[0]):

            raise InitError("The first element of the propagation map must "
                "be True.")
        self._wave_ids: List[int] = []     # Contain index of last wave per eq
        self._field_ids: List[int] = []    # Contain index of last field per eq
    # ==================================================================
    @property
    def nbr_eqs(self) -> int:
        """Return the total number of equations."""

        return self._nbr_eqs
    # ==================================================================
    @property
    def nbr_waves(self) -> int:
        """Return the total number of waves."""

        return (self._wave_ids[-1] + 1)
    # ==================================================================
    @property
    def nbr_fields(self) -> int:
        """Return the total number of fields."""

        return (self._field_ids[-1] + 1)
    # ==================================================================
    def reset(self):
        self._wave_ids = []
        self._field_ids = []
    # ==================================================================
    def initialize(self, fields_per_eq: List[List[Field]]) -> None:
        """Get positions of the waves and fields according to their
        related equation. Each element of wave_ids contains the last
        index of the wave contained in the equation, which id is the
        index in eq_ids, considering that all waves of all fields form
        one array. Each element of field_ids contains the index of the
        last fields from equation which id is the index in eq_ids.
        """
        self.reset()
        if (not (len(fields_per_eq)%self._nbr_eqs)):
            ind_wave = -1
            ind_field = -1
            for fields in fields_per_eq[:self._nbr_eqs]:
                ind_field += len(fields)
                for field in fields:
                    ind_wave += len(field)   # Number of waves in field
                self._wave_ids.append(ind_wave)
                self._field_ids.append(ind_field)
        else:

            raise InitError("The list of fields provided for initialization "
                "does not comply with the initial number of equations.")
    # ==================================================================
    def check_wave_id(self, wave_id: int) -> int:
        """Pre-process the wave id. """

        return (wave_id % (self._wave_ids[-1]+1))
    # ==================================================================
    def check_field_id(self, field_id: int) -> int:
        """Pre-process the field id. """

        return (field_id % (self._field_ids[-1]+1))
    # ==================================================================
    def eq_id_of_wave_id(self, wave_id: int) -> int:
        """Return the equation id to which wave_id belongs."""
        wave_id = self.check_wave_id(wave_id)
        for i, ind in enumerate(self._wave_ids):
            if (wave_id <= ind):

                return i

        return -1
    # ==================================================================
    def eq_id_of_field_id(self, field_id: int) -> int:
        """Return the equation id to which field_id belongs."""
        field_id = self.check_field_id(field_id)
        for i, ind in enumerate(self._field_ids):
            if (field_id <= ind):

                return i

        return -1
    # ==================================================================
    def rel_wave_id(self, wave_id: int) -> int:
        """Return the wave id if there were only waves from the equation
        to which wave_id belongs to.
        """
        wave_id = self.check_wave_id(wave_id)
        eq_id = self.eq_id_of_wave_id(wave_id)
        if (eq_id != -1):
            if (eq_id): # Not first eq

                return wave_id - (self._wave_ids[eq_id-1] + 1)
            else:

                return wave_id

        return -1
    # ==================================================================
    def nbr_waves_in_eq(self, eq_id: int) -> int:
        """Return the number of waves in the equation with id eq_id."""
        if (not eq_id):  # First eq

            return self._wave_ids[eq_id] + 1
        else:

            return (self._wave_ids[eq_id]
                    - self._wave_ids[eq_id-1])
    # ==================================================================
    def nbr_fields_in_eq(self, eq_id: int) -> int:
        """Return the number of fields in the equation with id eq_id."""
        if (not eq_id):  # First eq

            return self._field_ids[eq_id] + 1
        else:

            return (self._field_ids[eq_id]
                    - self._field_ids[eq_id-1])
    # ==================================================================
    def wave_ids_in_eq_id(self, eq_id: int) -> Tuple[int, int]:
        """Return the start and end wave_ids in the equation with id
        eq_id.
        """
        if (not eq_id):  # First eq
            start = 0    # Beginning of the list
        else:
            start = self._wave_ids[eq_id-1] + 1
        end = self._wave_ids[eq_id]

        return start, end
    # ==================================================================
    def field_ids_in_eq_id(self, eq_id: int) -> Tuple[int, int]:
        """Return the start and end field_ids in the equation with id
        eq_id.
        """
        if (not eq_id):  # First eq
            start = 0    # Beginning of the list
        else:
            start = self._field_ids[eq_id-1] + 1
        end = self._field_ids[eq_id]

        return start, end
    # ==================================================================
    def waves_in_eq_id(self, waves: np.ndarray, eq_id: int) -> np.ndarray:
        """Return all waves from waves in the equation with id eq_id."""
        start, end = self.wave_ids_in_eq_id(eq_id)

        return waves[start:end+1]
    # ==================================================================
    def waves_out_eq_id(self, waves: np.ndarray, eq_id: int) -> np.ndarray:
        """Return all waves from waves out of the equation with id
        eq_id.
        """
        start, end = self.wave_ids_in_eq_id(eq_id)

        return np.vstack((waves[0:start], waves[end+1:]))
    # ==================================================================
    def fields_in_eq_id(self, fields: np.ndarray, eq_id: int) -> np.ndarray:
        """Return all fields from fields in the equation with id eq_id.
        """
        start, end = self.field_ids_in_eq_id(eq_id)

        return fields[start:end+1]
    # ==================================================================
    def fields_out_eq_id(self, fields: np.ndarray, eq_id: int) -> np.ndarray:
        """Return all fields from fields out of the equation with id
        eq_id.
        """
        start, end = self.field_ids_in_eq_id(eq_id)

        return np.vstack((fields[0:start], fields[end+1:]))
    # ==================================================================
    def are_wave_ids_co_prop(self, id_1: int, id_2: int) -> bool:
        """Return True if id_1 and id_2 are related to co-propagating
        equations.
        """
        eq_id_1 = self.eq_id_of_wave_id(id_1)
        eq_id_2 = self.eq_id_of_wave_id(id_2)

        return (self._prop_dir[eq_id_1] == self._prop_dir[eq_id_2])
    # ==================================================================
    def are_field_ids_co_prop(self, id_1: int, id_2: int) -> bool:
        """Return True if id_1 and id_2 are related to co-propagating
        equations.
        """
        eq_id_1 = self.eq_id_of_field_id(id_1)
        eq_id_2 = self.eq_id_of_field_id(id_2)

        return (self._prop_dir[eq_id_1] == self._prop_dir[eq_id_2])
