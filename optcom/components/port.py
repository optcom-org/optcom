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

from __future__ import annotations

import copy
from typing import Any, List, Optional, TYPE_CHECKING

import numpy as np

import optcom.config as cfg
import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.field import Field
# To avoid import cycles -> see
# https://mypy.readthedocs.io/en/stable/common_issues.html#import-cycles
if TYPE_CHECKING:   # only for type check, always false at runtime
    import optcom.components.abstract_component as ab_comp


class Port(object):
    """Represent a port of a component. Components are linked via ports.
    The purpose of Port is not only link management, but can also be
    used to save fields.

    Attributes
    ----------
    comp :
        The component to which the port belongs.
    comp_port_id : int
        The ID of the port for the component `comp`.
    type : int
        The type of the port. For types, see
        :mod:`optcom/utils/constant_values/port_types`.
    ngbr :
        The component to which the port is connected.
    ngbr_port_id : int
        The ID of the port for the component `ngbr_comp`.

    """

    def __init__(self, comp: ab_comp.AbstractComponent, type: int,
                 max_nbr_pass: int = cfg.MAX_NBR_PASS) -> None:
        """
        Parameters
        ----------
        comp :
            The component to which the port belongs.
        type :
            The type of the port. For types, see
            :mod:`optcom/utils/constant_values/port_types`.
        max_nbr_pass :
            The maximum number of pass allowed through this port.

        """
        # Attr ---------------------------------------------------------
        self._comp: ab_comp.AbstractComponent = comp
        self._type: int = type
        self._linked_port: Optional[Port] = None
        self._fields: List[Field] = []  # recorded fields in the port
        self._unidir: bool = True
        self.max_nbr_pass: int = max_nbr_pass
        self._counter_pass: int = 1
        self.reset()
    # ==================================================================
    def __str__(self) -> str:
        """Return a str describing the current state of the port."""
        str_to_return: str = ''
        if (self.ngbr is None):
            str_to_return += ("Port {} of component '{}' is free.\n"
                              .format(self.comp_port_id, self.comp.name))
        else:
            if (self.is_unidir()):
                str_to_return += ("Port {} of component '{}' has an "
                    "unidirectionnal link from component '{}'.\n"
                    .format(self.comp_port_id, self.comp.name, self.ngbr.name))
            else:
                str_to_return += ("Port {} of component '{}' is linked "
                    "to port {} of component '{}'.\n"
                    .format(self.comp_port_id, self.comp.name,
                            self.ngbr_port_id, self.ngbr.name))
        if (self.fields):
            sep_charact: str = '\n    '
            field_names: str = ''
            field_names += sep_charact
            for i in range(len(self.fields)):
                field_names += self.fields[i].name
                if (i != len(self._fields)-1):
                    field_names += sep_charact
            str_to_return += ("The following fields have been saved in "
                "this port: {}\n".format(field_names))

        return str_to_return
    # ==================================================================
    def __len__(self) -> int:

        return len(self._fields)
    # ==================================================================
    def __getitem__(self, key: int) -> Field:

        return self._fields[key]
    # ==================================================================
    # Getters and Setters ==============================================
    # ==================================================================
    @property
    def counter_pass(self) -> int:

        return self._counter_pass
    # ==================================================================
    @property
    def field(self) -> Optional[Field]:
        """Return the last saved field if exists, otherwise None."""
        if (self._fields):

            return self._fields[-1]
        else:

            return None
    # ==================================================================
    @property
    def fields(self) -> List[Field]:

        return self._fields
    # ==================================================================
    @property
    def comp(self) -> ab_comp.AbstractComponent:

        return self._comp
    # ==================================================================
    @property
    def comp_port_id(self) -> int:

        return self.comp.port_id_of(self)
    # ==================================================================
    @property
    def ngbr(self) -> Optional[ab_comp.AbstractComponent]:
        if (self._linked_port is not None):

            return self._linked_port.comp

        return None
    # ==================================================================
    @property
    def ngbr_port_id(self) -> int:
        if (self._linked_port is not None):

            return self._linked_port.comp_port_id

        return cst.NULL_PORT_ID
    # ==================================================================
    @property
    def type(self) -> int:

        return self._type
    # ------------------------------------------------------------------
    @type.setter
    def type(self, type: int) -> None:

        self._type = type
    # ==================================================================
    @property
    def nbr_channels(self) -> int:
        """Return the number of channels by adding all channels of all
        fields.
        """
        res = 0
        for i in range(len(self._fields)):
            res += self._fields[i].nbr_channels

        return res
    # ==================================================================
    @property
    def channels(self) -> List[np.ndarray]:
        """Return a list of all channels arrays of all fields.
        """
        res = []
        for i in range(len(self._fields)):
            res.append(self._fields[i].channels)

        return res
    # ==================================================================
    @property
    def noises(self) -> List[np.ndarray]:
        """Return a list of all noises arrays of all fields.
        """
        res = []
        for i in range(len(self._fields)):
            res.append(self._fields[i].noise)

        return res
    # ==================================================================
    @property
    def time(self) -> np.ndarray:
        """Return a list of all times arrays of all fields.
        """
        res = []
        for i in range(len(self._fields)):
            res.append(self._fields[i].time)

        return res
    # ==================================================================
    @property
    def omega(self) -> List[np.ndarray]:
        """Return a list of all angular frequency arrays of all fields.
        """
        res = []
        for i in range(len(self._fields)):
            res.append(self._fields[i].omega)

        return res
    # ==================================================================
    @property
    def nu(self) -> List[np.ndarray]:
        """Return a list of all frequency arrays of all fields.
        """
        res = []
        for i in range(len(self._fields)):
            res.append(self._fields[i].nu)

        return res
    # ==================================================================
    # Link management ==================================================
    # ==================================================================
    def reset(self) -> None:
        """Reset the port parameters to initial value (disconnect)."""
        self._linked_port = None
        self._unidir = True
        self._counter_pass = 1
    # ==================================================================
    def link_to(self, port: Port, unidir: bool = False) -> None:
        """Link the present port to another port.

        Parameters
        ----------
        port :
            The port to be linked to the present port.
        unidir :
            If True, the connection is only one sided, from the present
            port to the connected port.

        """
        self._linked_port = port
        self._unidir = True if unidir else False
    # ==================================================================
    def link_unidir_to(self, port: Port) -> None:
        """Link the present port to port in an unidirectionnal manner.

        Parameters
        ----------
        port :
            The port to be linked to the present port.

        """
        self.link_to(port, True)
    # ==================================================================
    def is_linked_to(self, port: Port) -> bool:

        return (self._linked_port == port)
    # ==================================================================
    def is_linked_unidir_to(self, port: Port) -> bool:

        return self.is_linked_to(port) and self._unidir
    # ==================================================================
    def save_field(self, field: Field) -> None:
        self._fields.append(copy.deepcopy(field))
    # ==================================================================
    def is_free(self):

        return self._linked_port is None
    # ==================================================================
    def is_type_in(self) -> bool:

        return self._type in cst.IN_PORTS
    # ==================================================================
    def is_type_out(self) -> bool:

        return self._type in cst.OUT_PORTS
    # ==================================================================
    def is_type_valid(self, field_type: str) -> bool:

        valid = False
        allow_types = cst.FIELD_TO_PORT.get(field_type)
        if (allow_types is not None):
            valid = self._type in allow_types

        return valid
    # ==================================================================
    def is_unidir(self) -> bool:

        return self._unidir
    # ==================================================================
    def is_valid_for_propagation(self) -> bool:

        return (not self.is_free()) and (not self.is_unidir())
    # ==================================================================
    def inc_counter_pass(self) -> None:
        self._counter_pass += 1
    # ==================================================================
    def dec_counter_pass(self) -> None:
        self._counter_pass -= 1
    # ==================================================================
    def is_counter_below_max_pass(self) -> bool:

        return (self.counter_pass <= self.max_nbr_pass)
