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

from __future__ import annotations

import copy
from typing import Any, List

import optcom.utils.constants as cst
import optcom.utils.utilities as util
#from optcom.components.abstract_component import AbstractComponent
from optcom.field import Field


TypeComp = Any


class Port(object):
    """Represent a port of a component.

    Attributes
    ----------
    comp : TypeComp

    port : int

    ngbr_comp : TypeComp

    ngbr_port : int

    type : int

    """

    def __init__(self, comp: TypeComp, port: int, type: int,
                 ngbr_comp: Optional[TypeComp] = None,
                 ngbr_port: Optional[int] = None) -> None:

        self._comp: TypeComp = comp
        self._port: int = port
        self._type: int = type
        self._ngbr_comp: Optional[TypeComp] = ngbr_comp
        self._ngbr_port: Optional[int] = ngbr_port
        self._fields: List[Field] = []
        self._unidir: bool = True
    # ==================================================================
    def __str__(self) -> str:
        if (self._ngbr_comp is None):
            util.print_terminal("Port {} of component '{}' is free."
                  .format(self._port, self._comp.name), '')
        else:
            if (self._unidir):
                util.print_terminal("Port {} of component '{}' has an "
                    "unidirectionnal link from component '{}'."
                    .format(self._port, self._comp.name, self._ngbr_comp.name),
                    '')
            else:
                util.print_terminal("Port {} of component '{}' is linked "
                    "to port {} of component '{}'."
                    .format(self._port, self._comp.name, self._ngbr_port,
                            self._ngbr_comp.name), '')

        return str()
    # ==================================================================
    def __len__(self) -> int:

        return len(self._fields)
    # ==================================================================
    def __getitem__(self, key: int) -> Field:

        return self._fields[key]
    # ==================================================================
    @property
    def fields(self) -> List[Field]:

        return self._fields
    # ==================================================================
    @property
    def comp(self) -> TypeComp:

        return self._comp
    # ------------------------------------------------------------------
    @comp.setter
    def comp(self, comp: TypeComp) -> None:

        self._comp = comp
    # ==================================================================
    @property
    def port(self) -> int:

        return self._port
    # ------------------------------------------------------------------
    @port.setter
    def port(self, port: int) -> None:

        self._port = port
    # ==================================================================
    @property
    def ngbr_comp(self) -> Optional[TypeComp]:

        return self._ngbr_comp
    # ------------------------------------------------------------------
    @ngbr_comp.setter
    def ngbr_comp(self, ngbr_comp: Optional[TypeComp]) -> None:

        self._ngbr_comp = ngbr_comp
    # ==================================================================
    @property
    def ngbr_port(self) -> int:

        return self._ngbr_port
    # ------------------------------------------------------------------
    @ngbr_port.setter
    def ngbr_port(self, ngbr_port: int) -> None:

        self._ngbr_port = ngbr_port
    # ==================================================================
    @property
    def type(self) -> int:

        return self._type
    # ------------------------------------------------------------------
    @type.setter
    def type(self, type: int) -> None:

        self._type = type
    # ==================================================================
    def reset(self) -> None:
        self._ngbr_comp = None
        self._ngbr_port = None
        self._unidir = True
    # ==================================================================
    def link_to(self, port: Port, unidir: bool = False) -> None:
        self._ngbr_comp = port.comp
        self._ngbr_port = port.port
        self._unidir = True if unidir else False
    # ==================================================================
    def link_unidir_to(self, port: Port) -> None:
        self.link_to(port, True)
    # ==================================================================
    def is_linked_to(self, port: Port) -> bool:

        return (self._ngbr_comp == port.comp and self._ngbr_port == port.port)
    # ==================================================================
    def is_linked_unidir_to(self, port: Port) -> bool:

        return self.is_linked_to(port) and self._unidir
    # ==================================================================
    def save_field(self, field: Field) -> None:

        self._fields.append(copy.deepcopy(field))
    # ==================================================================
    def is_free(self):

        return self._ngbr_comp is None
    # ==================================================================
    def is_type_in(self) -> bool:

        return self._type in cst.IN_PORTS
    # ==================================================================
    def is_type_out(self) -> bool:

        return self._type in cst.OUT_PORTS
    # ==================================================================
    def is_type_valid(self, field_type: int) -> bool:

        return self._type in cst.FIELD_TO_PORT.get(field_type)
    # ==================================================================
    def is_unidir(self) -> bool:

        return self._unidir
