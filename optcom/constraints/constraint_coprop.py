
# This This file is part of Optcom.
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

from typing import Dict, List, Optional, Tuple, Union

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.components.abstract_component import AbstractComponent
from optcom.components.port import Port
from optcom.constraints.abstract_constraint import AbstractConstraint
from optcom.constraints.abstract_constraint import ConstraintError
from optcom.constraints.abstract_constraint import ConstraintWarning
from optcom.field import Field


# Exceptions
class CopropError(ConstraintError):
    pass

class CopropWarning(ConstraintWarning):
    pass


class ConstraintCoprop(AbstractConstraint):
    """Need to wait for other copropagating fields."""

    def __init__(self):

        self._stack: Dict[Port, Tuple[List[Field], List[int]]] = {}
    # ==================================================================
    def update(self, comp: AbstractComponent, port_ids: List[int],
               fields: List[Field]) -> None:
        """Update necessary information for the constraint."""
        for i, port_id in enumerate(port_ids):
            # If > 1, ouptut fields from same port -> need coprop manag.
            if (port_ids.count(port_id) > 1):
                if (self._stack.get(comp[port_id]) is not None):
                    self._stack[comp[port_id]][0].append(fields[i])
                    self._stack[comp[port_id]][1][0] += 1
                else:
                    self._stack[comp[port_id]] = ([fields[i]], [1])
    # ==================================================================
    def is_respected(self, comp: AbstractComponent, comp_port_id: int,
                     ngbr: AbstractComponent, ngbr_port_id: int, field: Field
                     ) -> bool:
        flag = True
        comp_port: Port = comp[comp_port_id]
        if (self._stack.get(comp_port) is not None):
            self._stack[comp_port][1][0] -= 1
            if (self._stack[comp_port][1][0] > 0):
                flag = False
                util.print_terminal("Signal is waiting for copropagating "
                                    "fields.", '')

        return flag
    # ==================================================================
    def get_compliance(self, comp: AbstractComponent, comp_port_id: int,
                       ngbr: AbstractComponent, ngbr_port_id: int, field: Field
                       ) -> Tuple[List[int], List[Field]]:
        ports: List[int] = []
        fields: List[Field] = []
        comp_port: Port = comp[comp_port_id]
        if (self._stack.get(comp_port) is not None):
            # The last field should be the current one
            if (field is not self._stack[comp_port][0][-1]):

                raise CopropError("The last field of coprop stack should "
                    "be the current field.")
            # If last one, can add the other coprop fields to it
            fields = self._stack[comp_port][0][:-1]
            ports = [ngbr_port_id for i in range(len(fields))]
            self._stack.pop(comp_port)

        return ports, fields
    # ==================================================================
    def reset(self) -> None:
        """Reset parameters of the constraint."""
        self._stack = {}
