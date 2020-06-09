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
