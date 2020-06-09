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

Comp = AbstractComponent

# Exceptions
class WaitingError(ConstraintError):
    pass

class WaitingWarning(ConstraintWarning):
    pass


class ConstraintWaiting(AbstractConstraint):
    """Need to wait for fields complying with waiting policy."""

    def __init__(self, ):
        self._stack: Dict[Comp, Tuple[List[int], List[Field]]] = {}
        self._stack_policy: Dict[Comp, List[int]] = {}
    # ==================================================================
    def is_respected(self, comp: AbstractComponent, comp_port_id: int,
                     ngbr: AbstractComponent, ngbr_port_id: int, field: Field
                     ) -> bool:
        flag = True
        wait_policy: List[List[int]] = ngbr.get_wait_policy(ngbr_port_id)
        if (wait_policy):
            if (self._stack.get(ngbr) is None):
                self._stack[ngbr] = ([ngbr_port_id], [field])
            else:
                self._stack[ngbr][0].append(ngbr_port_id)
                self._stack[ngbr][1].append(field)
            # If more than one policy matches, take the first one
            flag = False
            i = 0
            while (i < len(wait_policy) and not flag):
                port_count: List[int] = [self._stack[ngbr][0].count(wait_port)
                                         for wait_port in wait_policy[i]]
                if (0 not in port_count):
                    flag = True
                    self._stack_policy[ngbr] = wait_policy[i]
                i += 1
            if (not flag):
                util.print_terminal("Signal is waiting for fields "
                    "arriving at other ports.", '')

        return flag
    # ==================================================================
    def get_compliance(self, comp: AbstractComponent, comp_port_id: int,
                       ngbr: AbstractComponent, ngbr_port_id: int, field: Field
                       ) -> Tuple[List[int], List[Field]]:
        ports: List[int] = []
        fields: List[Field] = []
        if (self._stack.get(ngbr) is not None):
            # The last field should be the current one
            if (field is not self._stack[ngbr][1][-1]):

                raise WaitingError("The last field of coprop stack should "
                    "be the current field.")
            wait_policy = self._stack_policy[ngbr]
            # Ensuring consistency
            if (wait_policy is None):

                raise WaitingError("wait_policy shouldn't be None if "
                    "self._stack.get(ngbr) is not None")
            ports_ = []
            fields_ = []
            for i, elem in enumerate(self._stack[ngbr][0][:-1]):
                if (elem in wait_policy):
                    ports.append(elem)
                    fields.append(self._stack[ngbr][1][i])
                else:
                    ports_.append(elem)
                    fields_.append(self._stack[ngbr][1][i])
            self._stack[ngbr] = (ports_, fields_)
            # Clear variables
            if (not self._stack[ngbr][0]):
                self._stack.pop(ngbr)
            self._stack_policy.pop(ngbr)

        return ports, fields
    # ==================================================================
    def reset(self) -> None:
        """Reset parameters of the constraint."""
        self._stack = {}
        self._stack_policy = {}
