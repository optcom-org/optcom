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
from typing import Any, Dict, List, Optional, overload, Tuple, Union

import numpy as np
from nptyping import Array

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.components.port import Port
from optcom.domain import Domain
from optcom.field import Field
from optcom.storage import Storage


default_name = 'AbstractComponent'


class AbstractComponent(object):
    """Parent of any component object. Represent a node of the layout
    graph.

    Attributes
    ----------
        name : str
            The name of the component.
        ports_type : list of int
            Type of each port of the component, give also the number of
            ports in the component. For types, see
            :mod:`optcom/utils/constant_values/port_types`.
        save : bool
            If True, will save each field going through each port. The
            recorded fields can be accessed with the attribute
            :attr:`fields`.

    """

    _nbr_instances: int = 0
    _nbr_instances_with_default_name: int = 0

    def __init__(self, name: str, default_name: str, ports_type: List[int],
                 save: bool, wait: bool = False,
                 max_nbr_pass: Optional[List[int]] = None) -> None:
        """
        Parameters
        ----------
        name :
            The name of the component.
        default_name :
            The default name of the component.
        ports_type :
            Type of each port of the component, give also the number of
            ports in the component. For types, see
            :mod:`optcom/utils/constant_values/port_types`.
        save :
            If True, will save each field going through each port. The
            recorded fields can be accessed with the attribute
            :attr:`fields`.
        wait :
            If True, will wait for specified waiting port policy added
            with the function :func:`AbstractComponent.add_wait_policy`.
        max_nbr_pass :
            If not None, no fields will be propagated if the number of
            fields which passed through a specific port exceed the
            specified maximum number of pass for this port.

        """
        # Attr type check ----------------------------------------------
        util.check_attr_type(name, 'name', str)
        util.check_attr_type(default_name, 'default_name', str)
        util.check_attr_type(ports_type, 'ports_type', list)
        util.check_attr_type(save, 'save', bool)
        util.check_attr_type(wait, 'wait', bool)
        util.check_attr_type(max_nbr_pass, 'max_nbr_pass', None, list)
        # Nbr of instances and default name management -----------------
        self.inc_nbr_instances()
        self.name: str = name
        if (name == default_name):
            if (self._nbr_instances_with_default_name):
                self.name += ' ' + str(self._nbr_instances_with_default_name)
            self.inc_nbr_instances_with_default_name()
        # Others var ---------------------------------------------------
        self._nbr_ports: int = len(ports_type)
        self.save: bool = save
        self._storages: Storage = []
        self._ports: Port = []
        for i in range(self._nbr_ports):
            self._ports.append(Port(self, i, ports_type[i]))
        self._port_policy: Dict[Tuple[int,...], Tuple[int,...]] = {}
        self._wait_policy: List[List[int]] = []
        self._wait: bool = wait
        self._counter_pass: List[int] = [0 for i in range(self._nbr_ports)]
        self._max_nbr_pass: List[int]
        if (max_nbr_pass is None):
            # 999 bcs max default recursion depth with python
            self._max_nbr_pass = [999 for i in range(self._nbr_ports)]
        else:
            self._max_nbr_pass = util.make_list(max_nbr_pass, self._nbr_ports)
    # ==================================================================
    def __str__(self) -> str:

        util.print_terminal("State of component '{}':".format(self.name))
        for i in range(self._nbr_ports):
            print(self._ports[i])

        return str()
    # ==================================================================
    def __len__(self) -> int:

        return self._nbr_ports
    # ==================================================================
    def __del__(self) -> None:

        self.dec_nbr_instances()
        self.dec_nbr_instances_with_default_name()
    # ==================================================================
    def __getitem__(self, key: int) -> Tuple[AbstractComponent, int]:

        return self._ports[key]
    # ==================================================================
    @property
    def storages(self):

        return self._storages
    # ==================================================================
    # Counter management ===============================================
    # ==================================================================
    @classmethod
    def inc_nbr_instances(cls):

        cls._nbr_instances += 1
    # ==================================================================
    @classmethod
    def dec_nbr_instances(cls):

        cls._nbr_instances -= 1
    # ==================================================================
    @classmethod
    def inc_nbr_instances_with_default_name(cls):

        cls._nbr_instances_with_default_name += 1
    # ==================================================================
    @classmethod
    def dec_nbr_instances_with_default_name(cls):

        cls._nbr_instances_with_default_name -= 1
    # ==================================================================
    # Port management ==================================================
    # ==================================================================
    def is_port_valid(self, port_nbr: int) -> bool:

        if (0 <= port_nbr < self._nbr_ports):

            return True
        else:
            util.warning_terminal("Port {} out of range, component '{}' have "
                "only {} port(s)."
                .format(port_nbr, self.name, self._nbr_ports))

            return False
    # ==================================================================
    def del_port(self, port_nbr: int) -> None:

        if (self.is_port_valid(port_nbr)):
            self._ports[port_nbr].reset()
    # ==================================================================
    def add_port_policy(self, *policy: Tuple[List[int], List[int], bool]
                        ) -> None:
        """Append a new policy to automatically designate output port
        depending on the input port.

        Parameters
        ----------
            policy :
                The policy (a, b, flag) assigns input ports in list a
                to output ports in list b. If flag is True, also assign
                input ports of b to output ports of a. If there is -1
                in list b, the field entering at the corresponding
                port in list a will not be transmitted.
                N.B.: if (len(a) < len(b)), pad b with length of a.
                len(a) must be >= len(b)
        """
        for pol in policy:
            # Entry ports list must be same size as exit ports list
            if (len(pol[0]) >= len(pol[1])):
                pol_in = util.permutations(pol[0])
                pol_out = util.permutations(
                                util.pad_with_last_elem(pol[1], len(pol[0])))
                for i in range(len(pol_in)):
                    self._port_policy[tuple(pol_in[i])] = tuple(pol_out[i])
                    if (pol[2]):
                        self._port_policy[tuple(pol_out[i])] = tuple(pol_in[i])
            else:
                util.warning_terminal("The number of entry ports must be "
                    "equal or greater than the number of exit ports.")
    # ==================================================================
    def output_ports(self, input_ports: List[int]) -> List[int]:
        """Return a list of the corresponding output port(s) to the
        specified input port(s) depending on all ports in the provided
        list.

        Parameters
        ----------
        input_ports :
            The inputs ports.

        Returns
        -------
        :
            The output ports.

        """
        output_ports: List[int] = []
        if (not self._port_policy):
            util.warning_terminal("No policy for port management for "
                "component {}, no fields propagated.".format(self.name))
        else:
            uni_ports = util.unique(input_ports)
            uni_output_ports = self._port_policy.get(tuple(uni_ports))
            if (uni_output_ports is None):
                util.warning_terminal("The input ports {} provided for "
                    "component {} do not match any policy, no fields "
                    "propagated.".format(input_ports, self.name))
            else:
                for i in range(len(input_ports)):
                    index = uni_ports.index(input_ports[i])
                    output_ports.append(uni_output_ports[index])

        return output_ports
    # ==================================================================
    # Link management ==================================================
    # ==================================================================
    def is_link_to_neighbor_unique(self, neighbor_comp: AbstractComponent
                                   ) -> bool:

        count = 0
        for port in self._ports:
            if ((port.ngbr_comp is not None)
                    and (port.ngbr_comp == neighbor_comp)):
                    count += 1

        return count > 1
    # ==================================================================
    def is_linked_to(self, neighbor_comp: AbstractComponent) -> bool:

        for port in self._ports:
            if ((port.ngbr_comp is not None)
                    and (port.ngbr_comp == neighbor_comp)):

                return True

        return False
    # ==================================================================
    def is_linked_unidir_to(self, neighbor_comp: AbstractComponent) -> bool:

        for port in self._ports:
            if ((port.ngbr_comp is not None)
                    and (port.ngbr_comp == neighbor_comp)
                    and port.is_unidir()):

                return True

        return False
    # ==================================================================
    # Constraints management ===========================================
    # ==================================================================
    def add_wait_policy(self, *policy: List[int]) -> None:
        """Append a new policy to automatically make a port waiting for
        other port(s).

        Parameters
        ----------
        policy :
            The number(s) in the list correspond to the port number(s)
            which have to wait for each others.

        """
        for elem in policy:
            self._wait_policy.append(elem)
    # ==================================================================
    def get_wait_policy(self, port: int) -> List[List[int]]:

        waiting_ports: List[List[int]] = []
        if (self._wait):
            for policy in self._wait_policy:
                if (port in policy):
                    waiting_ports.append(policy)

        return waiting_ports
    # ==================================================================
    def inc_counter_pass(self, port: int) -> None:
        self._counter_pass[port] += 1
    # ==================================================================
    def dec_counter_pass(self, port: int) -> None:
        self._counter_pass[port] -= 1
    # ==================================================================
    def is_counter_below_max_pass(self, port: int) -> bool:

        return (self._counter_pass[port] <= self._max_nbr_pass[port])
