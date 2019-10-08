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
from optcom.domain import Domain
from optcom.field import Field
from optcom.field import EmptyField


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
        self.ports_type: List[int] = ports_type
        self._nbr_ports: int = len(ports_type)
        self.save: bool = save
        # _ports = [(neighbor_comp_1, input_port_neighbor_1), ...]
        self._ports: List[Optional[Tuple[AbstractComponent, int]]] =\
            [None for i in range(self._nbr_ports)]
        self._fields: List[Optional[Field]] =\
            [None for i in range(self._nbr_ports)]
        self._times: 'AbstractComponent.Times' = self.Times(self._nbr_ports)

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
    class Times():
        """Container for time array storage."""

        def __init__(self, nbr_ports: int) -> None:

            self._nbr_ports = nbr_ports
            self._times: List[Optional[Tuple[float, float, int]]] =\
                [None for i in range(self._nbr_ports)]
        # ==============================================================
        @overload
        def __getitem__(self, port: int) -> Array[float]: ...
        # ==============================================================
        @overload
        def __getitem__(self, port: slice) -> Array[float]: ...
        # ==============================================================
        def __getitem__(self, port):

            if (isinstance(port, slice)):
                res = []
                step: int = 0
                step = 1 if port.step is None else port.step
                for i in range(port.start, port.stop, step):
                    res.append(self.create_time_array(self._times[i]))

                return res
            else:

                return self.create_time_array(self._times[port])
        # ==============================================================
        def __setitem__(self, port: int, time: Array[float]) -> None:

            self._times[port] = (time[0], time[-1], len(time))
        # ==============================================================
        def create_time_array(self, time: Optional[Tuple[float, float, int]]
                              )-> Optional[Array[float]]:

            if (time is not None):

                return np.linspace(time[0], time[1], time[2], True, False)

            return None
    # ==================================================================
    def __str__(self) -> str:

        util.print_terminal("State of component '{}':".format(self.name))
        for i in range(self._nbr_ports):
            self.print_port_state(i)

        return str()
    # ==================================================================
    def __len__(self) -> int:

        return self._nbr_ports
    # ==================================================================
    def __del__(self) -> None:

        self.dec_nbr_instances()
        self.dec_nbr_instances_with_default_name()
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
    # Getters - setters - deleter ======================================
    # ==================================================================
    @property
    def fields(self) -> List[Field]:
        fields_: List[Field] = []
        for i in range(len(self._fields)):
            field = self._fields[i]
            if (field is not None):
                fields_.append(field)
            else:
                fields_.append(EmptyField())

        return fields_
    # ==================================================================
    @property
    def times(self) -> AbstractComponent.Times:

        return self._times
    # ==================================================================
    # Port management ==================================================
    # ==================================================================
    def print_port_state(self, port_nbr: int) -> None:

        if (self.is_port_valid(port_nbr)):
            port_ = self._ports[port_nbr]
            if (port_ is None):
                util.print_terminal("Port {} of component '{}' is free."
                      .format(port_nbr, self.name), '')
            else:
                if (port_[1] != cst.UNIDIR_PORT):
                    util.print_terminal("Port {} of component '{}' is linked "
                        "to port {} of component '{}'."
                        .format(port_nbr, self.name, port_[1], port_[0].name),
                        '')
                else:
                    util.print_terminal("Port {} of component '{}' has an "
                        "unidirectionnaly link from component '{}'."
                        .format(port_nbr, self.name, port_[0].name), '')
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
    def is_port_free(self, port_nbr: int) -> bool:

        if (self.is_port_valid(port_nbr)):

            return self._ports[port_nbr] is None

        return False
    # ==================================================================
    def is_port_type_in(self, port_nbr: int) -> bool:

        if (self.is_port_valid(port_nbr)):

            return self.ports_type[port_nbr] in cst.IN_PORTS

        return False
    # ==================================================================
    def is_port_type_out(self, port_nbr: int) -> bool:

        if (self.is_port_valid(port_nbr)):

            return self.ports_type[port_nbr] in cst.OUT_PORTS

        return False
    # ==================================================================
    def is_port_type_valid(self, port_nbr: int, field: Field) -> bool:

        if (self.is_port_valid(port_nbr)):
            return (self.get_port_type(port_nbr)
                    in cst.FIELD_TO_PORT.get(field.type))

        return False
    # ==================================================================
    def get_port_type(self, port_nbr: int) -> int:

        return self.ports_type[port_nbr]
    # ==================================================================
    def get_port_neighbor(self, port_nbr: int) -> int:

        if (self.is_port_valid(port_nbr)):
            port_ = self._ports[port_nbr]
            if (port_ is not None):

                return port_[1]

        return cst.UNIDIR_PORT
    # ==================================================================
    def get_neighbor(self, port_nbr: int) -> Optional[AbstractComponent]:

        if (self.is_port_valid(port_nbr)):
            port_ = self._ports[port_nbr]
            if (port_ is not None):

                return port_[0]

        return None
    # ==================================================================
    def get_port(self, port_nbr: int
                 ) -> Optional[Tuple[AbstractComponent, int]]:

        if (self.is_port_valid(port_nbr)):

            return self._ports[port_nbr]

        return None
    # ==================================================================
    def set_port(self, port_nbr: int,
                 comp_and_port: Tuple[AbstractComponent, int]) -> None:

        if (self.is_port_valid(port_nbr)
                and util.check_attr_type(comp_and_port, tuple)):
            # Could also check type of elem of tuple here
            self._ports[port_nbr] = comp_and_port
    # ==================================================================
    def del_port(self, port_nbr: int) -> None:

        if (self.is_port_valid(port_nbr)):
            self._ports[port_nbr] = None
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
    def link_to(self, port_nbr: int, neighbor_comp: AbstractComponent,
                neighbor_port_nbr: int) -> None:

        if (self.is_port_valid(port_nbr)):
            # Could also check type of elem here
            self._ports[port_nbr] = (neighbor_comp, neighbor_port_nbr)
    # ==================================================================
    def is_link_to_neighbor_unique(self, neighbor_comp: AbstractComponent
                                   ) -> bool:

        count = 0
        for comp_and_port in self._ports:
            if (comp_and_port is not None):
                if (comp_and_port[0] == neighbor_comp):
                    count += 1

        return count > 1
    # ==================================================================
    def is_linked_to(self, port_nbr: int, neighbor_comp: AbstractComponent,
                     neighbor_port_nbr: int) -> bool:

        port_ = self._ports[port_nbr]
        if (port_ is not None):

            return (port_[0] == neighbor_comp
                    and port_[1] == neighbor_port_nbr)

        return False
    # ==================================================================
    def is_linked_unidir_to(self, port_nbr: int,
                            neighbor_comp: AbstractComponent) -> bool:

        port_ = self._ports[port_nbr]
        if (port_ is not None):

            return (port_[0] == neighbor_comp
                    and port_[1] == cst.UNIDIR_PORT)

        return False
    # ==================================================================
    # Operator overloading for link management (accessible to user) ====
    # ==================================================================
    def __getitem__(self, key: int) -> Tuple[AbstractComponent, int]:

        return (self, key)
    # ==================================================================
    # Field management =================================================
    # ==================================================================
    def save_field(self, port_nbr: int, field: Field) -> None:
        current_field = self._fields[port_nbr]
        if (current_field is None):
            self._fields[port_nbr] = copy.deepcopy(field)
        else:
            current_field.extend(field)
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
