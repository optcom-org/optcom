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
from abc import ABCMeta, abstractmethod
from typing import Any, ClassVar, Dict, List, Optional, overload, Tuple, Union

import numpy as np

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.components.port import Port
from optcom.domain import Domain
from optcom.field import Field
from optcom.utils.storage import Storage


default_name = 'AbstractComponent'


class AbstractComponent(metaclass=ABCMeta):
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
    call_counter : int
        Count the number of times the function
        :func:`__call__` of the Component has been called.
    wait :
        If True, will wait for specified waiting port policy added
        with the function :func:`AbstractComponent.add_wait_policy`.
    pre_call_code :
        A string containing code which will be executed prior to
        the call to the function :func:`__call__`. The two parameters
        `input_ports` and `input_fields` are available.
    post_call_code :
        A string containing code which will be executed posterior to
        the call to the function :func:`__call__`. The two parameters
        `output_ports` and `output_fields` are available.

    """

    _nbr_instances: int = 0
    _nbr_instances_with_default_name: int = 0

    def __init__(self, name: str, default_name: str, ports_type: List[int],
                 save: bool, wait: bool = False,
                 max_nbr_pass: Optional[List[int]] = None,
                 pre_call_code: str = '', post_call_code: str = '') -> None:
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
        pre_call_code :
            A string containing code which will be executed prior to
            the call to the function :func:`__call__`. The two
            parameters `input_ports` and `input_fields` are available.
        post_call_code :
            A string containing code which will be executed posterior to
            the call to the function :func:`__call__`. The two
            parameters `output_ports` and `output_fields` are available.

        """
        # Attr type check ----------------------------------------------
        util.check_attr_type(name, 'name', str)
        util.check_attr_type(default_name, 'default_name', str)
        util.check_attr_type(ports_type, 'ports_type', list)
        util.check_attr_type(save, 'save', bool)
        util.check_attr_type(wait, 'wait', bool)
        util.check_attr_type(max_nbr_pass, 'max_nbr_pass', None, list)
        util.check_attr_type(pre_call_code, 'pre_call_code', str)
        util.check_attr_type(post_call_code, 'post_call_code', str)
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
        self._storages: List[Storage] = []
        self._ports: List[Port] = []
        max_nbr_pass_: List[Optional[int]]
        max_nbr_pass_ = util.make_list(max_nbr_pass, self._nbr_ports)
        for i in range(self._nbr_ports):
            max_nbr = max_nbr_pass_[i]
            if (max_nbr is not None):
                self._ports.append(Port(self, ports_type[i], max_nbr))
            else:
                self._ports.append(Port(self, ports_type[i]))
        self._port_policy: Dict[Tuple[int,...], Tuple[int,...]] = {}
        self._wait_policy: List[List[int]] = []
        self._wait: bool = wait
        self.pre_call_code: str = pre_call_code
        self.post_call_code: str = post_call_code
        self.call_counter: int = 0
        self._ptr: int = 0   # for __iter__() method
    # ==================================================================
    @abstractmethod
    def __call__(self, domain: Domain) -> Tuple[List[int], List[Field]]: pass
    # ==================================================================
    def __str__(self) -> str:
        str_to_return: str = ""
        str_to_return += "State of component '{}':\n\n".format(self.name)
        for i in range(self._nbr_ports):
            str_to_return += str(self._ports[i]) + '\n'

        return str_to_return
    # ==================================================================
    def __len__(self) -> int:

        return self._nbr_ports
    # ==================================================================
    def __del__(self) -> None:

        self.dec_nbr_instances()
        self.dec_nbr_instances_with_default_name()
    # ==================================================================
    def __getitem__(self, key: int) -> Port:

        return self._ports[key]
    # ==================================================================
    def __iter__(self) -> AbstractComponent:
        self._ptr = 0

        return self
    # ==================================================================
    def __next__(self) -> Port:
        if (self._ptr == len(self._ports)):

            raise StopIteration
        elem = self._ports[self._ptr]
        self._ptr += 1

        return elem
    # ==================================================================
    @property
    def storages(self) -> List[Storage]:

        return self._storages
    # ==================================================================
    @property
    def storage(self) -> Optional[Storage]:
        """Return the last saved storage if exists, otherwise None."""
        if (self._storages):

            return self._storages[-1]
        else:

            return None
    # ==================================================================
    @property
    def wait(self) -> bool:

        return self._wait
    # ------------------------------------------------------------------
    @wait.setter
    def wait(self, wait: bool) -> None:
        self._wait = wait
    # ==================================================================
    def get_port(self, port_nbr: int) -> Port:

        return self[port_nbr]
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
    def port_id_of(self, port: Port) -> int:
        if (port in self._ports):

            return self._ports.index(port)

        return cst.NULL_PORT_ID
    # ==================================================================
    def is_port_id_valid(self, port_id: int) -> bool:

        return (0 <= port_id < self._nbr_ports)
    # ==================================================================
    def del_port_id(self, port_id: int) -> None:

        if (self.is_port_id_valid(port_id)):
            self._ports[port_id].reset()
    # ==================================================================
    def reset_port_policy(self) -> None:
        self._port_policy = {}
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
                N.B.: if (len(a) < len(b)), pad a with length of b.
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
            util.warning_terminal("No policy specified on port management "
                "for component {}, no field propagated.".format(self.name))
        else:
            uni_ports = util.unique(input_ports)
            uni_output_ports = self._port_policy.get(tuple(uni_ports))
            if (uni_output_ports is None):
                util.warning_terminal("The input ports {} provided for "
                    "component {} do not match any policy, no field "
                    "propagated.".format(input_ports, self.name))
            else:
                for i in range(len(input_ports)):
                    index = uni_ports.index(input_ports[i])
                    output_ports.append(uni_output_ports[index])

        return output_ports
    # ==================================================================
    # Link management ==================================================
    # ==================================================================
    def is_ngbr_unique(self, ngbr: AbstractComponent) -> bool:

        count = 0
        for port in self._ports:
            if ((not port.is_free()) and (port.ngbr == ngbr)):
                count += 1

        return count > 1
    # ==================================================================
    def is_linked_to(self, ngbr: AbstractComponent) -> bool:

        for port in self._ports:
            if ((not port.is_free()) and (port.ngbr == ngbr)):

                return True

        return False
    # ==================================================================
    def is_linked_unidir_to(self, ngbr: AbstractComponent) -> bool:

        for port in self._ports:
            if ((not port.is_free()) and (port.ngbr == ngbr)
                    and port.is_unidir()):

                return True

        return False
    # ==================================================================
    # Waiting Policy ===================================================
    # ==================================================================
    def reset_wait_policy(self) -> None:
        self._wait_policy = []
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
