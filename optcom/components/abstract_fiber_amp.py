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

import copy
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.components.abstract_pass_comp import AbstractPassComp
from optcom.components.abstract_pass_comp import call_decorator
from optcom.domain import Domain
from optcom.field import Field


class AbstractFiberAmp(AbstractPassComp):
    r"""A non ideal Fiber Amplifier.

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
    REFL_SEED : bool
        If True, take into account the reflected seed waves for
        computation.
    REFL_PUMP : bool
        If True, take into account the reflected pump waves for
        computation.
    PROP_PUMP : bool
        If True, the pump is propagated forward in the layout.
    PROP_REFL : bool
        If True, the relfected fields are propagated in the layout
        as new fields.
    BISEED : bool
        If True, waiting policy waits for seed at both ends.
    BIPUMP : bool
        If True, waiting policy waits for pump at both ends.

    Notes
    -----
    Component diagram::

        [0] _____________________ [1]
            /                   \
           /                     \
        [2]                       [3]


    [0] and [1] : signal and [2] and [3] : pump

    """

    def __init__(self, name: str, default_name: str, save: bool,
                 max_nbr_pass: Optional[List[int]], pre_call_code: str,
                 post_call_code: str, REFL_SEED: bool, REFL_PUMP: bool,
                 PROP_PUMP: bool, PROP_REFL: bool, BISEED: bool, BIPUMP: bool
                 ) -> None:
        r"""
        Parameters
        ----------
        name :
            The name of the component.
        default_name :
            The default name of the amplifier.
        save :
            If True, the last wave to enter/exit a port will be saved.
        max_nbr_pass :
            No fields will be propagated if the number of
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
        REFL_SEED : bool
            If True, take into account the reflected seed waves for
            computation.
        REFL_PUMP : bool
            If True, take into account the reflected pump waves for
            computation.
        PROP_PUMP :
            If True, the pump is propagated forward in the layout.
        PROP_REFL :
            If True, the relfected fields are propagated in the layout
            as new fields.
        BISEED :
            If True, waiting policy waits for seed at both ends.
        BIPUMP :
            If True, waiting policy waits for pump at both ends.

        """
        # Parent constructor -------------------------------------------
        ports_type = [cst.OPTI_ALL, cst.OPTI_ALL, cst.OPTI_IN, cst.OPTI_IN]
        super().__init__(name, default_name, ports_type, save, wait=True,
                         max_nbr_pass=max_nbr_pass,
                         pre_call_code=pre_call_code,
                         post_call_code=post_call_code)
        # Attr types check ---------------------------------------------
        util.check_attr_type(REFL_SEED, 'REFL_SEED', bool)
        util.check_attr_type(REFL_PUMP, 'REFL_PUMP', bool)
        util.check_attr_type(PROP_PUMP, 'PROP_PUMP', bool)
        util.check_attr_type(PROP_REFL, 'PROP_REFL', bool)
        util.check_attr_type(BISEED, 'BISEED', bool)
        util.check_attr_type(BIPUMP, 'BIPUMP', bool)
        # Policy -------------------------------------------------------
        self.REFL_SEED = REFL_SEED
        self.REFL_PUMP = REFL_PUMP
        self._PROP_PUMP = PROP_PUMP
        self.PROP_REFL = PROP_REFL
        self._BISEED = BISEED
        self._BIPUMP = BIPUMP
        self.set_port_policy()
        self.set_wait_policy()
    # ==================================================================
    @property
    def PROP_PUMP(self) -> bool:

        return self._PROP_PUMP
    # ------------------------------------------------------------------
    @PROP_PUMP.setter
    def PROP_PUMP(self, PROP_PUMP: bool) -> None:
        self._PROP_PUMP = PROP_PUMP
        self.reset_port_policy()
        self.set_port_policy()
    # ==================================================================
    @property
    def BISEED(self) -> bool:

        return self._BISEED
    # ------------------------------------------------------------------
    @BISEED.setter
    def BISEED(self, BISEED: bool) -> None:
        self._BISEED = BISEED
        self.reset_wait_policy()
        self.set_wait_policy()
    # ==================================================================
    @property
    def BIPUMP(self) -> bool:

        return self._BIPUMP
    # ------------------------------------------------------------------
    @BIPUMP.setter
    def BIPUMP(self, BIPUMP: bool) -> None:
        self._BIPUMP = BIPUMP
        self.reset_wait_policy()
        self.set_wait_policy()
    # ==================================================================
    def set_port_policy(self) -> None:
        if (self.PROP_PUMP):
            self.add_port_policy(([0,2], [1,1], False),
                                 ([0,3], [1,0], False),
                                 ([1,2], [0,1], False),
                                 ([1,3], [0,0], False),
                                 ([0,2,3], [1,1,0], False),
                                 ([1,2,3], [0,1,0], False),
                                 ([0,1,2], [1,0,1], False),
                                 ([0,1,3], [1,0,0], False),
                                 ([0,1,2,3], [1,0,1,0], False))
        else:
            self.add_port_policy(([0,2], [1,-1], False),
                                 ([0,3], [1,-1], False),
                                 ([1,2], [0,-1], False),
                                 ([1,3], [0,-1], False),
                                 ([0,2,3], [1,-1,-1], False),
                                 ([1,2,3], [0,-1,-1], False),
                                 ([0,1,2], [1,0,-1], False),
                                 ([0,1,3], [1,0,-1], False),
                                 ([0,1,2,3], [1,0,-1,-1], False))
    # ==================================================================
    def output_ports(self, input_ports: List[int]) -> List[int]:
        seeds: List[int] = []
        pumps: List[int] = []
        for i in range(len(input_ports)): # Get indices of pumps and seeds
            if ((input_ports[i] == 2) or (input_ports[i] == 3)):
                pumps.append(i)
            else:
                seeds.append(i)
        output_ports: List[int] = super().output_ports(input_ports)
        for i in range(len(output_ports)):
            if (self.PROP_REFL and ((self.REFL_SEED and (i in seeds))
                    or (self.REFL_PUMP and (i in pumps)))):
                output_ports.append(output_ports[i]^1)
            else:
                output_ports.append(-1)

        return output_ports
    # ==================================================================
    def set_wait_policy(self) -> None:
        if (self.BISEED and self.BIPUMP):
            self.add_wait_policy([0,1,2,3])
        elif (self.BISEED):
            self.add_wait_policy([0,1,2], [0,1,3])
        elif (self.BIPUMP):
            self.add_wait_policy([0,2,3], [1,2,3])
        else:
            self.add_wait_policy([0,2], [0,3], [1,2], [1,3])
    # ==================================================================
    @call_decorator
    def __call__(self, domain: Domain, ports: List[int], fields: List[Field]
                 ) -> Tuple[List[int], List[Field]]: ...
