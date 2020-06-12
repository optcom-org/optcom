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

from typing import List, Optional, Tuple, Union

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.components.abstract_pass_comp import AbstractPassComp
from optcom.components.abstract_pass_comp import call_decorator
from optcom.domain import Domain
from optcom.field import Field

default_name = 'Ideal Isolator'

# Exceptions
class IdealIsolatorError(Exception):
    pass

class WrongPortError(IdealIsolatorError):
    pass


class IdealIsolator(AbstractPassComp):
    r"""An ideal Isolator.

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
    blocked_port : int
        The port id through which fields will not pass.

    Notes
    -----
    Component diagram::

        [0] __________________ [1]

    """

    _nbr_instances: int = 0
    _nbr_instances_with_default_name: int = 0

    def __init__(self, name: str = default_name, blocked_port: int = 0,
                 save: bool = False, max_nbr_pass: Optional[List[int]] = None,
                 pre_call_code: str = '', post_call_code: str = '') -> None:
        r"""
        Parameters
        ----------
        name :
            The name of the component.
        blocked_port :
            The port id through which fields will not pass (0 or 1).
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

        """
        # Parent constructor -------------------------------------------
        ports_type = [cst.ANY_ALL, cst.ANY_ALL]
        super().__init__(name, default_name, ports_type, save,
                         max_nbr_pass=max_nbr_pass,
                         pre_call_code=pre_call_code,
                         post_call_code=post_call_code)
        # Attr types check ---------------------------------------------
        util.check_attr_type(blocked_port, 'blocked_port', int)
        # Attr range check ---------------------------------------------
        util.check_attr_range(blocked_port, 'blocked_port', 0, 1)
        # Attr and Policy ----------------------------------------------
        self._blocked_port: int
        self.blocked_port = blocked_port
    # ==================================================================
    @property
    def blocked_port(self) -> int:

        return self._blocked_port
    # ------------------------------------------------------------------
    @blocked_port.setter
    def blocked_port(self, blocked_port: int) -> None:
        if (blocked_port == 1 or blocked_port == 0):
            self.reset_port_policy()
            self.add_port_policy(([blocked_port], [-1], False))
            self.add_port_policy(([blocked_port^1], [blocked_port], False))
            self._blocked_port = blocked_port
        else:
            error_msg: str = ("Ideal isolator has no port number {}."
                              .format(blocked_port))
    # ==================================================================
    @call_decorator
    def __call__(self, domain: Domain, ports: List[int], fields: List[Field]
                 ) -> Tuple[List[int], List[Field]]:

        return self.output_ports(ports), fields


if __name__ == "__main__":
    """Give an example of IdealIsolator usage.
    This piece of code is standalone, i.e. can be used in a separate
    file as an example.
    """

    from typing import List, Optional

    import numpy as np

    import optcom as oc

    lt: oc.Layout = oc.Layout()

    pulse: oc.Gaussian = oc.Gaussian(channels=1, peak_power=[10.0])
    isolator_1: oc.IdealIsolator = oc.IdealIsolator(blocked_port=1, save=True)

    lt.add_link(pulse[0], isolator_1[0])

    lt.run(pulse)

    plot_titles: List[str] = (['Initial Pulse',
                               'Output of first isolator (pass)'])

    y_datas: List[np.ndarray] = [oc.temporal_power(pulse[0][0].channels),
                                 oc.temporal_power(isolator_1[1][0].channels)]
    x_datas: List[np.ndarray] = [pulse[0][0].time, isolator_1[1][0].time]

    oc.plot2d(x_datas, y_datas, split=True, plot_titles=plot_titles,
              x_labels=['t'], y_labels=['P_t'], line_opacities=[0.3])
