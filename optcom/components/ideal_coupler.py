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

import math
import copy

import numpy as np
from typing import List, Optional, Sequence, Tuple

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.components.abstract_pass_comp import AbstractPassComp
from optcom.components.abstract_pass_comp import call_decorator
from optcom.components.ideal_divider import IdealDivider
from optcom.domain import Domain
from optcom.field import Field


default_name = 'Ideal Fiber Coupler'


class IdealCoupler(AbstractPassComp):
    r"""An ideal fiber coupler

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
    port_ratios :
        Each element of the list correspond to one input port and
        contain a list composed of the two dividing ratio
        for the two output ports. The ratio represents the
        fraction of the power that will be taken from the field.
    NOISE :
        If True, the noise is handled, otherwise is unchanged.

    Notes
    -----
    Component diagram::

        [0] _______        ______ [2]
                   \______/
        [1] _______/      \______ [3]

    """

    _nbr_instances: int = 0
    _nbr_instances_with_default_name: int = 0

    def __init__(self, name: str = default_name,
                 port_ratios: List[List[float]] = [[0.5, 0.5]],
                 NOISE: bool = False, save: bool = False,
                 max_nbr_pass: Optional[List[int]] = None,
                 pre_call_code: str = '', post_call_code: str = '') -> None:
        """
        Parameters
        ----------
        name :
            The name of the component.
        port_ratios :
            Each element of the list correspond to one input port and
            contain a list composed of the two dividing ratio
            for the two output ports. The ratio represents the
            fraction of the power that will be taken from the field.
        NOISE :
            If True, the noise is handled, otherwise is unchanged.
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
        ports_type = [cst.ANY_ALL for i in range(4)]
        super().__init__(name, default_name, ports_type, save,
                         max_nbr_pass=max_nbr_pass,
                         pre_call_code=pre_call_code,
                         post_call_code=post_call_code)
        # Attr types check ---------------------------------------------
        util.check_attr_type(port_ratios, 'port_ratios', list)
        util.check_attr_type(NOISE, 'NOISE', bool)
        # Attr ---------------------------------------------------------
        self.NOISE = NOISE
        self._port_ratios: List[List[float]]
        self.port_ratios = port_ratios
    # ==================================================================
    @property
    def port_ratios(self) -> List[List[float]]:

        return self._port_ratios
    # ------------------------------------------------------------------
    @port_ratios.setter
    def port_ratios(self, port_ratios: List[List[float]]) -> None:
        port_ratios = util.make_list(port_ratios, 4)
        # N.B. name='nocount' to avoid inc. default name counter
        self._divider_0 = IdealDivider(name='nocount', arms=2, divide=True,
                                       ratios=port_ratios[0], NOISE=self.NOISE)
        self._divider_1 = IdealDivider(name='nocount', arms=2, divide=True,
                                       ratios=port_ratios[1], NOISE=self.NOISE)
        self._divider_2 = IdealDivider(name='nocount', arms=2, divide=True,
                                       ratios=port_ratios[2], NOISE=self.NOISE)
        self._divider_3 = IdealDivider(name='nocount', arms=2, divide=True,
                                       ratios=port_ratios[3], NOISE=self.NOISE)
    # ==================================================================
    def output_ports(self, input_ports: List[int]) -> List[int]:
        output_ports: List[int] = []
        for i in range(len(input_ports)):
            # 0 -> 2,3 /\ 1 -> 2,3 /\ 2 -> 0,1 /\ 3 -> 0,1
            output_port = ((input_ports[i] | 1) + 2) % 4
            output_ports.append(output_port-1)
            output_ports.append(output_port)

        return output_ports
    # ==================================================================
    @call_decorator
    def __call__(self, domain: Domain, ports: List[int], fields: List[Field]
                 ) -> Tuple[List[int], List[Field]]:
        output_fields: List[Field] = []
        for i in range(len(fields)):    # Call the corresponding divider
            divider = getattr(self, "_divider_{}".format(ports[i]))
            output_fields.extend(divider(domain, [0], [fields[i]])[1])

        return self.output_ports(ports), output_fields

if __name__ == "__main__":
    """Give an example of IdealCoupler usage.
    This piece of code is standalone, i.e. can be used in a separate
    file as an example.
    """

    import random
    from typing import Callable, List, Optional

    import numpy as np

    import optcom as oc

    pulse: oc.Gaussian = oc.Gaussian(peak_power=[10.0])

    lt: oc.Layout = oc.Layout()

    port_ratios: List[List[float]] = [[0.6, 0.4], [0.5, 0.5],
                                      [0.4, 0.6], [0.0, 1.0]]

    line_labels: List[Optional[str]] = [None]
    plot_groups: List[int] = [0]
    plot_titles: List[str] = ["Original pulse"]
    y_datas: List[np.ndarray] = []
    x_datas: List[np.ndarray] = []
    coupler: oc.IdealCoupler
    plot_save: int
    for i in range(4):
        # Propagation
        coupler = oc.IdealCoupler(port_ratios=[port_ratios[i]])
        lt.add_link(pulse[0], coupler[i])
        lt.run(pulse)
        lt.reset()
        # Plot parameters and get waves
        port_saved = ((i | 1) + 2) % 4
        y_datas.append(oc.temporal_power(coupler[port_saved-1][0].channels))
        y_datas.append(oc.temporal_power(coupler[port_saved][0].channels))
        x_datas.append(coupler[port_saved-1][0].time)
        x_datas.append(coupler[port_saved][0].time)
        line_labels += ["port " + str(port_saved-1), "port " + str(port_saved)]
        plot_groups += [i+1, i+1]
        plot_titles += ["Pulses coming out of the ideal coupler from input "
                        "port {} with ratios {}".format(i, port_ratios[i])]

    y_datas = [oc.temporal_power(pulse[0][0].channels)] + y_datas
    x_datas = [pulse[0][0].time] + x_datas

    oc.plot2d(x_datas, y_datas, plot_groups=plot_groups,
              plot_titles=plot_titles, x_labels=['t'], y_labels=['P_t'],
              line_labels=line_labels, line_opacities=[0.3])
