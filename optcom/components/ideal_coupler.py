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
    ratios_ports :
        Each element of the list correspond to one input port and
        contain a list composed of the two dividing ratio
        for the two output ports. The ratio represents the
        fraction of the power that will be taken from the field.

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
                 ratios_ports: List[List[float]] = [[0.5, 0.5]],
                 save: bool = False, max_nbr_pass: Optional[List[int]] = None,
                 pre_call_code: str = '', post_call_code: str = '') -> None:
        """
        Parameters
        ----------
        name :
            The name of the component.
        ratios_ports :
            Each element of the list correspond to one input port and
            contain a list composed of the two dividing ratio
            for the two output ports. The ratio represents the
            fraction of the power that will be taken from the field.
        save :
            If True, the last wave to enter/exit a port will be saved.
        max_nbr_pass :
            No fields will be propagated if the number of
            fields which passed through a specific port exceed the
            specified maximum number of pass for this port.
        pre_call_code :
            A string containing code which will be executed prior to
            the call to the function :func:`__call__`. The two parameters
            `input_ports` and `input_fields` are available.
        post_call_code :
            A string containing code which will be executed posterior to
            the call to the function :func:`__call__`. The two parameters
            `output_ports` and `output_fields` are available.

        """
        # Parent constructor -------------------------------------------
        ports_type = [cst.ANY_ALL for i in range(4)]
        super().__init__(name, default_name, ports_type, save,
                         max_nbr_pass=max_nbr_pass,
                         pre_call_code=pre_call_code,
                         post_call_code=post_call_code)
        # Attr types check ---------------------------------------------
        util.check_attr_type(ratios_ports, 'ratios_ports', list)
        # Attr ---------------------------------------------------------
        self._ratios_ports: List[List[float]]
        self.ratios_ports = ratios_ports
    # ==================================================================
    @property
    def ratios_ports(self) -> List[List[float]]:

        return self._ratios_ports
    # ------------------------------------------------------------------
    @ratios_ports.setter
    def ratios_ports(self, ratios_ports: List[List[float]]) -> None:
        ratios_ports = util.make_list(ratios_ports, 4)
        # N.B. name='nocount' to avoid inc. default name counter
        self._divider_0 = IdealDivider(name='nocount', arms=2, divide=True,
                                       ratios=ratios_ports[0])
        self._divider_1 = IdealDivider(name='nocount', arms=2, divide=True,
                                       ratios=ratios_ports[1])
        self._divider_2 = IdealDivider(name='nocount', arms=2, divide=True,
                                       ratios=ratios_ports[2])
        self._divider_3 = IdealDivider(name='nocount', arms=2, divide=True,
                                       ratios=ratios_ports[3])
    # ==================================================================
    @call_decorator
    def __call__(self, domain: Domain, ports: List[int], fields: List[Field]
                 ) -> Tuple[List[int], List[Field]]:

        output_ports: List[int] = []
        output_fields: List[Field] = []
        for i in range(len(fields)):
            divider = getattr(self, "_divider_{}".format(ports[i]))
            output_fields.extend(divider(domain, [0], [fields[i]])[1])
            # 0 -> 2,3 /\ 1 -> 2,3 /\ 2 -> 0,1 /\ 3 -> 0,1
            output_port = ((ports[i] | 1) + 2) % 4
            output_ports.append(output_port-1)
            output_ports.append(output_port)

        return output_ports, output_fields

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

    ratios_ports: List[List[float]] = [[0.6, 0.4], [0.5, 0.5],
                                       [0.4, 0.6], [0.0, 1.0]]

    plot_labels: List[Optional[str]] = [None]
    plot_groups: List[int] = [0]
    plot_titles: List[str] = ["Original pulse"]
    y_datas: List[np.ndarray] = []
    x_datas: List[np.ndarray] = []
    coupler: oc.IdealCoupler
    plot_save: int
    for i in range(4):
        # Propagation
        coupler = oc.IdealCoupler(ratios_ports=[ratios_ports[i]])
        lt.link((pulse[0], coupler[i]))
        lt.run(pulse)
        lt.reset()
        # Plot parameters and get waves
        port_saved = ((i | 1) + 2) % 4
        y_datas.append(oc.temporal_power(coupler[port_saved-1][0].channels))
        y_datas.append(oc.temporal_power(coupler[port_saved][0].channels))
        x_datas.append(coupler[port_saved-1][0].time)
        x_datas.append(coupler[port_saved][0].time)
        plot_labels += ["port " + str(port_saved-1), "port " + str(port_saved)]
        plot_groups += [i+1, i+1]
        plot_titles += ["Pulses coming out of the ideal coupler from input "
                        "port {} with ratios {}".format(i, ratios_ports[i])]

    y_datas = [oc.temporal_power(pulse[0][0].channels)] + y_datas
    x_datas = [pulse[0][0].time] + x_datas

    oc.plot2d(x_datas, y_datas, plot_groups=plot_groups,
              plot_titles=plot_titles, x_labels=['t'], y_labels=['P_t'],
              plot_labels=plot_labels, opacity=[0.3])
