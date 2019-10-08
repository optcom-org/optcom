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
from optcom.components.ideal_divider import IdealDivider
from optcom.domain import Domain
from optcom.field import Field
from optcom.solvers.stepper import Stepper


default_name = 'Ideal Fiber Coupler'


class IdealFiberCoupler(AbstractPassComp):
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
        If True, the last wave to enter/exit a port will be saved.

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
                 max_nbr_pass: Optional[List[int]] = None, save: bool = False,
                 ) -> None:
        """
        Parameters
        ----------
        name :
            The name of the component.
        ratios_ports :
            Each element of the list contain a list with the two
            dividing percentages for the two output ports.
        max_nbr_pass :
            The maximum number of times a field can enter at the
            corresponding index-number port.
        save :
            If True, the last wave to enter/exit a port will be saved.

        """
        # Parent constructor -------------------------------------------
        ports_type = [cst.ANY_ALL for i in range(4)]
        super().__init__(name, default_name, ports_type, save,
                         max_nbr_pass=max_nbr_pass)
        # Attr types check ---------------------------------------------
        util.check_attr_type(ratios_ports, 'ratios_ports', list)
        # Attr ---------------------------------------------------------
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

    import random

    import optcom.utils.plot as plot
    import optcom.layout as layout
    import optcom.components.gaussian as gaussian
    import optcom.domain as domain
    from optcom.utils.utilities_user import temporal_power

    pulse = gaussian.Gaussian(peak_power=[10.0])

    lt = layout.Layout()

    ratios_ports=[[0.6, 0.4], [0.5, 0.5], [0.4, 0.6], [0.0, 1.0]]

    plot_labels: List[Optional[str]] = [None]
    plot_groups: List[Optional[int]] = [0]
    plot_titles: List[Optional[str]] = ["Original pulse"]
    fields = []
    time = []

    for i in range(4):
        # Propagation
        coupler = IdealFiberCoupler(ratios_ports=[ratios_ports[i]])
        lt.link((pulse[0], coupler[i]))
        lt.run(pulse)
        lt.reset()
        # Plot parameters and get waves
        port_saved = ((i | 1) + 2) % 4
        fields.append(temporal_power(coupler.fields[port_saved-1].channels))
        fields.append(temporal_power(coupler.fields[port_saved].channels))
        time.append(coupler.fields[port_saved-1].time)
        time.append(coupler.fields[port_saved].time)
        plot_labels += ["port " + str(port_saved-1), "port " + str(port_saved)]
        plot_groups += [i+1, i+1]
        plot_titles += ["Pulses coming out of the {} from input port {} with "
                        "ratios {}".format(default_name, i, ratios_ports[i])]

    fields= [temporal_power(pulse.fields[0].channels)] + fields
    time = [pulse.fields[0].time] + time

    plot.plot(time, fields,
              plot_groups=plot_groups, plot_titles=plot_titles, x_labels=['t'],
              y_labels=['P_t'], plot_labels=plot_labels, opacity=0.3)
