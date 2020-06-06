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
from typing import List, Optional, Sequence, Tuple

import numpy as np

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.components.abstract_pass_comp import AbstractPassComp
from optcom.components.abstract_pass_comp import call_decorator
from optcom.domain import Domain
from optcom.field import Field


default_name: str = 'Ideal Combiner'


class IdealCombiner(AbstractPassComp):
    r"""An ideal Combiner.

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
    arms : int
        The number of input arms.
    combine :
        If False, propagate all incoming fields in the last port.
        Otherwise, add common channels and append the others to
        create one single field. Both methods operate depending on
        ratios provided.
    ratios :
        A list of ratios where each index is related to one arm.
        The length of the list should be equal to the number of
        arms, if not it will be pad to it. The ratio represents the
        fraction of the power that will be taken from the field
        arriving at the corresponding arm.

    Notes
    -----
    Component diagram::

        [0]   _________
        [1]   __________\
        [2]   ___________\_______ [n]
        [3]   ___________/
            ...
        [n-1] _________/

    """

    _nbr_instances: int = 0
    _nbr_instances_with_default_name: int = 0

    def __init__(self, name: str = default_name, arms: int = 2,
                 combine: bool = False, ratios: List[float] = [],
                 wait: bool = True, save: bool = False,
                 max_nbr_pass: Optional[List[int]] = None,
                 pre_call_code: str = '', post_call_code: str = '') -> None:
        """
        Parameters
        ----------
        name :
            The name of the component.
        arms :
            The number of input arms.
        combine :
            If False, propagate all incoming fields in the last port.
            Otherwise, add common channels and append the others to
            create one single field. Both methods operate depending on
            ratios provided.
        ratios :
            A list of ratios where each index is related to one arm.
            The length of the list should be equal to the number of
            arms, if not it will be pad to it. The ratio represents the
            fraction of the power that will be taken from the field
            arriving at the corresponding arm.
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
        ports_type = [cst.ANY_IN for i in range(arms)] + [cst.ANY_OUT]
        super().__init__(name, default_name, ports_type, save, wait=wait,
                         max_nbr_pass=max_nbr_pass,
                         pre_call_code=pre_call_code,
                         post_call_code=post_call_code)
        # Check types attr ---------------------------------------------
        util.check_attr_type(arms, 'arms', int)
        util.check_attr_type(combine, 'combine', bool)
        util.check_attr_type(ratios, 'ratios', list)
        # Attr ---------------------------------------------------------
        self.arms: int = arms
        self.ratios: List[float] = []
        if (not ratios):
            self.ratios = [1.0 for i in range(self.arms)]
        else:
            self.ratios = ratios
        self.combine: bool = combine
        # Policy -------------------------------------------------------
        for i in range(arms):
            self.add_port_policy(([i], [arms], False))
        self.add_port_policy(([i for i in range(arms)], [arms], False))
        self.add_wait_policy([i for i in range(arms)])
    # ==================================================================
    @call_decorator
    def __call__(self, domain: Domain, ports: List[int], fields: List[Field]
                 ) -> Tuple[List[int], List[Field]]:
        output_ports: List[int] = []
        output_fields: List[Field] = []
        # Combine the wave in list fields ------------------------------
        if (self.combine):
            fields[0] *= math.sqrt(self.ratios[ports[0]])
            for i in range(1, len(fields)):
                fields[i] *= math.sqrt(self.ratios[ports[i]])
                fields[0].operator_or_extend('__iadd__', fields[i])
            for i in range(len(fields)-1, 0, -1):
                del fields[i]

            return self.output_ports([ports[0]]), [fields[0]]
        else:
            for i in range(len(fields)):
                output_fields.append(fields[i]
                                     * math.sqrt(self.ratios[ports[i]]))

        return self.output_ports(ports), output_fields


if __name__ == "__main__":
    """Give an example of IdealCombiner usage.
    This piece of code is standalone, i.e. can be used in a separate
    file as an example.
    """

    from typing import Callable, List, Optional

    import numpy as np

    import optcom as oc

    pulse_1: oc.Gaussian = oc.Gaussian(peak_power=[1.0],
                                       center_lambda=[1550.0])
    pulse_2: oc.Gaussian = oc.Gaussian(peak_power=[5.0],
                                       center_lambda=[1030.0])
    pulse_3: oc.Gaussian = oc.Gaussian(peak_power=[10.0],
                                       center_lambda=[976.0])
    # Dummy component to be able to test the not combining case
    pm: oc.IdealPhaseMod = oc.IdealPhaseMod()
    lt: oc.Layout = oc.Layout()

    combiner: oc.IdealCombiner = oc.IdealCombiner(arms=3, combine=False)

    lt.link((pulse_1[0], combiner[0]), (pulse_2[0], combiner[1]),
            (pulse_3[0], combiner[2]), (combiner[3], pm[0]))

    lt.run(pulse_1, pulse_2, pulse_3)

    plot_titles: List[str] = (["Original pulses", "Pulses coming out of the "
                               "ideal coupler \n without combination"])
    plot_groups: List[int] = [0,0,0,1]
    plot_labels: List[Optional[str]] = ['port 0', 'port 1', 'port 2', None]

    out_channels: np.ndarray = oc.temporal_power(pm[1][0].channels)
    for i in range(len(pm[1])):
        out_channels = np.vstack((out_channels,
                                  oc.temporal_power(pm[1][i].channels)))
    y_datas: List[np.ndarray] = [oc.temporal_power(pulse_1[0][0].channels),
                                 oc.temporal_power(pulse_2[0][0].channels),
                                 oc.temporal_power(pulse_3[0][0].channels),
                                 out_channels]
    x_datas: List[np.ndarray] = [pulse_1[0][0].time, pulse_2[0][0].time,
                                 pulse_3[0][0].time,pm[1][0].time]

    lt.reset()
    pm = oc.IdealPhaseMod()

    pulse_1 = oc.Gaussian(peak_power=[1.0], center_lambda=[1550.0])
    pulse_2 = oc.Gaussian(channels=2, peak_power=[5.0],
                                center_lambda=[1550.0, 1540.0])
    pulse_3 = oc.Gaussian(peak_power=[10.0], center_lambda=[976.0])

    combiner = oc.IdealCombiner(arms=3, combine=True)

    lt.link((pulse_1[0], combiner[0]), (pulse_2[0], combiner[1]),
            (pulse_3[0], combiner[2]), (combiner[3], pm[0]))

    lt.run(pulse_1, pulse_2, pulse_3)

    plot_titles.extend(["Original pulses",
                        "Pulses coming out of the ideal coupler \n with "
                        "combination"])
    plot_groups.extend([2,2,2,3])
    plot_labels.extend(['port 0', 'port 1', 'port 2', None])

    y_datas.extend([oc.temporal_power(pulse_1[0][0].channels),
                    oc.temporal_power(pulse_2[0][0].channels),
                    oc.temporal_power(pulse_3[0][0].channels),
                    oc.temporal_power(pm[1][0].channels)])
    x_datas.extend([pulse_1[0][0].time, pulse_2[0][0].time, pulse_3[0][0].time,
                    pm[1][0].time])

    oc.plot2d(x_datas, y_datas, plot_labels=plot_labels,
              plot_groups=plot_groups, plot_titles=plot_titles,
              x_labels=['t'], y_labels=['P_t'], opacity=[0.3])
