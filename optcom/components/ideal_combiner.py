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

import numpy as np
from typing import List, Optional, Sequence, Tuple

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.components.abstract_pass_comp import AbstractPassComp
from optcom.domain import Domain
from optcom.field import Field


default_name = 'Ideal Combiner'


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
        If True, the last wave to enter/exit a port will be saved.
    arms : int
        The number of input arms.
    ratios : list of float
        A list with the combining ratios of the corresponding
        index-number port.

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
                 combine: bool = False, ratios: Optional[List[float]] = None,
                 wait: bool = True, save: bool = False) -> None:
        """
        Parameters
        ----------
        name :
            The name of the component.
        arms :
            The number of input arms.
        combine :
            If False, propagate all the fields without changing them in
            the last port. Otherwise, add values of channels with same
            center angular frequencies and append the others depending
            on ratios provided, then propagate the resulting field.
        ratios :
            A list with the combining ratios of the corresponding
            index-number port.
        save :
            If True, the last wave to enter/exit a port will be saved.

        """
        # Parent constructor -------------------------------------------
        ports_type = [cst.ANY_IN for i in range(arms)] + [cst.ANY_OUT]
        super().__init__(name, default_name, ports_type, save, wait=wait)
        # Check types attr ---------------------------------------------
        util.check_attr_type(arms, 'arms', int)
        util.check_attr_type(combine, 'combine', bool)
        util.check_attr_type(ratios, 'ratios', None, list)
        # Attr ---------------------------------------------------------
        self._arms: int = arms
        self._ratios: List[float] = []
        self._combine: bool = combine
        if (ratios is None):
            self._ratios = [1.0 for i in range(arms)]
        else:
            self._ratios = ratios
        # Policy -------------------------------------------------------
        for i in range(arms):
            self.add_port_policy(([i], [arms], False))
        self.add_port_policy(([i for i in range(arms)], [arms], False))
        self.add_wait_policy([i for i in range(arms)])
    # ==================================================================
    def __call__(self, domain: Domain, ports: List[int], fields: List[Field]
                 ) -> Tuple[List[int], List[Field]]:
        output_ports: List[int] = []
        output_fields: List[Field] = []
        # Check if the fields are of same types ------------------------
        compatible = True
        for i in range(1, len(fields)):
            if (fields[i].type != fields[i-1].type):
                compatible = False
                util.warning_terminal("Fields of different types can not be "
                    "combined.")
        # Combine the wave in list fields ------------------------------
        if (compatible):
            if (self._combine):
                fields[0] *= math.sqrt(self._ratios[ports[0]])
                for i in range(1, len(fields)):
                    ratio = math.sqrt(self._ratios[ports[i]])
                    fields[0].add(fields[i]*math.sqrt(self._ratios[ports[i]]))
                for i in range(len(fields)-1, 0, -1):
                    del fields[i]

                return self.output_ports([ports[0]]), [fields[0]]
            else:
                for i in range(len(fields)):
                    output_fields.append(fields[i]
                                         * math.sqrt(self._ratios[ports[i]]))

        return self.output_ports(ports), output_fields


if __name__ == "__main__":

    import optcom.utils.plot as plot
    import optcom.layout as layout
    import optcom.components.gaussian as gaussian
    import optcom.components.modulator as modulator
    import optcom.domain as domain
    from optcom.utils.utilities_user import temporal_power

    pulse_1 = gaussian.Gaussian(peak_power=[1.0], center_lambda=[1550.0])
    pulse_2 = gaussian.Gaussian(peak_power=[5.0], center_lambda=[1550.0])
    pulse_3 = gaussian.Gaussian(peak_power=[10.0], center_lambda=[976.0])
    # Dummy component to be able to test the not combining case
    pm = modulator.IdealPhaseMod()

    lt = layout.Layout()

    combiner = IdealCombiner(arms=3, combine=False, ratios=[2.0, 1.5, 0.5])

    lt.link((pulse_1[0], combiner[0]), (pulse_2[0], combiner[1]),
            (pulse_3[0], combiner[2]), (combiner[3], pm[0]))

    lt.run(pulse_1, pulse_2, pulse_3)

    plot_titles = (["Original pulses",
                    "Pulses coming out of the {} with 'comibne=False'"
                    .format(default_name)])
    plot_groups = [0,0,0,1]
    plot_labels = ['port 0', 'port 1', 'port 2', None]

    fields = [temporal_power(pulse_1.fields[0].channels),
              temporal_power(pulse_2.fields[0].channels),
              temporal_power(pulse_3.fields[0].channels),
              temporal_power(pm.fields[1].channels)]

    times = [pulse_1.fields[0].time, pulse_2.fields[0].time,
             pulse_3.fields[0].time, pm.fields[1].time]

    lt.reset()
    pm = modulator.IdealPhaseMod()

    pulse_1 = gaussian.Gaussian(peak_power=[1.0], center_lambda=[1550.0])
    pulse_2 = gaussian.Gaussian(channels=2, peak_power=[5.0],
                                center_lambda=[1550.0])
    pulse_3 = gaussian.Gaussian(peak_power=[10.0], center_lambda=[976.0])

    combiner = IdealCombiner(arms=3, combine=True)

    lt.link((pulse_1[0], combiner[0]), (pulse_2[0], combiner[1]),
            (pulse_3[0], combiner[2]), (combiner[3], pm[0]))

    lt.run(pulse_1, pulse_2, pulse_3)

    plot_titles.extend(["Original pulses",
                        "Pulses coming out of the {} with 'combine=True'"
                        .format(default_name)])
    plot_groups.extend([2,2,2,3])
    plot_labels.extend(['port 0', 'port 1', 'port 2', None])

    fields.extend([temporal_power(pulse_1.fields[0].channels),
                   temporal_power(pulse_2.fields[0].channels),
                   temporal_power(pulse_3.fields[0].channels),
                   temporal_power(pm.fields[1].channels)])

    times.extend([pulse_1.fields[0].time, pulse_2.fields[0].time,
                  pulse_3.fields[0].time, pm.fields[1].time])


    plot.plot(times, fields, plot_labels=plot_labels,
              plot_groups=plot_groups, plot_titles=plot_titles, x_labels=['t'],
              y_labels=['P_t'], opacity=0.3)
