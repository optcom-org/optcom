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

import copy
import math

from typing import List, Optional, Sequence, Tuple

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.components.abstract_pass_comp import AbstractPassComp
from optcom.domain import Domain
from optcom.field import Field


default_name = 'Ideal Divider'


class IdealDivider(AbstractPassComp):
    r"""An ideal Divider.

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
        A list with the dividing ratios of the corresponding
        index-number port.

    Notes
    -----
    Component diagram::

                   __________ [1]
                 /___________ [2]
        [0] ____/____________ [3]
                \____________ [4]
                 ...
                 \___________ [n]

    """

    _nbr_instances: int = 0
    _nbr_instances_with_default_name: int = 0

    def __init__(self, name: str = default_name, arms: int = 2,
                 divide: bool = True, ratios: Optional[List[float]] = None,
                 save: bool = False) -> None:
        """
        Parameters
        ----------
        name :
            The name of the component.
        arms :
            The number of input arms.
        divide :
            If False, propagate a copy of entering fields to each
            output port. Otherwise, divide power depending on
            provided ratios.
        ratios :
            A list with the dividing ratios of the corresponding
            index-number port.
        save :
            If True, the last wave to enter/exit a port will be saved.

        """
        # Parent constructor -------------------------------------------
        ports_type = [cst.OPTI_IN] + [cst.OPTI_OUT for i in range(arms)]
        super().__init__(name, default_name, ports_type, save)
        # Check types attr ---------------------------------------------
        util.check_attr_type(arms, 'arms', int)
        util.check_attr_type(divide, 'combine', bool)
        util.check_attr_type(ratios, 'ratios', None, list)
        # Attr ---------------------------------------------------------
        self._arms: int = arms
        self._ratios: List[float]
        if (divide):
            if (ratios is None):
                ratio = 1.0 / arms
                self._ratios = [ratio for i in range(arms)]
            else:
                self._ratios = ratios
        else:
            self._ratios = [1.0 for i in range(arms)]
    # ==================================================================
    def __call__(self, domain: Domain, ports: List[int], fields: List[Field]
                 ) -> Tuple[List[int], List[Field]]:

        output_ports: List[int] = []
        output_fields: List[Field] = []
        for field in fields:
            for i in range(self._arms):
                output_fields.append(copy.deepcopy(field))
                output_fields[-1] *= math.sqrt(self._ratios[i])
                output_ports.append(i+1)

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

    arms = 3
    ratios = [round(random.uniform(0,1),2) for i in range(arms)]
    divider = IdealDivider(arms=arms, ratios=ratios, save=True)

    lt.link((pulse[0], divider[0]))

    lt.run(pulse)

    plot_titles = (["Original pulse", "Pulses coming out of the {} with "
                    "ratios {}".format(default_name, str(ratios))])
    plot_groups: List[int] = [0] + [1 for i in range(arms)]
    plot_labels: List[Optional[str]] = [None]
    plot_labels.extend(["port {}".format(str(i)) for i in range(arms)])

    fields = [temporal_power(pulse.fields[0].channels)]
    times = [pulse.fields[0].time]
    for i in range(1, arms+1):
        fields.append(temporal_power(divider.fields[i].channels))
        times.append(divider.fields[i].time)


    plot.plot(times, fields, plot_groups=plot_groups, plot_titles=plot_titles,
              x_labels=['t'], y_labels=['P_t'], plot_labels=plot_labels,
              opacity=0.3)
