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

import cmath

from typing import Callable, List, Optional, Sequence, Tuple, Union

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.components.abstract_pass_comp import AbstractPassComp
from optcom.domain import Domain
from optcom.field import Field


default_name = 'Ideal Amplifier'


class IdealAmplifier(AbstractPassComp):
    """An ideal Amplifier.

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
    gain : float
        The gain of the amplifier. :math:`[dB]`
    peak_power : float
        The target peak power to reach. (gain will be ignored if target
        power is provided) :math:`[W]`

    Notes
    -----
    Component diagram::

        [0] __________________ [1]

    """

    _nbr_instances: int = 0
    _nbr_instances_with_default_name: int = 0

    def __init__(self, name: str = default_name,
                 gain: Union[float, Callable] = 1.0,
                 peak_power: Optional[float] = None, save: bool = False
                 ) -> None:
        """
        Parameters
        ----------
        name :
            The name of the component.
        gain :
            The gain of the amplifier. :math:`[dB]`
            Can be a callable with time variable. :math:`[ps]`
            (will be ignored if peak power is provided))
        peak_power :
            The target peak power to reach. :math:`[W]`
        save :
            If True, the last wave to enter/exit a port will be saved.

        """
        # Parent constructor -------------------------------------------
        ports_type = [cst.ANY_ALL, cst.ANY_ALL]
        super().__init__(name, default_name, ports_type, save)
        # Attr types check ---------------------------------------------
        util.check_attr_type(gain, 'gain', int, float, Callable)
        util.check_attr_type(peak_power, 'peak_power', None, int, float)
        # Attr ---------------------------------------------------------
        # Use cmath to allow complex expo
        self.gain: Callable
        if (isinstance(gain, float)):
            self.gain = lambda t: gain
        else:
            self.gain = gain
        self.peak_power: Optional[float] = peak_power
        # Policy -------------------------------------------------------
        self.add_port_policy(([0],[1],True))
    # ==================================================================
    def __call__(self, domain: Domain, ports: List[int], fields: List[Field]
                 ) -> Tuple[List[int], List[Field]]:

        output_fields: List[Field] = []
        if (self.peak_power is not None):
            for field in fields:
                for i in range(len(field)):
                    max_power = np.square(np.amax(field[i]))
                    field[i] = field[i] * np.sqrt(self.peak_power/max_power)
                output_fields.append(field)
        else:
            gain = np.zeros_like(domain.time, dtype=complex)
            for i in range(len(domain.time)):
                gain[i] = cmath.sqrt(10**(0.1*self.gain(domain.time[i])))
            for i in range(len(fields)):
                output_fields.append(fields[i] * gain)

        return self.output_ports(ports), output_fields


if __name__ == "__main__":

    import numpy as np
    import optcom.utils.plot as plot
    import optcom.layout as layout
    import optcom.components.gaussian as gaussian
    import optcom.domain as domain
    from optcom.utils.utilities_user import temporal_power

    lt = layout.Layout()

    pulse = gaussian.Gaussian(channels=1, peak_power=[10.0])
    gain = 3.0
    amp = IdealAmplifier(gain=gain)
    lt.link((pulse[0], amp[0]))
    lt.run(pulse)
    plot_titles = (["Original pulse", "Pulses coming out of the {} with gain "
                    "{} dB."
                    .format(default_name, gain)])
    fields = [temporal_power(pulse.fields[0].channels),
              temporal_power(amp.fields[1].channels)]
    times = [pulse.fields[0].time, amp.fields[1].time]

    pulse = gaussian.Gaussian(channels=1, peak_power=[10.0])
    peak_power = 15.0
    amp = IdealAmplifier(gain=5.6, peak_power=peak_power)
    lt.reset()
    lt.link((pulse[0], amp[0]))
    lt.run(pulse)
    plot_titles.extend(["Pulses coming out of the {} with target peak power "
                        "{} W.".format(default_name, peak_power)])
    fields.extend([temporal_power(amp.fields[1].channels)])
    times.extend([amp.fields[1].time])

    pulse = gaussian.Gaussian(channels=1, peak_power=[10.0])
    gain_fct = lambda t: 1e-1*t
    amp = IdealAmplifier(gain=gain_fct)
    lt.reset()
    lt.link((pulse[0], amp[0]))
    lt.run(pulse)
    plot_titles.extend(["Pulses coming out of the {} with gain: f(t) = t."
                        .format(default_name)])
    fields.extend([temporal_power(amp.fields[1].channels)])
    times.extend([amp.fields[1].time])

    plot_groups = [0,1,2,3]

    plot.plot(times, fields, plot_groups=plot_groups, plot_titles=plot_titles,
              x_labels=['t'], y_labels=['P_t'], opacity=0.3)
