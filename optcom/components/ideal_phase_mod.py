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

# Optcom's development comments:
#
# - Make target phase shift
# - make electro mod with driving voltage and bias voltage


import numpy as np
import cmath

from typing import Callable, List, Optional, Sequence, Tuple, Union

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.components.abstract_pass_comp import AbstractPassComp
from optcom.domain import Domain
from optcom.field import Field

default_name = 'Ideal Phase Modulator'


class IdealPhaseMod(AbstractPassComp):
    """An ideal phase Modulator

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
    phase_shift : float
        The phase_shift induced by the modulator.
    phase : float
        The target phase to reach. (phase_shift will be ignored if
        target is provided)

    Notes
    -----
    Component diagram::

        [0] __________________ [1]

    """

    _nbr_instances: int = 0
    _nbr_instances_with_default_name: int = 0

    def __init__(self, name: str = default_name,
                 phase_shift: Union[float, Callable] = cst.PI,
                 phase: Optional[float] = None, save: bool = False) -> None:
        r"""
        Parameters
        ----------
        name :
            The name of the component.
        phase_shift :
            The phase_shift induced by the modulator. Can be a callable
            with time variable. :math:`[ps]`
        phase :
            The target phase to reach. (phase_shift will be ignored if
            target is provided)
        save :
            If True, the last wave to enter/exit a port will be saved.

        """
        # Parent constructor -------------------------------------------
        ports_type = [cst.ANY_ALL, cst.ANY_ALL]
        super().__init__(name, default_name, ports_type, save)
        # Attr types check ---------------------------------------------
        util.check_attr_type(phase_shift, 'phase_shift', float, Callable)
        util.check_attr_type(phase, 'phase', None, float)
        # Attr ---------------------------------------------------------
        self.phase_shift: Callable
        if (isinstance(phase_shift, float)):
            self.phase_shift = lambda t: phase_shift
        else:
            self.phase_shift = phase_shift
        self.phase: Optional[float] = phase
        # Policy -------------------------------------------------------
        self.add_port_policy(([0],[1], True))
    # ==================================================================
    def __call__(self, domain: Domain, ports: List[int], fields: List[Field]
                 ) -> Tuple[List[int], List[Field]]:

        output_fields: List[Field] = []
        if (self.phase is not None):
            pass
            # To do
        else:
            # Need cmath for complex expo
            phase_shift = np.zeros_like(domain.time, dtype=complex)
            for i in range(len(domain.time)):
                phase_shift[i] = cmath.exp(1j*self.phase_shift(domain.time[i]))
            for i in range(len(fields)):
                output_fields.append(fields[i] * phase_shift)

        return self.output_ports(ports), output_fields


if __name__ == "__main__":

    import optcom.utils.plot as plot
    import optcom.layout as layout
    import optcom.components.gaussian as gaussian
    import optcom.domain as domain
    from optcom.utils.utilities_user import phase

    lt = layout.Layout()

    chirps = [0.0, 1.0]
    phase_shifts = [cst.PI, lambda t: cst.PI/2]
    fields = []

    plot_groups: List[int] = []
    plot_titles: List[str] = []
    plot_labels: List[str] = []

    count = 0
    for i, chirp in enumerate(chirps):
        for j, phase_shift in enumerate(phase_shifts):
            # Propagation
            pulse = gaussian.Gaussian(peak_power=[10.0], chirp=[chirp])
            mod = IdealPhaseMod(phase_shift=phase_shift)
            lt.link((pulse[0], mod[0]))
            lt.run(pulse)
            lt.reset()
            # Plot parameters and get waves
            fields.append(phase(pulse.fields[0].channels))
            fields.append(phase(mod.fields[1].channels))
            plot_groups += [count, count]
            count += 1
            plot_labels += ["Original pulse", "Exit pulse"]
            if (isinstance(phase_shift, float)):
                temp_phase = phase_shift
            else:
                temp_phase = phase_shift(0)
            plot_titles += ["Pulses through the {} with chirp {} and phase "
                            "shift {}".format(default_name, str(chirp),
                                              str(round(temp_phase,2)))]

    time = [pulse.fields[0].time, mod.fields[1].time]

    plot.plot(time, fields,
              plot_groups=plot_groups, plot_titles=plot_titles,
              plot_labels=plot_labels,
              x_labels=['t'], y_labels=['phi'], opacity=0.3)
