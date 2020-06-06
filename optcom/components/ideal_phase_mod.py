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
from optcom.components.abstract_pass_comp import call_decorator
from optcom.domain import Domain
from optcom.field import Field

default_name = 'Ideal Phase Modulator'


class IdealPhaseMod(AbstractPassComp):
    """An ideal phase Modulator.

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
    phase_shift : float
        The phase_shift induced by the modulator.

    Notes
    -----
    Component diagram::

        [0] __________________ [1]

    """

    _nbr_instances: int = 0
    _nbr_instances_with_default_name: int = 0

    def __init__(self, name: str = default_name,
                 phase_shift: Union[float, Callable] = cst.PI,
                 save: bool = False, max_nbr_pass: Optional[List[int]] = None,
                 pre_call_code: str = '', post_call_code: str = '') -> None:
        r"""
        Parameters
        ----------
        name :
            The name of the component.
        phase_shift :
            The phase_shift induced by the modulator. Can be a callable
            with time variable. :math:`[ps]`
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
        ports_type = [cst.ANY_ALL, cst.ANY_ALL]
        super().__init__(name, default_name, ports_type, save,
                         max_nbr_pass=max_nbr_pass,
                         pre_call_code=pre_call_code,
                         post_call_code=post_call_code)
        # Attr types check ---------------------------------------------
        util.check_attr_type(phase_shift, 'phase_shift', float, Callable)
        # Attr ---------------------------------------------------------
        self.phase_shift: Union[float, Callable] = phase_shift
        # Policy -------------------------------------------------------
        self.add_port_policy(([0],[1], True))
    # ==================================================================
    @call_decorator
    def __call__(self, domain: Domain, ports: List[int], fields: List[Field]
                 ) -> Tuple[List[int], List[Field]]:

        output_fields: List[Field] = []
        # Need cmath for complex expo
        phase_shift: np.ndarray = np.zeros_like(domain.time, dtype=complex)
        phase_shift_: float = 0.0
        for i in range(len(domain.time)):
            if (callable(self.phase_shift)):
                phase_shift_ = self.phase_shift(domain.time[i])
            else:
                phase_shift_ = self.phase_shift
            phase_shift[i] = cmath.exp(1j*phase_shift_)
        for i in range(len(fields)):
            output_fields.append(fields[i] * phase_shift)

        return self.output_ports(ports), output_fields


if __name__ == "__main__":
    """Give an example of IdealPhaseMod usage.
    This piece of code is standalone, i.e. can be used in a separate
    file as an example.
    """

    from typing import Callable, List, Optional, Union

    import numpy as np

    import optcom as oc

    lt: oc.Layout = oc.Layout()
    chirps: List[float] = [0.0, 1.0]
    phase_shifts: List[Union[float, Callable]] = [oc.PI, lambda t: oc.PI/2]
    y_datas_t: List[np.ndarray] = []
    y_datas_nu: List[np.ndarray] = []

    plot_groups: List[int] = []
    plot_titles: List[str] = []
    plot_labels: List[Optional[str]] = []

    pulse: oc.Gaussian
    mod: oc.IdealPhaseMod
    count: int = 0
    for i, chirp in enumerate(chirps):
        for j, phase_shift in enumerate(phase_shifts):
            # Propagation
            pulse = oc.Gaussian(peak_power=[10.0], chirp=[chirp])
            mod = oc.IdealPhaseMod(phase_shift=phase_shift)
            lt.link((pulse[0], mod[0]))
            lt.run(pulse)
            lt.reset()
            # Plot parameters and get waves
            y_datas_t.append(oc.temporal_phase(pulse[0][0].channels))
            y_datas_t.append(oc.temporal_phase(mod[1][0].channels))
            y_datas_nu.append(oc.spectral_phase(pulse[0][0].channels))
            y_datas_nu.append(oc.spectral_phase(mod[1][0].channels))
            plot_groups += [count, count]
            count += 1
            plot_labels += ["Original pulse", "Exit pulse"]
            if (isinstance(phase_shift, float)):
                temp_phase = phase_shift
            else:
                temp_phase = phase_shift(0)
            plot_titles += ["Pulses through the ideal phase modulator with "
                            "chirp {} and phase shift {}"
                            .format(str(chirp), str(round(temp_phase,2)))]

    x_datas_t: List[np.ndarray] = [pulse[0][0].time, mod[1][0].time]

    oc.plot2d(x_datas_t, y_datas_t, plot_groups=plot_groups,
              plot_titles=plot_titles, plot_labels=plot_labels,
              x_labels=['t'], y_labels=['phi_t'], opacity=[0.3])

    x_datas_nu: List[np.ndarray] = [pulse[0][0].nu, mod[1][0].nu]

    oc.plot2d(x_datas_nu, y_datas_nu, plot_groups=plot_groups,
              plot_titles=plot_titles, plot_labels=plot_labels,
              x_labels=['nu'], y_labels=['phi_nu'], opacity=[0.3])
