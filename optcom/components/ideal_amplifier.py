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

import cmath
from typing import Callable, List, Optional, Sequence, Tuple, Union, overload

import numpy as np

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.components.abstract_pass_comp import AbstractPassComp
from optcom.components.abstract_pass_comp import call_decorator
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
    gain : float
        The gain of the amplifier. :math:`[dB]`
    peak_power : float
        The target peak power to reach. (gain will be ignored if target
        power is provided) :math:`[W]`
    NOISE :
        If True, the noise is handled, otherwise is unchanged.

    Notes
    -----
    Component diagram::

        [0] __________________ [1]

    """

    _nbr_instances: int = 0
    _nbr_instances_with_default_name: int = 0

    def __init__(self, name: str = default_name,
                 gain: Union[float, Callable] = 1.0,
                 peak_power: Optional[float] = None, NOISE: bool = True,
                 save: bool = False, max_nbr_pass: Optional[List[int]] = None,
                 pre_call_code: str = '', post_call_code: str = '') -> None:
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
        NOISE :
            If True, the noise is handled, otherwise is unchanged.
            (will be ignored if peak poewr is provided). If callable
            gain is provided and noise enabled, take value at mid domain
            time point.
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
        util.check_attr_type(gain, 'gain', int, float, Callable)
        util.check_attr_type(peak_power, 'peak_power', None, int, float)
        util.check_attr_type(NOISE, 'NOISE', bool)
        # Attr ---------------------------------------------------------
        # Use cmath to allow complex expo
        self.gain: Union[float, Callable] = gain
        self.peak_power: Optional[float] = peak_power
        self.NOISE = NOISE
        # Policy -------------------------------------------------------
        self.add_port_policy(([0],[1],True))
    # ==================================================================
    @call_decorator
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
            gain: np.ndarray = np.ones_like(domain.time, dtype=complex)
            gain_noise: float = 0.0
            for field in fields:
                for i in range(len(field)):
                    gain = np.ones_like(domain.time, dtype=complex)
                    if (callable(self.gain)):
                        for j in range(len(field.time[i])):
                            gain[j] = self.gain(field.time[i][j])
                    else:
                        gain *= self.gain
                    field[i] *= np.sqrt(util.db_to_linear(gain))
                if (self.NOISE):
                    if (callable(self.gain)):
                        gain_noise = self.gain(domain.time[domain.samples//2])
                    else:
                        gain_noise = self.gain
                    fields[i].noise *= util.db_to_linear(gain_noise)
                output_fields.append(fields[i])

        return self.output_ports(ports), output_fields


if __name__ == "__main__":

    from typing import Callable, List, Optional

    import numpy as np

    import optcom as oc

    lt: oc.Layout = oc.Layout()

    pulse: oc.Gaussian = oc.Gaussian(channels=1, peak_power=[10.0])
    gain: float = 3.0
    amp: oc.IdealAmplifier = oc.IdealAmplifier(gain=gain)
    lt.add_link(pulse[0], amp[0])
    lt.run(pulse)
    plot_titles: List[str] = (["Original pulse", "Pulses coming out of the "
                               "ideal amplifier with gain {} dB"
                               .format(gain)])
    y_datas: List[np.ndarray] = [oc.temporal_power(pulse[0][0].channels),
                                 oc.temporal_power(amp[1][0].channels)]
    x_datas: List[np.ndarray] = [pulse[0][0].time, amp[1][0].time]

    pulse = oc.Gaussian(channels=1, peak_power=[10.0])
    peak_power: float = 15.0
    amp = oc.IdealAmplifier(gain=5.6, peak_power=peak_power)
    lt.reset()
    lt.add_link(pulse[0], amp[0])
    lt.run(pulse)
    plot_titles.extend(["Pulses coming out of the ideal amplifier with target "
                        "peak power {} W".format(peak_power)])
    y_datas.extend([oc.temporal_power(amp[1][0].channels)])
    x_datas.extend([amp[1][0].time])

    pulse = oc.Gaussian(channels=1, peak_power=[10.0])
    gain_fct: Callable = lambda t: (0.001 * t**2 + 1.0)
    amp = oc.IdealAmplifier(gain=gain_fct)
    lt.reset()
    lt.add_link(pulse[0], amp[0])
    lt.run(pulse)
    plot_titles.extend([r"Pulses coming out of the ideal amplifier with gain: "
                        r"$f(t) = (0.001*t^2) + 1.0$"])
    y_datas.extend([oc.temporal_power(amp[1][0].channels)])
    x_datas.extend([amp[1][0].time])

    plot_groups: List[int] = [0,1,2,3]

    oc.plot2d(x_datas, y_datas, plot_groups=plot_groups,
              plot_titles=plot_titles, x_labels=['t'], y_labels=['P_t'],
              line_opacities=[0.3])
