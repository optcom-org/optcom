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

from typing import Callable, List, Optional, Sequence, Tuple, Union

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.components.ideal_amplifier import IdealAmplifier
from optcom.components.abstract_pass_comp import AbstractPassComp
from optcom.components.ideal_combiner import IdealCombiner
from optcom.components.ideal_divider import IdealDivider
from optcom.components.ideal_phase_mod import IdealPhaseMod
from optcom.domain import Domain
from optcom.field import Field


default_name = 'Ideal MZ Modulator'


class IdealMZ(AbstractPassComp):
    r"""An ideal Mach Zehnder Modulator

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

    .. math::  \phi_k(t)= \pi \frac{V_{mod,k}(t)+V_{bias,k}}{V_{\pi,k}}
               \quad k\in\{1,2\}

    Component diagram::

                  _______
        [0] _____/       \______ [1]
                 \_______/

    """

    _nbr_instances: int = 0
    _nbr_instances_with_default_name: int = 0

    def __init__(self, name: str = default_name,
                 phase_shift: Union[List[float], List[Callable]] = [0.0, 0.0],
                 loss: float = 0.0, ext_ratio: float = 0.0,
                 v_pi: Optional[List[float]] = None,
                 v_bias: Optional[List[float]] = None,
                 v_mod: Optional[List[Callable]] = None,
                 save: bool = False) -> None:
        r"""
        Parameters
        ----------
        name :
            The name of the component.
        phase_shift :
            The phase difference induced between the two arms of the MZ.
            Can be a list of callable with time variable. :math:`[ps]`
            (will be ignored if (v_pi and v_bias) or (v_pi and v_mod)
            are provided)
        loss :
            The loss induced by the MZ. :math:`[dB]`
        ext_ratio :
            The extinction ratio.
        v_pi :
            The half-wave voltage. :math:`[V]`
        v_bias :
            The bias voltage. :math:`[V]`
        v_mod :
            The modulation voltage :math:`[V]`. Must be a callable with
            time variable. :math:`[ps]`
        save :
            If True, the last wave to enter/exit a port will be saved.

        """
        # Parent constructor -------------------------------------------
        ports_type = [cst.ANY_ALL, cst.ANY_ALL]
        super().__init__(name, default_name, ports_type, save)
        # Attr types check ---------------------------------------------
        util.check_attr_type(phase_shift, 'phase_shift', float, Callable, list)
        util.check_attr_type(loss, 'loss', float)
        util.check_attr_type(ext_ratio, 'ext_ratio', float)
        util.check_attr_type(v_pi, 'v_pi', None, float, list)
        util.check_attr_type(v_bias, 'v_bias', None, float, list)
        util.check_attr_type(v_mod, 'v_mod', None, Callable, list)
        # Attr ---------------------------------------------------------
        if (v_pi is not None and (v_bias is not None or v_mod is not None)):
            pi_ = util.make_list(v_pi, 2)
            bias_ = util.make_list(v_bias, 2) if v_bias is not None\
                        else [0.0, 0.0]
            mod_ = util.make_list(v_mod, 2) if v_mod is not None\
                        else [lambda t: 0.0, lambda t: 0.0]
            phase_shift_ = [lambda t: cst.PI * (bias_[0]+mod_[0](t)) / pi_[0],
                            lambda t: cst.PI * (bias_[1]+mod_[1](t)) / pi_[1]]
        else:
            phase_shift_ = util.make_list(phase_shift, 2, 0.0)
        # N.B. name='nocount' to avoid inc. default name counter
        self._divider = IdealDivider(name='nocount', arms=2, divide=True,
                                     ratios=[0.5, 0.5])
        self._combiner = IdealCombiner(name='nocount', arms=2, combine=True,
                                       ratios=[0.5, 0.5])
        self._phasemod_1 = IdealPhaseMod(name='nocount',
                                         phase_shift=phase_shift_[0])
        self._phasemod_2 = IdealPhaseMod(name='nocount',
                                         phase_shift=phase_shift_[1])
        self._amp = IdealAmplifier(name='nocount', gain=-loss)
        # Policy -------------------------------------------------------
        self.add_port_policy(([0],[1], True))
    # ==================================================================
    def __call__(self, domain: Domain, ports: List[int], fields: List[Field]
                 ) -> Tuple[List[int], List[Field]]:

        output_fields: List[Field] = []
        fields_: List[Field] = [] # Temp var
        for i in range(len(fields)):
            fields_ = self._divider(domain, [0], [fields[i]])[1]
            fields_[0] = self._phasemod_1(domain, [0], [fields_[0]])[1][0]
            fields_[1] = self._phasemod_2(domain, [0], [fields_[1]])[1][0]
            output_fields.append(self._combiner(domain, [0,1], fields_)[1][0])
            output_fields[-1] = self._amp(domain, [0],
                                          [output_fields[-1]])[1][0]

        return self.output_ports(ports), output_fields


if __name__ == "__main__":

    import math

    import optcom.utils.plot as plot
    import optcom.layout as layout
    import optcom.components.gaussian as gaussian
    import optcom.domain as domain
    from optcom.utils.utilities_user import temporal_power
    from random import random

    pulse = gaussian.Gaussian(peak_power=[30.0])

    lt = layout.Layout()

    loss = 0.0
    random_phase = random() * math.pi
    random_phase_bis = random() * math.pi
    phase_shifts = [[random_phase, random_phase], [math.pi/2,0.0],
                    [random_phase, random_phase_bis]]
    fields = []

    plot_titles = ["Original pulse"]

    for i, phase_shift in enumerate(phase_shifts):
        # Propagation
        mz = IdealMZ(phase_shift=phase_shift, loss=loss)
        lt.link((pulse[0], mz[0]))
        lt.run(pulse)
        lt.reset()
        # Plot parameters and get waves
        fields.append(temporal_power(mz.fields[1].channels))
        if (isinstance(phase_shift[0], float)):
            temp_phase = phase_shift
        else:
            temp_phase = [phase_shift[0](0), phase_shift[1](0)]
        plot_titles += ["Pulses coming out of the {} with phase "
                        "shift {} and {}"
                        .format(default_name, str(round(temp_phase[0], 2)),
                                str(round(temp_phase[1], 2)))]

    v_pi = [1.0]
    v_mod = [lambda t: math.sin(math.pi*t), lambda t: math.sin(math.pi/2.0*t)]
    v_bias = [1.2, 2.1]
    mz = IdealMZ(v_pi=v_pi, v_mod=v_mod, v_bias=v_bias)
    lt.link((pulse[0], mz[0]))
    lt.run(pulse)
    # Plot parameters and get waves
    fields.append(temporal_power(mz.fields[1].channels))
    plot_titles += ["Pulses coming out of the {}".format(default_name)]

    fields  = [temporal_power(pulse.fields[0].channels)] + fields
    time = [pulse.fields[0].time, mz.fields[1].time]

    plot.plot(time, fields, split=True, plot_titles=plot_titles,
              x_labels=['t'], y_labels=['P_t'], opacity=0.3)
