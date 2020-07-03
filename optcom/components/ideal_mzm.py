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

import math
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.components.ideal_amplifier import IdealAmplifier
from optcom.components.abstract_pass_comp import AbstractPassComp
from optcom.components.abstract_pass_comp import call_decorator
from optcom.components.ideal_combiner import IdealCombiner
from optcom.components.ideal_divider import IdealDivider
from optcom.components.ideal_phase_mod import IdealPhaseMod
from optcom.domain import Domain
from optcom.field import Field


default_name = 'Ideal MZ Modulator'


class IdealMZM(AbstractPassComp):
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
    phase_shift :
        The phase difference induced between the two arms of the MZ.
        Can be a list of callable with time variable. :math:`[ps]`
        (will be ignored if (v_pi and v_bias) or (v_pi and v_mod)
        are provided)
    loss :
        The loss induced by the MZ. :math:`[dB]`
    extinction :
        The extinction ratio. :math:`[dB]`
    v_pi :
        The half-wave voltage. :math:`[V]`
    v_bias :
        The bias voltage. :math:`[V]`
    v_mod :
        The modulation voltage :math:`[V]`. Must be a callable with
        time variable. :math:`[ps]`

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
                 loss: float = 0.0, extinction: Optional[float] = None,
                 v_pi: Optional[List[float]] = None,
                 v_bias: Optional[List[float]] = None,
                 v_mod: Optional[List[Callable]] = None,
                 save: bool = False, max_nbr_pass: Optional[List[int]] = None,
                 pre_call_code: str = '', post_call_code: str = '') -> None:
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
        extinction :
            The extinction ratio. :math:`[dB]`
        v_pi :
            The half-wave voltage. :math:`[V]`
        v_bias :
            The bias voltage. :math:`[V]`
        v_mod :
            The modulation voltage :math:`[V]`. Must be a callable with
            time variable. :math:`[ps]`
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
        util.check_attr_type(phase_shift, 'phase_shift', float, Callable, list)
        util.check_attr_type(loss, 'loss', float)
        util.check_attr_type(extinction, 'extinction', None, float)
        util.check_attr_type(v_pi, 'v_pi', None, float, list)
        util.check_attr_type(v_bias, 'v_bias', None, float, list)
        util.check_attr_type(v_mod, 'v_mod', None, Callable, list)
        # Attr ---------------------------------------------------------
        self._v_pi: Optional[List[float]]
        self._v_pi = v_pi if v_pi is None else util.make_list(v_pi, 2)
        self._v_bias: Optional[List[float]]
        self._v_bias = v_bias if v_bias is None else util.make_list(v_bias, 2)
        self._v_mod: Optional[List[Callable]]
        self._v_mod = v_mod if v_mod is None else util.make_list(v_mod, 2)
        if (v_pi is not None and (v_bias is not None or v_mod is not None)):
            self._update_phase_shift()
        else:
            self.phase_shift = phase_shift
        self.loss = loss
        self._extinction: Optional[float]
        self.extinction = extinction
        self._divider = IdealDivider(name='nocount', arms=2, divide=True,
                                     ratios=[0.5, 0.5])
        # Policy -------------------------------------------------------
        self.add_port_policy(([0],[1], True))
    # ==================================================================
    def _update_phase_shift(self):
        pi_ = self.v_pi
        bias_ = self.v_bias if self.v_bias is not None else [0.0, 0.0]
        null_fct = lambda t: 0.0
        mod_ = self.v_mod if self.v_mod is not None else [null_fct, null_fct]
        phase_shift_ = [lambda t: cst.PI * (bias_[0]+mod_[0](t)) / pi_[0],
                        lambda t: cst.PI * (bias_[1]+mod_[1](t)) / pi_[1]]
        self._phasemod_1 = IdealPhaseMod(name='nocount',
                                         phase_shift=phase_shift_[0])
        self._phasemod_2 = IdealPhaseMod(name='nocount',
                                         phase_shift=phase_shift_[1])
    # ==================================================================
    @property
    def loss(self) -> float:

        return self._loss
    # ------------------------------------------------------------------
    @loss.setter
    def loss(self, loss: float) -> None:
        self._loss = loss
        self._amp = IdealAmplifier(name='nocount', gain=-loss)
    # ==================================================================
    @property
    def v_pi(self) -> Optional[List[float]]:

        return self._v_pi
    # ------------------------------------------------------------------
    @v_pi.setter
    def v_pi(self, v_pi: Optional[List[float]]) -> None:
        self._v_pi = util.make_list(v_pi, 2)
        self._update_phase_shift()
    # ==================================================================
    @property
    def v_bias(self) -> Optional[List[float]]:

        return self._v_bias
    # ------------------------------------------------------------------
    @v_bias.setter
    def v_bias(self, v_bias: Optional[List[float]]) -> None:
        self._v_bias = util.make_list(v_bias, 2)
        self._update_phase_shift()
    # ==================================================================
    @property
    def v_mod(self) -> Optional[List[Callable]]:

        return self._v_mod
    # ------------------------------------------------------------------
    @v_mod.setter
    def v_mod(self, v_mod: Optional[List[Callable]]) -> None:
        self._v_mod = util.make_list(v_mod, 2)
        self._update_phase_shift()
    # ==================================================================
    @property
    def phase_shift(self) -> Union[List[float], List[Callable]]:

        return [self._phasemod_1.phase_shift, self._phasemod_2.phase_shift]
    # ------------------------------------------------------------------
    @phase_shift.setter
    def phase_shift(self, phase_shift: Union[List[float], List[Callable]]
                    ) -> None:
        phase_shift_ = util.make_list(phase_shift, 2, 0.0)
        self._phasemod_1 = IdealPhaseMod(name='nocount',
                                         phase_shift=phase_shift_[0])
        self._phasemod_2 = IdealPhaseMod(name='nocount',
                                         phase_shift=phase_shift_[1])
    # ==================================================================
    @property
    def extinction(self) -> Optional[float]:

        return self._extinction
    # ------------------------------------------------------------------
    @extinction.setter
    def extinction(self, extinction: Optional[float]) -> None:
        self._extinction = extinction
        if (extinction is None):
            gamma_er = 1.0
        else:
            extinction_ = 10**(0.1*extinction) # db -> non db
            gamma_er = (math.sqrt(extinction_)-1) / (math.sqrt(extinction_)+1)
        # N.B. name='nocount' to avoid inc. default name counter
        self._combiner = IdealCombiner(name='nocount', arms=2, combine=True,
                                       ratios=[0.5, 0.5*gamma_er])
    # ==================================================================
    @staticmethod
    def transfer_function(time: np.ndarray, v_pi: List[float],
                          v_bias: List[float],
                          v_mod: List[Union[float, Callable]]) -> np.ndarray:

        v_pi = util.make_list(v_pi, 2)
        v_bias = util.make_list(v_bias, 2)
        v_mod = util.make_list(v_mod, 2)
        v_mod_: List[Callable] = []
        for v in v_mod:
            if (callable(v)):
                v_mod_.append(v)
            else:
                v_mod_.append(lambda t: v)
        print(v_pi, v_bias, v_mod_)
        phase_shift = [lambda t: cst.PI * (v_bias[0]+v_mod_[0](t)) / v_pi[0],
                       lambda t: cst.PI * (v_bias[1]+v_mod_[1](t)) / v_pi[1]]
        tf = np.cos((phase_shift[0](time) - phase_shift[1](time))/ 2.)

        return Field.temporal_power(tf)
    # ==================================================================
    @call_decorator
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
    """Give an example of IdealIsolator usage.
    This piece of code is standalone, i.e. can be used in a separate
    file as an example.
    """

    import random
    import math
    from typing import Callable, List, Optional

    import numpy as np

    import optcom as oc

    pulse: oc.Gaussian = oc.Gaussian(peak_power=[30.0])

    lt: oc.Layout = oc.Layout()

    loss: float = 0.0
    random_phase: float = random.random() * math.pi
    random_phase_bis: float = random.random() * math.pi
    phase_shifts: List[List[float]] = [[random_phase, random_phase],
                                       [math.pi/2,0.0],
                                       [random_phase, random_phase_bis]]
    y_datas: List[np.ndarray] = []
    plot_titles: List[str] = ["Original pulse"]
    mz: oc.IdealMZM
    for i, phase_shift in enumerate(phase_shifts):
        # Propagation
        mz = oc.IdealMZM(phase_shift=phase_shift, loss=loss)
        lt.add_link(pulse[0], mz[0])
        lt.run(pulse)
        lt.reset()
        # Plot parameters and get waves
        y_datas.append(oc.temporal_power(mz[1][0].channels))
        if (isinstance(phase_shift[0], float)):
            temp_phase = phase_shift
        else:
            temp_phase = [phase_shift[0](0), phase_shift[1](0)]
        plot_titles += ["Pulses coming out of the Ideal MZM with phase "
                        "shift {} and {}"
                        .format(str(round(temp_phase[0], 2)),
                                str(round(temp_phase[1], 2)))]
    v_pi: List[float] = [1.0]
    v_mod: List[Callable] = [lambda t: math.sin(math.pi*t),
                             lambda t: math.sin(math.pi/2.0*t)]
    v_bias: List[float] = [1.2, 2.1]
    mz = oc.IdealMZM(v_pi=v_pi, v_mod=v_mod, v_bias=v_bias)
    lt.add_link(pulse[0], mz[0])
    lt.run(pulse)
    lt.reset()
    # Plot parameters and get waves
    y_datas.append(oc.temporal_power(mz[1][0].channels))
    plot_titles += ["Pulses coming out of the Ideal MZM (time dep)"]

    phase_shift_s: List[float] = [0., 0.]
    er: float = 20.0  # db
    mz = oc.IdealMZM(phase_shift=phase_shift_s, extinction=er)
    lt.add_link(pulse[0], mz[0])
    lt.run(pulse)
    # Plot parameters and get waves
    y_datas.append(oc.temporal_power(mz[1][0].channels))
    plot_titles += ["Pulses coming out of the Ideal MZM on 'on' mode with "
                    "extinction ratio {} dB".format(er)]

    y_datas = [oc.temporal_power(pulse[0][0].channels)] + y_datas
    x_datas: List[np.ndarray] = [pulse[0][0].time, mz[1][0].time]

    oc.plot2d(x_datas, y_datas, split=True, plot_titles=plot_titles,
              x_labels=['t'], y_labels=['P_t'], line_opacities=[0.3])
