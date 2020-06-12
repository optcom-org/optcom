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
import copy

import numpy as np
from typing import List, Optional, Sequence, Tuple

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.components.abstract_start_comp import AbstractStartComp
from optcom.components.abstract_start_comp import call_decorator
from optcom.domain import Domain
from optcom.field import Field


default_name = 'CW'


class CW(AbstractStartComp):
    r"""A CW pulse Generator.

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
    channels : int
        The number of channels in the field.
    center_lambda : list of float
        The center wavelength of the channels. :math:`[nm]`
    peak_power : list of float
        Peak power of the pulses. :math:`[W]`
    energy :
        Total power of the pulses. :math:`[J]` (peak_power will be
        ignored if energy provided)
    offset_nu : list of float
        The offset frequency. :math:`[THz]`
    init_phi : list of float
        The nitial phase of the pulses.
    field_name :
        The name of the field.

    Notes
    -----

    .. math:: A(0,t) = \sqrt{P_0} \exp\Big[i\big(\phi_0 - 2\pi (\nu_c
                                            +\nu_{offset})t\big)\Big]

    Component diagram::

        __________________ [0]

    """

    _nbr_instances: int = 0
    _nbr_instances_with_default_name: int = 0

    def __init__(self, name: str = default_name, channels: int = 1,
                 center_lambda: List[float] = [cst.DEF_LAMBDA],
                 peak_power: List[float] = [1e-3], energy: List[float] = [],
                 offset_nu: List[float] = [0.0], init_phi: List[float] = [0.0],
                 field_name: str = '', save: bool = False,
                 pre_call_code: str = '', post_call_code: str = '') -> None:
        r"""
        Parameters
        ----------
        name :
            The name of the component.
        channels :
            The number of channels in the field.
        center_lambda :
            The center wavelength of the channels. :math:`[nm]`
        peak_power :
            Peak power of the pulses. :math:`[W]`
        energy :
            Total power of the pulses. :math:`[J]` (peak_power will be
            ignored if energy provided)
        offset_nu :
            The offset frequency. :math:`[THz]`
        init_phi :
            The initial phase of the pulses.
        field_name :
            The name of the field.
        save :
            If True, the last wave to enter/exit a port will be saved.
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
        ports_type = [cst.OPTI_OUT]
        super().__init__(name, default_name, ports_type, save,
                         pre_call_code=pre_call_code,
                         post_call_code=post_call_code)
        # Attr types check ---------------------------------------------
        util.check_attr_type(channels, 'channels', int)
        util.check_attr_type(center_lambda, 'center_lambda', float, list)
        util.check_attr_type(peak_power, 'peak_power', float, list)
        util.check_attr_type(energy, 'energy', None, float, list)
        util.check_attr_type(offset_nu, 'offset_nu', float, list)
        util.check_attr_type(init_phi, 'init_phi', float, list)
        util.check_attr_type(field_name, 'field_name', str)
        # Attr ---------------------------------------------------------
        self.channels: int = channels
        self.center_lambda: List[float] = util.make_list(center_lambda,
                                                         channels)
        self.peak_power: List[float] = util.make_list(peak_power, channels)
        self._energy: List[float] = []
        self.energy = energy
        self.offset_nu: List[float] = util.make_list(offset_nu, channels)
        self.init_phi: List[float] = util.make_list(init_phi, channels)
        self.field_name: str = field_name
    # ==================================================================
    @property
    def energy(self) -> List[float]:

        return self._energy
    # ------------------------------------------------------------------
    @energy.setter
    def energy(self, energy: List[float]) -> None:
        if (energy):   # Not empty list
            self._energy =  util.make_list(energy, self.channels)
        else:
            self._energy = energy
    # ==================================================================
    @call_decorator
    def __call__(self, domain: Domain) -> Tuple[List[int], List[Field]]:

        output_ports: List[int] = []
        output_fields: List[Field] = []
        field = Field(domain, cst.OPTI, self.field_name)
        # Check offset -------------------------------------------------
        for i in range(len(self.offset_nu)):
            if (abs(self.offset_nu[i]) > domain.nu_window):
                self.offset_nu[i] = 0.0
                util.warning_terminal("The offset of channel {} in component "
                    "{} is bigger than half the frequency window, offset will "
                    "be ignored.".format(str(i), self.name))
        # Field initialization -----------------------------------------
        if (self.energy):
            peak_power: List[float] = []
            time_window = domain.time_window * 1e-12 # ps -> s
            for i in range(len(self.energy)):
                peak_power.append(self.energy[i] / time_window)
        else:
            peak_power = self.peak_power
        rep_freq = np.nan
        for i in range(self.channels):   # Nbr of channels
            res = np.zeros(domain.time.shape, dtype=cst.NPFT)
            phi = (self.init_phi[i]
                   - Domain.nu_to_omega(self.offset_nu[i])*domain.time)
            res += math.sqrt(peak_power[i]) * np.exp(1j*phi)
            field.add_channel(res,
                              Domain.lambda_to_omega(self.center_lambda[i]),
                              rep_freq)

        output_fields.append(field)
        output_ports.append(0)

        return output_ports, output_fields


if __name__ == "__main__":
    """Give an example of CW usage.
    This piece of code is standalone, i.e. can be used in a separate
    file as an example.
    """

    from typing import List

    import numpy as np

    import optcom as oc

    dm: oc.Domain = oc.Domain(samples_per_bit=512, bit_width=100.0)
    lt: oc.Layout = oc.Layout(dm)

    channels: int = 1
    center_lambda: List[float] = [1552.0, 1549.0, 1596.0]
    peak_power: List[float] = [1e-3, 2e-3, 6e-3]
    offset_nu: List[float] = [0.0, 1.56, -1.6]
    init_phi: List[float] = [1.0, 1.0, 0.0]

    cw: oc.CW = oc.CW(channels=channels, center_lambda=center_lambda,
                      peak_power=peak_power, offset_nu=offset_nu,
                      init_phi=init_phi, save=True)

    lt.run(cw)

    plot_titles: List[str] = ["CW pulse temporal power",
                              "CW pulse spectral power",
                              "CW pulse temporal phase",
                              "CW pulse spectral phase"]
    x_datas: List[np.ndarray] = [cw[0][0].time, cw[0][0].nu, cw[0][0].time,
                                 cw[0][0].nu]
    y_datas: List[np.ndarray] = [oc.temporal_power(cw[0][0].channels),
                                 oc.spectral_power(cw[0][0].channels),
                                 oc.temporal_phase(cw[0][0].channels),
                                 oc.spectral_phase(cw[0][0].channels)]

    oc.plot2d(x_datas, y_datas, x_labels=["t","nu","t","nu"],
              y_labels=["P_t", "P_nu", "phi_t", "phi_nu"],
              plot_titles=plot_titles, split=True, line_opacities=[0.2])
