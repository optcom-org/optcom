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


default_name = 'Gaussian'


class Gaussian(AbstractStartComp):
    r"""A gaussian pulse Generator.

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
    position : list of float
        Relative position of the pulses in the time window.
        :math:`\in [0,1]`
    width : list of float
        Half width of the pulse. :math:`[ps]`
    fwhm : list of float, optional
        Full band width at half maximum. :math:`[ps]`  If fwhm is
        provided, the width will be ignored. If fwhm is not provided or
        set to None, will use the width.
    peak_power : list of float
        Peak power of the pulses. :math:`[W]`
    rep_freq :
        The repetition frequency of the pulse in the time window.
        :math:`[THz]`
    offset_nu : list of float
        The offset frequency. :math:`[THz]`
    order : list of int
        The order of the super gaussian pulse.
    chirp : list of float
        The chirp parameter for chirped pulses.
    init_phi : list of float
        The initial phase of the pulses.
    noise : np.ndarray
        The initial noise along the pulses.

    Notes
    -----

    .. math:: A(0,t) = \sqrt{P_0}
              \exp\bigg[-\frac{1+iC}{2}
              \bigg(\frac{t-t_0}{T_0}\bigg)^{2m}
              + i\big(\phi_0 - 2\pi (\nu_c  + \nu_{offset})t\big)\bigg]

    where :math:`T_0` is the half width at :math:`1/e`-intensity.

    Component diagram::

        __________________ [0]

    """

    _nbr_instances: int = 0
    _nbr_instances_with_default_name: int = 0

    def __init__(self, name: str = default_name, channels: int = 1,
                 center_lambda: List[float] = [cst.DEF_LAMBDA],
                 position: List[float] = [0.5], width: List[float] = [10.0],
                 fwhm: Optional[List[float]] = None,
                 peak_power: List[float] = [1e-3],
                 rep_freq: List[float] = [0.0], offset_nu: List[float] = [0.0],
                 order: List[int] = [1], chirp: List[float] = [0.0],
                 init_phi: List[float] = [0.0],
                 noise: Optional[np.ndarray] = None, field_name: str = '',
                 save: bool = False, pre_call_code: str = '',
                 post_call_code: str = '') -> None:
        r"""
        Parameters
        ----------
        name :
            The name of the component.
        channels :
            The number of channels in the field.
        center_lambda :
            The center wavelength of the channels. :math:`[nm]`
        position :
            Relative position of the pulses in the time window.
            :math:`\in [0,1]`
        width :
            Half width of the pulse. :math:`[ps]`
        fwhm :
            Full band width at half maximum. :math:`[ps]`  If fwhm is
            provided, the width will be ignored. If fwhm is not provided
            or set to None, will use the width.
        peak_power :
            Peak power of the pulses. :math:`[W]`
        rep_freq :
            The repetition frequency of the pulse in the time window.
            :math:`[THz]`
        offset_nu :
            The offset frequency. :math:`[THz]`
        order :
            The order of the super gaussian pulse.
        chirp :
            The chirp parameter for chirped pulses.
        init_phi :
            The initial phase of the pulses.
        noise :
            The initial noise along the pulses.
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
        util.check_attr_type(position, 'position', float, list)
        util.check_attr_type(width, 'width', float, list)
        util.check_attr_type(fwhm, 'fwhm', float, list, None)
        util.check_attr_type(peak_power, 'peak_power', float, list)
        util.check_attr_type(rep_freq, 'rep_freq', float, list)
        util.check_attr_type(offset_nu, 'offset_nu', float, list)
        util.check_attr_type(order, 'order', int, list)
        util.check_attr_type(chirp, 'chirp', float, list)
        util.check_attr_type(noise, 'noise', None, np.ndarray)
        util.check_attr_type(init_phi, 'init_phi', float, list)
        util.check_attr_type(field_name, 'field_name', str)
        # Attr ---------------------------------------------------------
        self.channels: int = channels
        self.center_lambda: List[float] = util.make_list(center_lambda,
                                                         channels)
        self.position: List[float] = util.make_list(position, channels)
        self.width: List[float] = util.make_list(width, channels)
        self._fwhm: Optional[List[float]]
        self.fwhm = fwhm
        self.peak_power: List[float] = util.make_list(peak_power, channels)
        self.rep_freq: List[float] = util.make_list(rep_freq, channels)
        self.offset_nu: List[float] = util.make_list(offset_nu, channels)
        self.order: List[int] = util.make_list(order, channels)
        self.chirp: List[float] = util.make_list(chirp, channels)
        self.init_phi: List[float] = util.make_list(init_phi, channels)
        self.noise: Optional[np.ndarray] = noise
        self.field_name: str = field_name
    # ==================================================================
    @property
    def fwhm(self) -> Optional[List[float]]:

        return self._fwhm
    # ------------------------------------------------------------------
    @fwhm.setter
    def fwhm(self, fwhm: Optional[List[float]]) -> None:
        if (fwhm is None):
            self._fwhm = None
        else:
            self._fwhm = util.make_list(fwhm, self.channels)
    # ==================================================================
    @staticmethod
    def fwhm_to_width(fwhm: float, order: int) -> float:

        return fwhm * 0.5 / math.pow(math.log(2.0), 0.5*order)
    # ==================================================================
    @staticmethod
    def width_to_fwhm(width: float, order: int) -> float:

        return width * 2.0 * math.pow(math.log(2.0), 0.5*order)
    # ==================================================================
    @call_decorator
    def __call__(self, domain: Domain) -> Tuple[List[int], List[Field]]:
        output_ports: List[int] = []
        output_fields: List[Field] = []
        field: Field = Field(domain, cst.OPTI, self.field_name)
        # Bit rate initialization --------------------------------------
        rel_pos: List[np.ndarray]
        rel_pos = util.pulse_positions_in_time_window(self.channels,
                                                      self.rep_freq,
                                                      domain.time_window,
                                                      self.position)
        # Check offset -------------------------------------------------
        for i in range(len(self.offset_nu)):
            if (abs(self.offset_nu[i]) > domain.nu_window):
                self.offset_nu[i] = 0.0
                util.warning_terminal("The offset of channel {} in component "
                    "{} is bigger than half the frequency window, offset will "
                    "be ignored.".format(str(i), self.name))
        # Field initialization -----------------------------------------
        width: float
        for i in range(self.channels):   # Nbr of channels
            if (self.fwhm is None):
                width = self.width[i]
            else:
                width = self.fwhm_to_width(self.fwhm[i], self.order[i])
            res: np.ndarray = np.zeros(domain.time.shape, dtype=cst.NPFT)
            for j in range(len(rel_pos[i])):
                norm_time = domain.get_shift_time(rel_pos[i][j]) / width
                var_time = np.power(norm_time, 2 * self.order[i])
                phi = (self.init_phi[i]
                       - Domain.nu_to_omega(self.offset_nu[i])*domain.time
                       - 0.5*self.chirp[i]*var_time)
                res += (math.sqrt(self.peak_power[i])
                        * np.exp((-0.5*var_time) + 1j*phi))
            field.add_channel(res,
                              Domain.lambda_to_omega(self.center_lambda[i]),
                              self.rep_freq[i])
            if (self.noise is not None):
                field.noise = self.noise
        output_fields.append(field)
        output_ports.append(0)

        return output_ports, output_fields


if __name__ == "__main__":
    """Give an example of Gaussian usage.
    This piece of code is standalone, i.e. can be used in a separate
    file as an example.
    """

    from typing import Callable, List, Optional

    import numpy as np

    import optcom as oc

    dm: oc.Domain = oc.Domain(samples_per_bit=512, bit_width=500.0)
    lt: oc.Layout = oc.Layout(dm)

    channels: int = 2
    center_lambda: List[float] = [1552.0, 1549.0, 1596.0]
    position: List[float] = [0.5, 0.3, 0.5]
    width: List[float] = [10.0, 5.3, 6]
    peak_power: List[float] = [1e-3, 2e-3, 6e-3]
    rep_freq: List[float] =  [0.0, 0.03, 0.04]
    offset_nu: List[float] = [0.0, 0.56, -0.6]
    order: List[int] = [1, 2, 1]
    chirp: List[float] = [0.0, 0.5, 0.1]
    init_phi: List[float] = [0.0, 1.0, 0.0]

    gssn = oc.Gaussian(channels=channels, center_lambda=center_lambda,
                       position=position, width=width, peak_power=peak_power,
                       rep_freq=rep_freq, offset_nu=offset_nu, order=order,
                       chirp=chirp, init_phi=init_phi, save=True)

    lt.run(gssn)

    x_datas: List[np.ndarray] = [gssn[0][0].time, gssn[0][0].nu]
    y_datas: List[np.ndarray] = [oc.temporal_power(gssn[0][0].channels),
                                 oc.spectral_power(gssn[0][0].channels)]

    oc.plot2d(x_datas, y_datas, x_labels=["t","nu"], y_labels=["P_t", "P_nu"],
              plot_titles=["Gaussian pulse"], split=True)
