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
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.components.abstract_pass_comp import AbstractPassComp
from optcom.components.abstract_pass_comp import call_decorator
from optcom.domain import Domain
from optcom.field import Field
from optcom.utils.fft import FFT


default_name = 'Ideal Filter'


class IdealFilter(AbstractPassComp):
    """An ideal filter.

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
    center_nu :
        The center frequency. :math:`[ps^{-1}]`
    nu_bw : float
        The frequency spectral bandwidth. :math:`[ps^{-1}]`
    nu_offset : float
        The offset frequency. :math:`[ps^{-1}]`
    gain_in_bw :
        The spectral power gain in the specified bandwidth.
        :math:`[dB]`
    gain_out_bw :
        The spectral power gain outside of the specified bandwidth.
        :math:`[dB]`
    NOISE :
        If True, the noise is handled, otherwise is unchanged.

    Notes
    -----
    Component diagram::

        [0] __________________ [1]

    """

    _nbr_instances: int = 0
    _nbr_instances_with_default_name: int = 0

    def __init__(self, name: str = default_name, center_nu: float = cst.DEF_NU,
                 nu_bw: float = 1.0, nu_offset: float = 0.0,
                 gain_in_bw: float = -1.0, gain_out_bw: float = -50.0,
                 NOISE: bool = True, save: bool = False,
                 max_nbr_pass: Optional[List[int]] = None,
                 pre_call_code: str = '', post_call_code: str = '') -> None:
        """
        Parameters
        ----------
        name :
            The name of the component.
        center_nu :
            The center frequency. :math:`[ps^{-1}]`
        nu_bw :
            The spectral bandwidth. :math:`[ps^{-1}]`
        nu_offset :
            The offset frequency. :math:`[ps^{-1}]`
        gain_in_bw :
            The spectral power gain in the specified bandwidth.
            :math:`[dB]`
        gain_out_bw :
            The spectral power gain outside of the specified bandwidth.
            :math:`[dB]`
        NOISE :
            If True, the noise is handled, otherwise is unchanged.
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
        util.check_attr_type(center_nu, 'center_nu', float)
        util.check_attr_type(nu_bw, 'nu_bw', float)
        util.check_attr_type(nu_offset, 'nu_offset', float)
        util.check_attr_type(gain_in_bw, 'gain_in_bw', float)
        util.check_attr_type(gain_out_bw, 'gain_out_bw', float)
        util.check_attr_type(NOISE, 'NOISE', bool)
        # Attr ---------------------------------------------------------
        self.center_nu = center_nu
        self.nu_bw = nu_bw
        self.nu_offset = nu_offset
        self.gain_in_bw = gain_in_bw
        self.gain_out_bw = gain_out_bw
        self.NOISE = NOISE
        # Policy -------------------------------------------------------
        self.add_port_policy(([0],[1],True))
    # ==================================================================
    @call_decorator
    def __call__(self, domain: Domain, ports: List[int], fields: List[Field]
                 ) -> Tuple[List[int], List[Field]]:

        output_fields: List[Field] = []
        gain_in_bw_lin = cmath.sqrt(util.db_to_linear(self.gain_in_bw))
        gain_out_bw_lin = cmath.sqrt(util.db_to_linear(self.gain_out_bw))
        for field in fields:
            # Pulse
            for i in range(len(field)):
                nu: np.ndarray = field.nu[i] - self.nu_offset
                nu_gains = np.where((nu>(self.center_nu+self.nu_bw)) |
                                    (nu<(self.center_nu-self.nu_bw)),
                                    gain_out_bw_lin, gain_in_bw_lin)
                nu_gains_shift = FFT.ifftshift(nu_gains)
                field[i] = FFT.ifft(nu_gains_shift * FFT.fft(field[i]))
            # Noise
            if (self.NOISE):
                gain_in_bw_lin_ = util.db_to_linear(self.gain_in_bw)
                gain_out_bw_lin_ = util.db_to_linear(self.gain_out_bw)
                nu_noise: np.ndarray = domain.noise_nu
                nu_gains_ = np.where((nu_noise>(self.center_nu+self.nu_bw)) |
                                     (nu_noise<(self.center_nu-self.nu_bw)),
                                     gain_out_bw_lin_, gain_in_bw_lin_)
                print(nu_gains_)
                field.noise *= nu_gains_
            output_fields.append(field)

        return self.output_ports(ports), output_fields


if __name__ == "__main__":
    """Give an example of IdealFilter usage.
    This piece of code is standalone, i.e. can be used in a separate
    file as an example.
    """

    from typing import Callable, List, Optional

    import numpy as np

    import optcom as oc

    center_lambda = 1030.

    domain = oc.Domain(bit_width=600., noise_samples=int(1e3),
                       noise_range=(center_lambda-1.0, center_lambda+1.0))
    lt: oc.Layout = oc.Layout(domain)

    nu_bw: float = 0.01
    center_nu = oc.lambda_to_nu(center_lambda)
    pulse: oc.Gaussian = oc.Gaussian(channels=2, peak_power=[10.0, 19.0],
                                     width=[10., 6.],
                                     center_lambda=[center_lambda],
                                     noise=100*np.ones(domain.noise_samples))
    filter: oc.IdealFilter = oc.IdealFilter(nu_bw=nu_bw, nu_offset=0.,
                                            center_nu=center_nu, NOISE=True)
    lt.add_link(pulse[0], filter[0])
    lt.run(pulse)
    plot_titles: List[str] = ["Original pulse", r"After filter with "
                              "frequency bandwidth {} THz.".format(nu_bw)]
    plot_titles += plot_titles
    y_datas: List[np.ndarray] = [oc.temporal_power(pulse[0][0].channels),
                                 oc.temporal_power(filter[1][0].channels),
                                 oc.spectral_power(pulse[0][0].channels),
                                 oc.spectral_power(filter[1][0].channels),
                                 pulse[0][0].noise, filter[1][0].noise]
    x_datas: List[np.ndarray] = [pulse[0][0].time, filter[1][0].time,
                                 pulse[0][0].nu, filter[1][0].nu,
                                 pulse[0][0].domain.noise_nu,
                                 filter[1][0].domain.noise_nu]
    x_labels: List[str] = ['t', 't', 'nu', 'nu', 'nu', 'nu']
    y_labels: List[str] = ['P_t', 'P_t', 'P_nu', 'P_nu', 'P (W)', 'P (W)']

    oc.plot2d(x_datas, y_datas, plot_titles=plot_titles, x_labels=x_labels,
              y_labels=y_labels, split=True, line_opacities=[0.3])
