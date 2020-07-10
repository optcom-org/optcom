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

from typing import Callable, List, Optional, Sequence, Tuple, Union
import numpy as np

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.components.abstract_pass_comp import AbstractPassComp
from optcom.components.abstract_pass_comp import call_decorator
from optcom.domain import Domain
from optcom.field import Field
from optcom.utils.fft import FFT


TYPE_WINDOW = Callable[[np.ndarray, float, float, float], np.ndarray]
UNITY_WINDOW = lambda nu, center_nu, nu_bw, nu_offset: np.ones_like(nu)

default_name = 'Generic Filter'


class GenericFilter(AbstractPassComp):
    r"""A generic filter with an user specified window. The window must
    accept the following parameters: (nu, center_nu, nu_bw, nu_offset).
    By chance of the provided window
    :math:`f_w(\nu, \nu_c, \Delta_nu, nu_{offset})`, the following
    operation is performed:

    .. math:: A_{out} = \mathcal{F}^{-1}\Big\{
                        f_w(\nu, \nu_c, \Delta_nu, nu_{offset})
                        \mathcal{F}\{A_{in}\}\Big\}


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
    window :
        The filter window, must accept (nu, center_nu, nu_bw, nu_offset)
        as argument and return a ndarray of the same size as the
        input frequency array size.
    center_nu :
        The center frequency. :math:`[ps^{-1}]`
    nu_bw : float
        The frequency spectral bandwidth. :math:`[ps^{-1}]`
    nu_offset : float
        The offset frequency. :math:`[ps^{-1}]`
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
                 window: TYPE_WINDOW = UNITY_WINDOW,
                 center_nu: float = cst.DEF_NU, nu_bw: float = 1.0,
                 nu_offset: float = 0.0, NOISE: bool = True,
                 save: bool = False, max_nbr_pass: Optional[List[int]] = None,
                 pre_call_code: str = '', post_call_code: str = '') -> None:
        """
        Parameters
        ----------
        name :
            The name of the component.
        window :
            The filter window, must accept (nu, center_nu, nu_bw, nu_offset)
            as argument and return a ndarray of the same size as the
            input frequency array size.
        center_nu :
            The center frequency. :math:`[ps^{-1}]`
        nu_bw :
            The spectral bandwidth. :math:`[ps^{-1}]`  Correspond to
            the FWHM of the Flat Top window.
        nu_offset :
            The offset frequency. :math:`[ps^{-1}]`
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
        util.check_attr_type(window, 'window', Callable)
        util.check_attr_type(center_nu, 'center_nu', float)
        util.check_attr_type(nu_bw, 'nu_bw', float)
        util.check_attr_type(nu_offset, 'nu_offset', float)
        util.check_attr_type(NOISE, 'NOISE', bool)
        # Attr ---------------------------------------------------------
        self.window = window
        self.center_nu = center_nu
        self.nu_bw = nu_bw
        self.nu_offset = nu_offset
        self.NOISE = NOISE
        # Policy -------------------------------------------------------
        self.add_port_policy(([0],[1],True))
    # ==================================================================
    @call_decorator
    def __call__(self, domain: Domain, ports: List[int], fields: List[Field]
                 ) -> Tuple[List[int], List[Field]]:

        output_fields: List[Field] = []

        for field in fields:
            # Channels
            for i in range(len(field)):
                window = np.zeros(field[i].shape, dtype=complex)
                window = self.window(field.nu[i], self.center_nu, self.nu_bw,
                                     self.nu_offset)
                window_shift = FFT.ifftshift(window)
                field[i] = FFT.ifft(window_shift * FFT.fft(field[i]))
            # Noise
            if (self.NOISE):
                field.noise *= Field.temporal_power(self.window(
                                    domain.noise_nu, self.center_nu,
                                    self.nu_bw, self.nu_offset))
            output_fields.append(field)

        return self.output_ports(ports), output_fields


if __name__ == "__main__":
    """Give an example of GenericFilter usage.
    This piece of code is standalone, i.e. can be used in a separate
    file as an example.
    """

    from typing import Callable, List, Optional

    import numpy as np

    import optcom as oc

    center_lambda = 1030.
    center_nu = oc.lambda_to_nu(center_lambda)
    lambda_bw = 2.0
    nu_bw = oc.lambda_bw_to_nu_bw(lambda_bw, center_lambda)
    # Apply on pulse and plot
    bit_width = 1000.
    domain = oc.Domain(samples_per_bit=2**13, bit_width=bit_width,
                       noise_samples=int(1e3),
                       noise_range=(center_lambda-1.0, center_lambda+1.0))
    lt: oc.Layout = oc.Layout(domain)
    lambda_bw = 0.05 # nm
    nu_bw = oc.lambda_bw_to_nu_bw(lambda_bw, center_lambda)
    pulse: oc.Gaussian = oc.Gaussian(channels=2, peak_power=[10.0, 19.0],
                                     width=[10., 6.],
                                     center_lambda=[center_lambda],
                                     noise=100*np.ones(domain.noise_samples))
    filter: oc.GenericFilter = oc.GenericFilter(nu_bw=nu_bw, nu_offset=0.,
                                                center_nu=center_nu)
    lt.add_link(pulse[0], filter[0])
    lt.run(pulse)
    plot_titles: List[str] = ["Original pulse", r"After flat top filter with "
                              "frequency bandwidth {} THz."
                              .format(round(nu_bw,2))]
    plot_titles += plot_titles
    y_datas: List[np.ndarray] = [oc.temporal_power(pulse[0][0].channels),
                                 oc.temporal_power(filter[1][0].channels),
                                 oc.spectral_power(pulse[0][0].channels),
                                 oc.spectral_power(filter[1][0].channels),
                                 pulse[0][0].noise, filter[1][0].noise]
    x_datas: List[np.ndarray] = [pulse[0][0].time, filter[1][0].time,
                                 pulse[0][0].nu, filter[1][0].nu,
                                 pulse[0][0].domain.noise_omega,
                                 filter[1][0].domain.noise_omega]
    x_labels: List[str] = ['t', 't', 'nu', 'nu', 'nu', 'nu']
    y_labels: List[str] = ['P_t', 'P_t', 'P_nu', 'P_nu', 'P (W)', 'P (W)']
    nu_range = (center_nu-.1, center_nu+.1)
    time_range = (bit_width/2.+75., bit_width/2.-75.)
    noise_range = (x_datas[-1][0], x_datas[-1][-1])
    x_ranges = [time_range, time_range, nu_range, nu_range, noise_range]

    oc.plot2d(x_datas, y_datas, plot_titles=plot_titles, x_labels=x_labels,
              y_labels=y_labels, split=True, line_opacities=[0.3],
              x_ranges=x_ranges)
