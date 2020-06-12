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


default_name = 'Gaussian Filter'


class GaussianFilter(AbstractPassComp):
    """A Gaussian filter.

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
    nu_bw : float
        The frequency spectral bandwidth. :math:`[ps^{-1}]`
    nu_offset : float
        The offset frequency. :math:`[ps^{-1}]`
    order :
        The order of the gaussian filter.

    Notes
    -----
    Component diagram::

        [0] __________________ [1]

    """

    _nbr_instances: int = 0
    _nbr_instances_with_default_name: int = 0

    def __init__(self, name: str = default_name,
                 nu_bw: float = 1.0, nu_offset: float = 0.0, order: int = 1,
                 save: bool = False, max_nbr_pass: Optional[List[int]] = None,
                 pre_call_code: str = '', post_call_code: str = '') -> None:
        """
        Parameters
        ----------
        name :
            The name of the component.
        nu_bw :
            The frequency spectral bandwidth. :math:`[ps^{-1}]`
        nu_offset :
            The offset frequency. :math:`[ps^{-1}]`
        order :
            The order of the gaussian filter.
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
        util.check_attr_type(nu_bw, 'nu_bw', float)
        util.check_attr_type(nu_offset, 'nu_offset', float)
        util.check_attr_type(order, 'order', int)
        # Attr ---------------------------------------------------------
        self.nu_bw = nu_bw
        self.nu_offset = nu_offset
        self.order = order
        # Policy -------------------------------------------------------
        self.add_port_policy(([0],[1],True))
    # ==================================================================
    @call_decorator
    def __call__(self, domain: Domain, ports: List[int], fields: List[Field]
                 ) -> Tuple[List[int], List[Field]]:

        output_fields: List[Field] = []

        for field in fields:
            nu = domain.nu - self.nu_offset
            for i in range(len(field)):
                arg = np.zeros(nu.shape, dtype=complex)
                arg = FFT.ifftshift(np.power(nu/self.nu_bw, 2*self.order))
                field[i] = FFT.ifft(np.exp(-0.5*arg) * FFT.fft(field[i]))
            output_fields.append(field)

        return self.output_ports(ports), output_fields


if __name__ == "__main__":
    """Give an example of GaussianFilter usage.
    This piece of code is standalone, i.e. can be used in a separate
    file as an example.
    """

    from typing import Callable, List, Optional

    import numpy as np

    import optcom as oc

    lt: oc.Layout = oc.Layout(oc.Domain(bit_width=600.))

    nu_bw: float = 0.01
    pulse: oc.Gaussian = oc.Gaussian(channels=2, peak_power=[10.0, 19.0],
                                     width=[10., 6.])
    filter: oc.GaussianFilter = oc.GaussianFilter(nu_bw=nu_bw, nu_offset=0.,
                                                  order=1)
    lt.add_link(pulse[0], filter[0])
    lt.run(pulse)
    plot_titles: List[str] = ["Original pulse", r"After Gaussian filter with "
                              "frequency bandwidth {} THz.".format(nu_bw)]
    plot_titles += plot_titles
    y_datas: List[np.ndarray] = [oc.temporal_power(pulse[0][0].channels),
                                 oc.temporal_power(filter[1][0].channels),
                                 oc.spectral_power(pulse[0][0].channels),
                                 oc.spectral_power(filter[1][0].channels)]
    x_datas: List[np.ndarray] = [pulse[0][0].time, filter[1][0].time,
                                 pulse[0][0].nu, filter[1][0].nu]
    x_labels: List[str] = ['t', 't', 'nu', 'nu']
    y_labels: List[str] = ['P_t', 'P_t', 'P_nu', 'P_nu']

    oc.plot2d(x_datas, y_datas, plot_titles=plot_titles, x_labels=x_labels,
              y_labels=y_labels, split=True, line_opacities=[0.3])
