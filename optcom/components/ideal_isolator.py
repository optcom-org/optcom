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

from typing import List, Optional, Tuple, Union

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.components.abstract_pass_comp import AbstractPassComp
from optcom.components.abstract_pass_comp import call_decorator
from optcom.domain import Domain
from optcom.field import Field

default_name = 'Ideal Isolator'


class IdealIsolator(AbstractPassComp):
    r"""An ideal Isolator.

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
    blocked_port : int
        The port id through which fields will not pass.

    Notes
    -----
    Component diagram::

        [0] __________________ [1]

    """

    _nbr_instances: int = 0
    _nbr_instances_with_default_name: int = 0

    def __init__(self, name: str = default_name, blocked_port: int = 0,
                 save: bool = False, max_nbr_pass: Optional[List[int]] = None,
                 pre_call_code: str = '', post_call_code: str = '') -> None:
        r"""
        Parameters
        ----------
        name :
            The name of the component.
        blocked_port :
            The port id through which fields will not pass.
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
        util.check_attr_type(blocked_port, 'blocked_port', int)
        # Attr range check ---------------------------------------------
        util.check_attr_range(blocked_port, 'blocked_port', 0, 1)
        # Attr ---------------------------------------------------------
        self.blocked_port: int = blocked_port
        # Policy -------------------------------------------------------
        self.add_port_policy(([0],[1],True))
    # ==================================================================
    @call_decorator
    def __call__(self, domain: Domain, ports: List[int], fields: List[Field]
                 ) -> Tuple[List[int], List[Field]]:

        output_fields: List[Field] = []
        output_ports: List[int] = []
        for i in range(len(ports)):
            if (ports[i] != self.blocked_port):
                output_fields.append(fields[i])
                output_ports.extend(self.output_ports([ports[i]]))

        return output_ports, output_fields


if __name__ == "__main__":
    """Give an example of IdealIsolator usage.
    This piece of code is standalone, i.e. can be used in a separate
    file as an example.
    """

    from typing import List, Optional

    import numpy as np

    import optcom.utils.plot as plot
    from optcom.components.gaussian import Gaussian
    from optcom.components.ideal_amplifier import IdealAmplifier
    from optcom.components.ideal_isolator import IdealIsolator
    from optcom.domain import Domain
    from optcom.layout import Layout
    from optcom.utils.utilities_user import temporal_power, spectral_power,\
                                            temporal_phase, spectral_phase

    lt: Layout = Layout()

    pulse: Gaussian = Gaussian(channels=1, peak_power=[10.0])
    isolator_1: IdealIsolator = IdealIsolator(blocked_port=1, save=True)

    lt.link((pulse[0], isolator_1[0]))

    lt.run(pulse)

    plot_titles: List[str] = (['Initial Pulse',
                               'Output of first isolator (pass)'])

    y_datas: List[np.ndarray] = [temporal_power(pulse[0][0].channels),
                                 temporal_power(isolator_1[1][0].channels)]
    x_datas: List[np.ndarray] = [pulse[0][0].time, isolator_1[1][0].time]

    plot.plot2d(x_datas, y_datas, split=True, plot_titles=plot_titles,
                x_labels=['t'], y_labels=['P_t'], opacity=[0.3])