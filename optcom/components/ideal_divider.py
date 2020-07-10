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

import copy
import math

from typing import List, Optional, Sequence, Tuple

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.components.abstract_pass_comp import AbstractPassComp
from optcom.components.abstract_pass_comp import call_decorator
from optcom.domain import Domain
from optcom.field import Field


default_name = 'Ideal Divider'


class IdealDivider(AbstractPassComp):
    r"""An ideal Divider.

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
    arms : int
        The number of input arms.
    divide : bool
        If False, propagate a copy of entering fields to each
        output port. (ratios will be ignored) Otherwise, divide
        power depending on provided ratios.
    ratios :
        A list of ratios where each index is related to one arm.
        The length of the list should be equal to the number of
        arms, if not it will be pad to it. The ratio represents the
        fraction of the power that will be taken from the field
        arriving at the corresponding arm. If None is provided and
        divide is set to True, dispatch equally among arms.
    NOISE :
        If True, the noise is handled, otherwise is unchanged.

    Notes
    -----
    Component diagram::

                   __________ [1]
                 /___________ [2]
        [0] ____/____________ [3]
                \____________ [4]
                 ...
                 \___________ [n]

    """

    _nbr_instances: int = 0
    _nbr_instances_with_default_name: int = 0

    def __init__(self, name: str = default_name, arms: int = 2,
                 divide: bool = True, ratios: List[float] = [],
                 NOISE: bool = False, save: bool = False,
                 max_nbr_pass: Optional[List[int]] = None,
                 pre_call_code: str = '', post_call_code: str = '') -> None:
        """
        Parameters
        ----------
        name :
            The name of the component.
        arms :
            The number of input arms.
        divide :
            If False, propagate a copy of entering fields to each
            output port. (ratios will be ignored) Otherwise, divide
            power depending on provided ratios.
        ratios :
            A list of ratios where each index is related to one arm.
            The length of the list should be equal to the number of
            arms, if not it will be pad to it. The ratio represents the
            fraction of the power that will be taken from the field
            arriving at the corresponding arm. If None is provided and
            divide is set to True, dispatch equally among arms.
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
        ports_type = [cst.OPTI_IN] + [cst.OPTI_OUT for i in range(arms)]
        super().__init__(name, default_name, ports_type, save,
                         max_nbr_pass=max_nbr_pass,
                         pre_call_code=pre_call_code,
                         post_call_code=post_call_code)
        # Check types attr ---------------------------------------------
        util.check_attr_type(arms, 'arms', int)
        util.check_attr_type(divide, 'combine', bool)
        util.check_attr_type(ratios, 'ratios', list)
        util.check_attr_type(NOISE, 'NOISE', bool)
        # Attr ---------------------------------------------------------
        self.divide: bool = divide
        self.arms: int = arms
        self.ratios: List[float]
        if (not ratios):
            self.ratios = [1.0/arms for i in range(arms)]
        else:
            self.ratios = util.make_list(ratios, arms)
        self.NOISE = NOISE
    # ==================================================================
    def output_ports(self, input_ports: List[int]) -> List[int]:

        return [(i+1) for i in range(self.arms)] * len(input_ports)
    # ==================================================================
    @call_decorator
    def __call__(self, domain: Domain, ports: List[int], fields: List[Field]
                 ) -> Tuple[List[int], List[Field]]:
        output_fields: List[Field] = []
        for field in fields:
            for i in range(self.arms):
                output_fields.append(copy.deepcopy(field))
                if (self.divide):   # Consider ratio if need to divide
                    output_fields[-1] *= math.sqrt(self.ratios[i])
                    if (self.NOISE):
                        output_fields[-1].noise *= self.ratios[i]

        return self.output_ports(ports), output_fields



if __name__ == "__main__":
    """Give an example of IdealDivider usage.
    This piece of code is standalone, i.e. can be used in a separate
    file as an example.
    """

    import random
    from typing import Callable, List, Optional

    import numpy as np

    import optcom as oc

    # with division
    pulse: oc.Gaussian = oc.Gaussian(channels=1, peak_power=[10.0])
    lt: oc.Layout = oc.Layout()
    arms: int = 3
    ratios: List[float] = [round(random.uniform(0,1),2) for i in range(arms)]
    divider: oc.IdealDivider = oc.IdealDivider(arms=arms, divide=True,
                                               ratios=ratios, save=True)
    lt.add_link(pulse[0], divider[0])
    lt.run(pulse)
    y_datas: List[np.ndarray] = [oc.temporal_power(pulse[0][0].channels)]
    x_datas: List[np.ndarray] = [pulse[0][0].time]

    plot_titles: List[str] = (["Original pulse", "Pulses coming out of the "
                               "ideal divider (3 ports) \n with ratios {}"
                               .format(str(ratios))])
    plot_groups: List[int] = [0] + [1 for i in range(arms)]
    line_labels: List[Optional[str]] = [None]
    line_labels.extend(["port {}".format(str(i)) for i in range(arms)])

    for i in range(1, arms+1):
        y_datas.append(oc.temporal_power(divider[i][0].channels))
        x_datas.append(divider[i][0].time)
    # Without division
    arms = 3
    pulse = oc.Gaussian(channels=2, peak_power=[10.0, 7.0])
    divider = oc.IdealDivider(arms=arms, divide=False, save=True)
    lt.reset()
    lt.add_link(pulse[0], divider[0])
    lt.run(pulse)
    plot_titles.extend(["Original pulse", "Pulses coming out of the ideal "
                        "divider (3 ports) \n with no division."])
    plot_groups.extend([2] + [3 for i in range(arms)])
    line_labels.extend([None])
    line_labels.extend(["port {}".format(str(i)) for i in range(arms)])
    y_datas.extend([oc.temporal_power(pulse[0][0].channels)])
    x_datas.extend([pulse[0][0].time])
    for i in range(1, arms+1):
        y_datas.append(oc.temporal_power(divider[i][0].channels))
        x_datas.append(divider[i][0].time)

    oc.plot2d(x_datas, y_datas, plot_groups=plot_groups,
              plot_titles=plot_titles, x_labels=['t'], y_labels=['P_t'],
              line_labels=line_labels, line_opacities=[0.3])
