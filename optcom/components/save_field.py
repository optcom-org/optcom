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

from typing import List, Tuple

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.components.abstract_pass_comp import AbstractPassComp
from optcom.components.abstract_pass_comp import call_decorator
from optcom.domain import Domain
from optcom.field import Field


default_name = 'Field Saver'


class SaveField(AbstractPassComp):
    """This component allows one to save one or multiple field, can be
    access via fields attribute.

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
    fields : list of Field
        List of fields which have been saved.

    Notes
    -----
    Component diagram::

        [0] __________________

    """

    _nbr_instances: int = 0
    _nbr_instances_with_default_name: int = 0

    def __init__(self, name: str = default_name, save: bool = False,
                 pre_call_code: str = '', post_call_code: str = '') -> None:
        """
        Parameters
        ----------
        name :
            The name of the component.
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
        ports_type = [cst.ANY_IN]
        super().__init__(name, default_name, ports_type, save,
                         pre_call_code=pre_call_code,
                         post_call_code=post_call_code)
        # Attr ---------------------------------------------------------
        self.fields: List[Field] = []
    # ==================================================================
    def output_ports(self, input_ports: List[int]) -> List[int]:

        return []
    # ==================================================================
    @call_decorator
    def __call__(self, domain: Domain, ports: List[int], fields: List[Field]
                 ) -> Tuple[List[int], List[Field]]:

        for field in fields:
            self.fields.append(field)

        return self.output_ports(ports), []


if __name__ == "__main__":
    """Give an example of LoadField usage.
    This piece of code is standalone, i.e. can be used in a separate
    file as an example.
    """

    from typing import List

    import numpy as np

    import optcom as oc

    lt: oc.Layout = oc.Layout()
    gssn_1: oc.Gaussian = oc.Gaussian(channels=1, width=[5.0],
                                      field_name='field 1 to be saved in file')
    field_saver_1: oc.SaveField = oc.SaveField()
    gssn_2: oc.Gaussian = oc.Gaussian(channels=1, width=[10.0],
                                      field_name='field 2 to be saved in file')
    field_saver_2: oc.SaveField = oc.SaveField()

    lt.add_links((gssn_1[0], field_saver_1[0]), (gssn_2[0], field_saver_2[0]))
    lt.run(gssn_1, gssn_2)

    fields: List[oc.Field] = field_saver_1.fields + field_saver_2.fields

    x_datas: List[np.ndarray] = [fields[0].time, fields[1].time,
                                 fields[0].nu, fields[1].nu]
    y_datas: List[np.ndarray] = [oc.temporal_power(fields[0].channels),
                                 oc.temporal_power(fields[1].channels),
                                 oc.spectral_power(fields[0].channels),
                                 oc.spectral_power(fields[1].channels)]

    oc.plot2d(x_datas, y_datas, x_labels=["t", "t", "nu","nu"],
              y_labels=["P_t", "P_t", "P_nu", "P_nu"],
              plot_titles=["Gaussian pulse which has been saved"],
              plot_groups=[0,0,1,1])
