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

import os
from typing import Callable, List, Optional, Sequence, Tuple, Union

import pickle

import optcom.config as cfg
import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.components.abstract_pass_comp import AbstractPassComp
from optcom.components.abstract_pass_comp import call_decorator
from optcom.domain import Domain
from optcom.field import Field


default_name = 'Field File Saver'


class SaveFieldToFile(AbstractPassComp):
    """This component allows one to save one or multiple field(s) into a
    file. Dump all simultaneous incoming fields to a list of fields
    as a new line in a pk1 file.

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
    file_name : str
        The name of the file in which the fields are saved.
    add_fields :
        If True and the file_name provided exists, add fields if
        fields already in file_name.

    Notes
    -----
    Component diagram::

        [0] __________________

    """

    _nbr_instances: int = 0
    _nbr_instances_with_default_name: int = 0

    def __init__(self, name: str = default_name,
                 file_name: str = 'fields_saved', add_fields: bool = True,
                 save: bool = False, pre_call_code: str = '',
                 post_call_code: str = '') -> None:
        """
        Parameters
        ----------
        name :
            The name of the component.
        file_name :
            The name of the file in which the fields are saved. Please
            specify full path to file folder if not saving in current
            directory. If no extension is provided, the default file
            extension will be considered. See :mod:`config` for default
            extension.
        add_fields :
            If True and the file_name provided exists, add fields if
            fields already in file_name.
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
        ports_type = [cst.ANY_IN]
        super().__init__(name, default_name, ports_type, save,
                         pre_call_code=pre_call_code,
                         post_call_code=post_call_code)
        # Attr types check ---------------------------------------------
        util.check_attr_type(file_name, 'file_name', str)
        util.check_attr_type(add_fields, 'add_fields', bool)
        # Attr ---------------------------------------------------------
        self.file_name: str = file_name
        if (self.file_name.split('.')[-1] != cfg.FILE_EXT):
            self.file_name += '.' + cfg.FILE_EXT
        self.add_fields: bool = add_fields
    # ==================================================================
    @call_decorator
    def __call__(self, domain: Domain, ports: List[int], fields: List[Field]
                 ) -> Tuple[List[int], List[Field]]:

        if (os.path.isfile(self.file_name) and self.add_fields):
            with open(self.file_name, 'ab') as file_container:
                pickle.dump(fields, file_container)
            util.print_terminal("{} field(s) added to existing file '{}'."
                                .format(len(fields), self.file_name))
        else:
            with open(self.file_name, 'wb') as file_container:
                pickle.dump(fields, file_container)
                util.print_terminal("{} field(s) added in new file '{}'."
                                    .format(len(fields), self.file_name))

        return [], []


if __name__ == "__main__":
    """Give an example of LoadFieldFromFile usage.
    This piece of code is standalone, i.e. can be used in a separate
    file as an example.
    """

    import pickle
    from typing import Callable, List, Optional

    import numpy as np

    import optcom as oc

    file_name: str = 'example_saved_fields.pk1'

    lt: oc.Layout = oc.Layout()
    gssn_1: oc.Gaussian = oc.Gaussian(channels=1, width=[5.0],
                                      field_name='field 1 to be saved in file')
    field_saver_1: oc.SaveFieldToFile = oc.SaveFieldToFile(file_name=file_name,
                                                           add_fields=False)
    gssn_2: oc.Gaussian = oc.Gaussian(channels=1, width=[10.0],
                                      field_name='field 2 to be saved in file')
    field_saver_2: oc.SaveFieldToFile = oc.SaveFieldToFile(file_name=file_name,
                                                           add_fields=True)

    lt.link((gssn_1[0], field_saver_1[0]), (gssn_2[0], field_saver_2[0]))
    lt.run(gssn_1, gssn_2)

    fields: List[oc.Field] = []
    with open(file_name, 'rb') as file_to_load:
        fields.append(pickle.load(file_to_load)[0])
        fields.append(pickle.load(file_to_load)[0])

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
