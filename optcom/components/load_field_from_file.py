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
from typing import List, Optional, Sequence, Tuple, Union

import pickle

import optcom.config as cfg
import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.components.abstract_start_comp import AbstractStartComp
from optcom.components.abstract_start_comp import call_decorator
from optcom.domain import Domain
from optcom.field import Field


default_name = 'Field File Loader'


# Exceptions
class LoadFieldFromFileError(Exception):
    pass

class FileError(LoadFieldFromFileError):
    pass

class DataTypeError(LoadFieldFromFileError):
    pass


class LoadFieldFromFile(AbstractStartComp):
    r"""This component allows one to load fields from a file which
    contains list of Field objects as a file line.

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
    list_index_to_load :
        The indices of the fields from the loaded Field list to
        propagate. If the index is out of range, this index is
        ignore. If the list is empty, will propagate all fields.
    raise_type_error :
        If True, raise an error if the type of the loaded object
        from file is not a list of Field. If False, ignore the object.

    Notes
    -----

    Component diagram::

        __________________ [0]

    """

    _nbr_instances: int = 0
    _nbr_instances_with_default_name: int = 0

    def __init__(self, name: str = default_name,
                 file_name: str = 'fields_saved',
                 list_index_to_load: List[int] = [],
                 raise_type_error: bool = True, save: bool = False,
                 pre_call_code: str = '', post_call_code: str = '') -> None:
        r"""
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
        list_index_to_load :
            The indices of the fields from the loaded Field list to
            propagate. If the index is out of range, this index is
            ignore. If the list is empty, will propagate all fields.
        raise_type_error :
            If True, raise an error if the type of the loaded object
            from file is not a list of Field. If False, ignore the
            object.
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
        ports_type = [cst.ANY_OUT]
        super().__init__(name, default_name, ports_type, save,
                         pre_call_code=pre_call_code,
                         post_call_code=post_call_code)
        # Attr types check ---------------------------------------------
        util.check_attr_type(file_name, 'file_name', str)
        util.check_attr_type(list_index_to_load, 'list_index_to_load', list)
        util.check_attr_type(raise_type_error, 'raise_type_error', bool)
        # Attr ---------------------------------------------------------
        self.file_name: str = file_name
        if (len(self.file_name.split('.')) < 2):
            self.file_name += '.' + cfg.FILE_EXT
        self.list_index_to_load: List[int] = list_index_to_load
        self.raise_type_error: bool = raise_type_error
    # ==================================================================
    @call_decorator
    def __call__(self, domain: Domain) -> Tuple[List[int], List[Field]]:

        output_ports: List[int] = []
        output_fields: List[Field] = []
        if (not os.path.isfile(self.file_name)):

            raise FileError("The specified file '{}' has not been found, "
                            "please verify that the file exists."
                            .format(self.file_name))

        else:
            loading_failed = False
            # Open file
            file_container = open(self.file_name, 'rb')
            while (not loading_failed):
                try:
                    new_fields = pickle.load(file_container)
                except:
                    loading_failed = True
                else:
                    # The loaded object must be a list of fields
                    accepted_fields: List[Field] = []
                    i: int = 0
                    right_type = True if isinstance(new_fields,list) else False
                    if (right_type):
                        while (right_type and i < len(new_fields)):
                            accepted: bool = True
                            if (not isinstance(new_fields[i], Field)):
                                right_type = False
                            if (self.list_index_to_load # To eval first
                                    and not (i in self.list_index_to_load)):
                                accepted = False
                            if (accepted):
                                accepted_fields.append(new_fields[i])
                            i += 1
                    if (right_type):
                        output_fields.extend(accepted_fields)
                    else:
                        if (self.raise_type_error):

                            raise DataTypeError("The object loaded from file "
                                                "'{}' is not a list of Field, "
                                                "can not be loaded."
                                                .format(self.file_name))

            util.print_terminal("{} field(s) loaded from file '{}'"
                                .format(len(output_fields), self.file_name))
            # Close file
            file_container.close()
            output_ports = [0 for i in range(len(output_fields))]

        return output_ports, output_fields


if __name__ == "__main__":
    """Give an example of LoadFieldFromFile usage.
    This piece of code is standalone, i.e. can be used in a separate
    file as an example.
    """

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

    lt_: oc.Layout = oc.Layout()
    load_field: oc.LoadFieldFromFile = oc.LoadFieldFromFile(file_name=file_name)

    lt_.run(load_field)
    fields: List[oc.Field] = load_field[0].fields

    x_datas: List[np.ndarray] = [fields[0].time, fields[1].time,
                                 fields[0].nu, fields[1].nu]
    y_datas: List[np.ndarray] = [oc.temporal_power(fields[0].channels),
                                 oc.temporal_power(fields[1].channels),
                                 oc.spectral_power(fields[0].channels),
                                 oc.spectral_power(fields[1].channels)]

    oc.plot2d(x_datas, y_datas, x_labels=["t", "t", "nu","nu"],
              y_labels=["P_t", "P_t", "P_nu", "P_nu"],
              plot_titles=["Gaussian pulse which has been saved and loaded"],
              plot_groups=[0,0,1,1])
