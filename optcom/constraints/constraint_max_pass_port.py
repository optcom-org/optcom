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

import warnings
from typing import Dict, List, Optional, Tuple, Union

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.components.abstract_component import AbstractComponent
from optcom.components.port import Port
from optcom.constraints.abstract_constraint import AbstractConstraint
from optcom.constraints.abstract_constraint import ConstraintError
from optcom.constraints.abstract_constraint import ConstraintWarning
from optcom.field import Field


# Exceptions
class MaxPassPortError(ConstraintError):
    pass

class MaxPassPortWarning(ConstraintWarning):
    pass


class ConstraintMaxPassPort(AbstractConstraint):
    """A field can pass only a specific number of times through a port.
    """

    def __init__(self):

        return None
    # ==================================================================
    def is_respected(self, comp: AbstractComponent, comp_port_id: int,
                     ngbr: AbstractComponent, ngbr_port_id: int, field: Field
                     ) -> bool:
        ngbr_port: Port = ngbr[ngbr_port_id]
        ngbr_port.inc_counter_pass()
        flag: bool = ngbr_port.is_counter_below_max_pass()
        if (not flag):
            warning_message: str = ("Max number of times a field can go "
                "through port {} of component '{}' has been reached, field "
                "will be further ignored.".format(ngbr_port_id, ngbr.name))
            warnings.warn(warning_message, MaxPassPortWarning)

        return flag
