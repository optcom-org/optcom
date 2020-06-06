# This This file is part of Optcom.
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
