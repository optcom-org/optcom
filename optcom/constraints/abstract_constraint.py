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
from abc import ABCMeta
from typing import Dict, List, Optional, Tuple, Union

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.components.abstract_component import AbstractComponent
from optcom.components.port import Port
from optcom.field import Field


# Exceptions
class ConstraintError(Exception):
    pass

class ConstraintWarning(UserWarning):
    pass


class AbstractConstraint(metaclass=ABCMeta):

    def __init__(self):

        return None
    # ==================================================================
    def update(self, comp: AbstractComponent, port_ids: List[int],
               fields: List[Field]) -> None:
        """Update necessary information for the constraint."""

        return None
    # ==================================================================
    def is_respected(self, comp: AbstractComponent, comp_port_id: int,
                     ngbr: AbstractComponent, ngbr_port_id: int, field: Field
                     ) -> bool:
        """Return True if the constraint is respected."""

        return True
    # ==================================================================
    def get_compliance(self, comp: AbstractComponent, comp_port_id: int,
                       ngbr: AbstractComponent, ngbr_port_id: int, field: Field
                       ) -> Tuple[List[int], List[Field]]:
        """Return the ports and fields to propagate further which
        comply with the constraint."""

        return [], []
    # ==================================================================
    def reset(self) -> None:
        """Reset parameters of the constraint."""

        return None
