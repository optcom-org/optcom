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

from typing import List, Optional, Tuple

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.components.abstract_component import AbstractComponent
from optcom.domain import Domain
from optcom.field import Field


class AbstractStartComp(AbstractComponent):

    def __init__(self, name: str, default_name: str, ports_type: List[int],
                 save: bool, wait: bool = False,
                 max_nbr_pass: Optional[List[int]] = None) -> None:

        super().__init__(name, default_name, ports_type, save, wait,
                         max_nbr_pass)
    # ==================================================================
    def __call__(self, domain: Domain) -> Tuple[List[int], List[Field]]: ...
