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
