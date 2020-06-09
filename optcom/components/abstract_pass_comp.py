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

from abc import abstractmethod
from typing import List, Optional, Tuple

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.components.abstract_component import AbstractComponent
from optcom.domain import Domain
from optcom.field import Field


def call_decorator(call):
    def func_wrapper(self, domain, input_ports, input_fields):
        self.call_counter += 1
        exec(self.pre_call_code)
        output_ports, output_fields = call(self, domain, input_ports,
                                           input_fields)
        exec(self.post_call_code)

        return output_ports, output_fields

    return func_wrapper


class AbstractPassComp(AbstractComponent):

    def __init__(self, name: str, default_name: str, ports_type: List[int],
                 save: bool, wait:bool = False,
                 max_nbr_pass: Optional[List[int]] = None,
                 pre_call_code: str = '', post_call_code: str = '') -> None:

        super().__init__(name, default_name, ports_type, save, wait,
                         max_nbr_pass, pre_call_code, post_call_code)
    # ==================================================================
    @abstractmethod
    def __call__(self, domain: Domain, ports: List[int]=[],
                 fields: List[Field]=[]) -> Tuple[List[int], List[Field]]: pass
