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

import typing

NULL_PORT_ID = -1

# port types
OPTI_ALL: int = 1
OPTI_IN: int = 2
OPTI_OUT: int = 3
ELEC_ALL: int = 4
ELEC_IN: int = 5
ELEC_OUT: int = 6
ANY_ALL: int = 7
ANY_IN: int = 8
ANY_OUT: int = 9

IN_PORTS: typing.List[int] = [OPTI_IN, OPTI_ALL, ELEC_IN, ELEC_ALL, ANY_IN,
                              ANY_ALL]
OUT_PORTS: typing.List[int] = [OPTI_OUT, OPTI_ALL, ELEC_OUT, ELEC_ALL, ANY_OUT,
                               ANY_ALL]
OPTI_PORTS: typing.List[int] = [OPTI_IN, OPTI_OUT, OPTI_ALL]
ELEC_PORTS: typing.List[int] = [ELEC_IN, ELEC_OUT, ELEC_ALL]
ANY_PORTS: typing.List[int] = [ANY_IN, ANY_OUT, ANY_ALL]
OPTI_PORTS.extend(ANY_PORTS)
ELEC_PORTS.extend(ANY_PORTS)
ANY_PORTS.extend(OPTI_PORTS)
ANY_PORTS.extend(ELEC_PORTS)
