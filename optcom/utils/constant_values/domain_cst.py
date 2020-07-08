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

DEF_LAMBDA: float = 1552.0 # nm
DEF_NU: float = 193.16  # Thz

# domain boundaries (range)
MAX_BITS: int = 4096
MIN_BITS: int = 1

MAX_SAMPLES_PER_BIT: int = int(pow(2,18))
MIN_SAMPLES_PER_BIT: int = 1

MAX_BIT_WIDTH: float = 10000.0
MIN_BIT_WIDTH: float = 0.01
