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

import numpy as np

from optcom.utils.constant_values.port_types import *
from optcom.utils.constant_values.field_types import *
from optcom.utils.constant_values.domain_cst import *
from optcom.utils.constant_values.physic_cst import *
from optcom.utils.constant_values.fiber_cst import *
from optcom.utils.constant_values.solver_cst import *

FIELD_TO_PORT = {OPTI:OPTI_PORTS, ELEC:ELEC_PORTS, ANY:ANY_PORTS}

# field numpy type
NPFT = np.clongdouble
#NPFT = np.complex
#NP_REAL_FIELD_TYPE = np.longdouble



AUTO_PAD_PLOT: bool = True # pad the different channels array of one field
                    # automatically to display all channel on one graph


DEFAULT_FIELD_NAME = 'Field'
