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

import math

# default value for fiber
approx_type_1: int = 1
approx_type_2: int = 2   # Keeping |A|^2
approx_type_3: int = 3   # Subst. A*conj_A instead of |A|^2
DEFAULT_APPROX_TYPE: int = approx_type_1

GAMMA: float = 1.0
RAMAN_COEFF: float = 3e-3
TAU_1: float = 12.2e-3
TAU_2: float = 32.0e-3
XPM_COEFF: float = 2.0
XPM_COEFF_CROSS: float = 0.  # Didn't find reliable source
XNL_COEFF: float = 2.0
XNL_COEFF_CROSS: float = 0.    # Didn't find reliable source
TAU_B: float = 96.0e-3
F_B: float = 0.21
F_C: float = 0.04
F_A: float = 1.0 - F_C - F_B # can be 0.75
F_R: float = 0.245 #0.18
DEF_EFF_AREA: float = 80.0   # um^2 (vary in 1-100)


# fiber amp
TAU_META: float = 840.0   # um
TAU_META_YB: float = 840.0   # um
N_T: float = 6.3e-2    # nm^-3
CORE_RADIUS: float = 5.0   # um
CLAD_RADIUS: float = 12.5  # um
CLAD_RADIUS_COUP: float = 62.5  # um
ETA_SIGNAL: float = 1.26   # km^-1
ETA_PUMP: float = 1.41     # km^-1
R_L: float = 8e-4
R_0: float = 8e-4
NL_INDEX: float = 2.6e-20    #m^2 W^-1
DOPED_AREA: float = (math.pi*CORE_RADIUS**2) # um^2
OVERLAP: float = 0.97
SIGMA_A: float = 2.4e-6 # nm^2
SIGMA_E: float = 2.6e-6 # nm^2
FIBER_GAIN: float = 0.0 # km^{-1}

# kappas
V: float = 2.0
C2C_SPACING: float = 15.0    # um
REF_INDEX: float = 1.0
REF_CORE_INDEX: float = 1.405
REF_CLAD_INDEX: float = 1.400
NA: float = 0.1

# fibers media
DEF_FIBER_MEDIUM: str = "sio2"
DEF_FIBER_DOPANT: str = "yb"
FIBER_CORE_DOPANT: str = "geo2"
FIBER_CLAD_DOPANT: str = "F"
CORE_DOPANT_CONCENT: float = 5. # mole%
CLAD_DOPANT_CONCENT: float = 1. # mole%
FIBER_MEDIUM_CORE: str = DEF_FIBER_MEDIUM
FIBER_MEDIUM_CLAD: str = DEF_FIBER_MEDIUM
