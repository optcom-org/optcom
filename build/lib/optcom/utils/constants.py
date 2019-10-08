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
#NP_REAL_FIELD_TYPE = np.longdouble

#
STR_SEPARATOR_TERMINAL: str = "----------------------------------------------"\
                              "---------------------------------\n"# 79 charact

#
UNIDIR_PORT: int = -1   # Can not be a positiv integer


AUTO_PAD_PLOT: bool = True # pad the different channels array of one field
                    # automatically to display all channel on one graph
