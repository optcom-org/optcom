import pytest

from optcom.domain import Domain
from optcom.layout import Layout
import optcom.utils.constants as cst
from optcom.components.abstract_pass_comp import AbstractPassComp
from optcom.components.ideal_combiner import IdealCombiner
from optcom.components.gaussian import Gaussian
from optcom.components.ideal_amplifier import IdealAmplifier
from optcom.constraints.constraint_port_valid import PortValidWarning


# ----------------------------------------------------------------------
# Tests ----------------------------------------------------------------
# ----------------------------------------------------------------------

@pytest.mark.constraint
def test_constraint_port_valid():
    r"""Should fail if the propagating field can enter the dummy_comp.

    Notes
    -----
    Test case::

        Gaussian_1 ______ Combiner

    """
    # Environment creations
    class DummyPassComp(AbstractPassComp):
        def __init__(self):
            super().__init__('', '', [cst.ELEC_IN, cst.ELEC_OUT], True)
        def __call__(self, domain, ports, fields):

            return ([1 for i in range(len(fields))], fields)

    lt = Layout()
    gssn = Gaussian()
    dummy = DummyPassComp()
    lt.link((gssn[0], dummy[0]))
    lt.run(gssn)
    # Testing
    pytest.warns(PortValidWarning, lt.run, gssn)
