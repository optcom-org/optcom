import pytest

from optcom.layout import Layout
from optcom.components.ideal_coupler import IdealCoupler
from optcom.components.gaussian import Gaussian
from optcom.components.ideal_amplifier import IdealAmplifier
from optcom.constraints.constraint_max_pass_port import MaxPassPortWarning


# ----------------------------------------------------------------------
# Tests ----------------------------------------------------------------
# ----------------------------------------------------------------------

def test_constraint_max_pass():
    """Should fail if the system loop to infinity.

    Notes
    -----
    Test case::



    gssn ___      ___ null
            \____/
         ___/    \___
        |            |
        |____ mod ___|


    """
    # Environment creation
    lt = Layout()
    max_pass = 100
    cplr = IdealCoupler(save=True, max_nbr_pass=[max_pass])
    gssn = Gaussian()
    amp = IdealAmplifier()
    lt.link((gssn[0], cplr[0]), (cplr[1], amp[0]), (cplr[3], amp[1]))
    # Testing
    pytest.warns(MaxPassPortWarning, lt.run, gssn)
    assert len(cplr[3]) == max_pass
    assert len(cplr[1]) == max_pass
