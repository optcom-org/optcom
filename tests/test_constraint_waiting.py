import pytest

from optcom.domain import Domain
from optcom.layout import Layout
from optcom.components.ideal_combiner import IdealCombiner
from optcom.components.gaussian import Gaussian


# ----------------------------------------------------------------------
# Tests ----------------------------------------------------------------
# ----------------------------------------------------------------------

@pytest.mark.constraint
def test_constraint_waiting():
    r"""Should fail if the component is not waiting for other fields.

    Notes
    -----
    Test case::

        [0]   _________
        [1]   __________\
        [2]   ___________\__ Combiner __ check output
        [3]   ___________/
            ...
        [n-1] _________/

    """

    lt = Layout()
    nbr_sig = 5
    combiner = IdealCombiner(arms=nbr_sig, combine=False, save=True)
    for i in range(nbr_sig):
        lt.add_link(Gaussian()[0], combiner[i])
    lt.run_all()

    assert (len(combiner[nbr_sig]) == nbr_sig)
