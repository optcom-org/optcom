import pytest

from optcom.domain import Domain
from optcom.layout import Layout
from optcom.components.ideal_combiner import IdealCombiner
from optcom.components.gaussian import Gaussian
from optcom.components.ideal_amplifier import IdealAmplifier


# ----------------------------------------------------------------------
# Tests ----------------------------------------------------------------
# ----------------------------------------------------------------------

@pytest.mark.constraint
def test_constraint_coprop():
    r"""Should fail if the copropagating fields are not waiting for each
    others.

    Notes
    -----
    Test case::

        [0]   _________
        [1]   __________\
        [2]   ___________\__ Combiner ___ Dummy Comp __ check output
        [3]   ___________/
            ...
        [n-1] _________/

    """

    lt = Layout()
    nbr_sig = 5
    combiner = IdealCombiner(arms=nbr_sig, combine=False)
    for i in range(nbr_sig):
        lt.link((Gaussian()[0], combiner[i]))
    dummy_comp = IdealAmplifier(save=True)
    lt.link((combiner[nbr_sig], dummy_comp[0]))
    lt.run_all()

    assert (len(dummy_comp[1]) == nbr_sig)
