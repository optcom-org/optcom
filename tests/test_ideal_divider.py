import pytest
import numpy as np
from numpy.testing.utils import assert_array_almost_equal, assert_array_equal

import optcom.utils.constants as cst
from optcom.components.gaussian import Gaussian
from optcom.components.ideal_divider import IdealDivider
from optcom.domain import Domain
from optcom.field import Field
from optcom.layout import Layout

temporal_power = Field.temporal_power
spectral_power = Field.spectral_power
energy = Field.energy
fwhm = Field.fwhm


# ----------------------------------------------------------------------
# Tests ----------------------------------------------------------------
# ----------------------------------------------------------------------

@pytest.mark.component
@pytest.mark.parametrize("nbr_channels, ratios",
    [(2, [1.0, 1.0]), (6, [0.5, 0.5]), (1, [0.4, 0.6]), (8, [0.1, 0.8, 0.3])
    ])
def test_output_divider(nbr_channels, ratios):
    """Should fail if the output temporal power division does not correpsond to
    the dividing ratios.
    """
    # Environment creation
    base_power = 10.0
    gssn = Gaussian(channels=nbr_channels, save=True,
                    peak_power=[(i+1)*base_power for i in range(nbr_channels)])
    nbr_arms = len(ratios)
    divider = IdealDivider(arms=nbr_arms, ratios=ratios, save=True)
    lt = Layout()
    lt.add_link(gssn[0], divider[0])
    lt.run_all()
    # Testing
    init_power = temporal_power(gssn[0][0].channels)
    for i in range(nbr_arms):
        assert len(divider[i+1]) == 1
        arm_power = temporal_power(divider[i+1][0].channels)
        assert len(arm_power) == len(init_power)
        for j in range(len(arm_power)):
            # Taking into account rounding errors
            assert_array_almost_equal((ratios[i]*init_power[j]), arm_power[j],
                                      10)


@pytest.mark.component
def test_no_divide_output():
    """Should fail if the incoming fields are not the same as the output
    fields at each arm.
    """
    nbr_channels = 10
    arms = 5
    gssn = Gaussian(channels=nbr_channels, save=True)
    divider = IdealDivider(arms=arms, divide=False, save=True)
    lt = Layout()
    lt.add_link(gssn[0], divider[0])
    lt.run_all()
    # Testing
    init_power = temporal_power(gssn[0][0].channels)
    for i in range(arms):
        assert len(divider[i+1]) == 1
        arm_power = temporal_power(divider[i+1][0].channels)
        assert len(arm_power) == len(init_power)
        for j in range(len(arm_power)):
            assert_array_equal(arm_power[j], init_power[j])
