import pytest
import numpy as np
from numpy.testing.utils import assert_array_almost_equal, assert_array_less

import optcom.utils.constants as cst
from optcom.components.gaussian import Gaussian
from optcom.components.fiber import Fiber
from optcom.domain import Domain
from optcom.field import Field
from optcom.layout import Layout


temporal_power = Field.temporal_power
spectral_power = Field.spectral_power
fwhm = Field.fwhm

# ----------------------------------------------------------------------
# Fixtures -------------------------------------------------------------
# ----------------------------------------------------------------------

@pytest.fixture
def fiber_layout():
    """Return temporal and spectral power and fwhm arrays from a layout
    composed of a gaussian initial pulse and a fiber.

    Notes
    -----
    Test case::

    starter ___ fiber

    """
    def layout(starter, ATT, DISP, SPM, SS, RS, approx):
        domain = Domain()
        lt = Layout(domain)
        dtime = domain.dtime
        domega = domain.domega
        out_temporal_power = []
        out_spectral_power = []
        out_temporal_fwhm = []
        out_spectral_fwhm = []
        nlse_method = "ssfm_symmetric"
        flag = True
        till_nbr = 4 if approx else 5
        for i in range(1,till_nbr):
            if (i==4):
                flag = False
            fiber = Fiber(length=1.0, nlse_method=nlse_method, alpha=[0.046],
                          nl_approx=flag, ATT=ATT, DISP=DISP, SPM=SPM, SS=SS,
                          RS=RS, steps=1000, save=True, approx_type=i)
            lt.add_link(starter[0], fiber[0])
            lt.run(starter)
            lt.reset()
            out_temporal_power.append(temporal_power(fiber[1][0].channels))
            out_spectral_power.append(spectral_power(fiber[1][0].channels))
            out_temporal_fwhm.append(fwhm(out_temporal_power[-1], dtime))
            out_spectral_fwhm.append(fwhm(out_spectral_power[-1], domega))
        in_temporal_power = temporal_power(starter[0][0].channels)
        in_spectral_power = spectral_power(starter[0][0].channels)
        in_temporal_fwhm = fwhm(in_temporal_power, dtime)
        in_spectral_fwhm = fwhm(in_spectral_power, domega)

        return (in_temporal_power, in_spectral_power, in_temporal_fwhm,
                in_spectral_fwhm, out_temporal_power, out_spectral_power,
                out_temporal_fwhm, out_spectral_fwhm)

    return layout

# ----------------------------------------------------------------------
# Tests ----------------------------------------------------------------
# ----------------------------------------------------------------------

gssn_1_ch = Gaussian(channels=1, peak_power=[1.0])
gssn_2_ch = Gaussian(channels=2, peak_power=[1.0, 0.5])

@pytest.mark.equations
@pytest.mark.parametrize("starter, ATT, DISP, SPM, SS, RS, approx",
    [(gssn_1_ch, True, False, False, False, False, False),
     (gssn_1_ch, True, True, False, False, False, False),
     (gssn_1_ch, True, True, True, False, False, False),
     (gssn_1_ch, True, True, True, True, False, True),
     (gssn_1_ch, True, True, True, False, True, True),
     (gssn_1_ch, True, True, True, True, True, True),
     (gssn_2_ch, True, False, False, False, False, False),
     (gssn_2_ch, True, True, False, False, False, False),
     (gssn_2_ch, True, True, True, False, False, False),
     (gssn_2_ch, True, True, True, True, False, True),
     (gssn_2_ch, True, True, True, False, True, True),
     (gssn_2_ch, True, True, True, True, True, True)
    ])
def test_nlses(fiber_layout, starter, ATT, DISP, SPM, SS, RS, approx):
    """Should fail if the outputs are not equal."""
    outputs = fiber_layout(starter, ATT, DISP, SPM, SS, RS, approx)
    temporal_powers = outputs[4]
    spectral_powers = outputs[5]
    for i in range(1, len(temporal_powers)):
        assert_array_almost_equal(temporal_powers[i-1], temporal_powers[i], 5)
        assert_array_almost_equal(spectral_powers[i-1], spectral_powers[i], 5)


@pytest.mark.effects
def test_attenuation(fiber_layout):
    """Should fail if the output power not less than initial pulse."""
    outputs = fiber_layout(gssn_1_ch, True, False, False, False, False, False)
    in_temporal_power = outputs[0]
    out_temporal_power = outputs[4]
    for i in range(len(out_temporal_power)):
        assert_array_less(out_temporal_power[i], in_temporal_power)


@pytest.mark.effects
def test_dispersion(fiber_layout):
    """Should fail if the output fwhm not more than initial pulse."""
    # Make sure to have a positive GVD to pass the test
    gssn = Gaussian(channels=1, peak_power=[1.0], center_lambda=[1030.])
    outputs = fiber_layout(gssn, False, True, False, False, False, False)
    in_temporal_fwhm = outputs[2]
    out_temporal_fwhm = outputs[6]
    for i in range(len(out_temporal_fwhm)):
        assert in_temporal_fwhm[0] < out_temporal_fwhm[i][0]


@pytest.mark.effects
def test_spm(fiber_layout):
    """Should fail if the fwhm of the output spectral powers are more
    than the initial pulse."""
    # Make sure to have a positive GVD to pass the test
    gssn = Gaussian(channels=1, peak_power=[1.0], center_lambda=[1030.])
    outputs = fiber_layout(gssn, False, False, True, False, False, False)
    in_spectral_fwhm = outputs[3]
    out_spectral_fwhm = outputs[7]

    for i in range(len(out_spectral_fwhm)):
        assert in_spectral_fwhm[0] < out_spectral_fwhm[i][0]


@pytest.mark.effects
def test_interplay_dispersion_and_spm():
    """Should fail if (i) fwhm of temporal output powers for small
    N square number is greater than fwhm of initial pulse or (ii)
    fwhm of temporal output powers for big N square number is
    greater than initial fwhm in case of positive GVD and smaller
    than initial fwhm in case of negative GVD.

    Notes
    -----
    Test case::

    gssn ___ fiber

    """
    # Environment creation
    domain = Domain()
    lt = Layout(domain)
    dtime = domain.dtime
    fwhms_pos_gvd = []
    fwhms_neg_gvd = []
    time_pos_gvd = []
    time_neg_gvd = []
    nlse_method = "ssfm_symmetric"
    # Make sure to have a positive GVD to pass the test
    gssn = Gaussian(channels=1, peak_power=[1.0], center_lambda=[1030.])
    flag = True
    for i in range(1,5):
        if (i==4):
            flag = False
        fiber = Fiber(length=1.0, nlse_method=nlse_method, alpha=[0.46],
                      gamma=2.0, nl_approx=flag, ATT=False, DISP=True,
                      SPM=True, SS=False, RS=False, steps=1000, save=True,
                      approx_type=i, beta=[1e5, 1e3, 20.0])
        lt.add_link(gssn[0], fiber[0])
        lt.run(gssn)
        lt.reset()
        time_pos_gvd.append(fiber[1][0].time)
        fwhms_pos_gvd.append(fwhm(temporal_power(fiber[1][0].channels), dtime))
    flag = True
    for i in range(1,5):
        if (i==4):
            flag = False
        fiber = Fiber(length=1.0, nlse_method=nlse_method, alpha=[0.46],
                      gamma=2.0, nl_approx=flag, ATT=False, DISP=True,
                      SPM=True, SS=False, RS=False, steps=1000, save=True,
                      approx_type=i, beta=[1e5, 1e3, -20.0])
        lt.add_link(gssn[0], fiber[0])
        lt.run(gssn)
        lt.reset()
        time_neg_gvd.append(fiber[1][0].time)
        fwhms_neg_gvd.append(fwhm(temporal_power(fiber[1][0].channels), dtime))
    fwhm_temporal_gssn = fwhm(temporal_power(gssn[0][0].channels), dtime)
    time_gssn = gssn[0][0].time
    # Testing
    for i in range(len(fwhms_neg_gvd)):
        assert_array_less(time_gssn, time_pos_gvd[i])
        assert_array_less(time_gssn, time_neg_gvd[i])
        assert fwhm_temporal_gssn[0] < fwhms_pos_gvd[i][0]
        assert fwhms_neg_gvd[i][0] < fwhm_temporal_gssn[0]
