import pytest
import numpy as np
from numpy.testing.utils import assert_array_almost_equal, assert_array_equal

import optcom.config as cfg
import optcom.utils.constants as cst
from optcom.components.gaussian import Gaussian
from optcom.components.fiber import Fiber
from optcom.domain import Domain
from optcom.field import Field
from optcom.layout import Layout


temporal_power = Field.temporal_power
spectral_power = Field.spectral_power


nlse_methods = ["ssfm", "ssfm_reduced", "ssfm_symmetric",
                "ssfm_opti_reduced", "ssfm_super_sym",
                "ssfm_opti_super_sym", "rk4ip"]

# ----------------------------------------------------------------------
# Fixtures -------------------------------------------------------------
# ----------------------------------------------------------------------

@pytest.fixture
def gssn_fiber_layout():
    """Return temporal and spectral power arrays from a layout
    composed of a gaussian initial pulse and a fiber for several
    nlse methods.

    Notes
    -----
    Test case::

    gssn ___ fiber

    """
    def layout(channels, peak_powers, widths, center_lambdas):
        domain = Domain(bit_width=100.0, samples_per_bit=2048)
        lt = Layout(domain)
        dtime = domain.dtime
        out_temporal_power = []
        out_spectral_power = []
        gssn = Gaussian(channels=channels, peak_power=peak_powers,
                        width=widths, center_lambda=center_lambdas)
        for nlse_method in nlse_methods:
            fiber = Fiber(length=0.2, nlse_method=nlse_method, alpha=[0.5],
                          beta_order=3, nl_approx=False, ATT=True, DISP=True,
                          SPM=True, XPM=True, SS=True, RS=True, steps=1000,
                          save=True)
            lt.link((gssn[0], fiber[0]))
            lt.run(gssn)
            lt.reset()
            out_temporal_power.append(temporal_power(fiber[1][0].channels))
            out_spectral_power.append(spectral_power(fiber[1][0].channels))

        return (out_temporal_power, out_spectral_power)

    return layout


@pytest.fixture
def gssn_fiber_gnlse():
    """Return temporal and spectral power arrays from a layout
    composed of a gaussian initial pulse and a fiber for gnlse equation
    with and without optimized rk4ip method.

    Notes
    -----
    Test case::

    gssn ___ fiber

    """
    def layout(channels, peak_powers, widths, center_lambdas):
        domain = Domain(bit_width=100.0, samples_per_bit=2048)
        lt = Layout(domain)
        dtime = domain.dtime
        out_temporal_power = []
        out_spectral_power = []
        gssn = Gaussian(channels=channels, peak_power=peak_powers,
                        width=widths, center_lambda=center_lambdas)
        for i in range(2):
            if (i):
                cfg.RK4IP_OPTI_GNLSE = True
            else:
                cfg.RK4IP_OPTI_GNLSE = False
            fiber = Fiber(length=0.2, nlse_method='rk4ip', alpha=[0.5],
                          beta_order=3, nl_approx=False, ATT=True, DISP=True,
                          SPM=True, XPM=True, SS=True, RS=True, steps=1000,
                          save=True)
            lt.link((gssn[0], fiber[0]))
            lt.run(gssn)
            lt.reset()
            out_temporal_power.append(temporal_power(fiber[1][0].channels))
            out_spectral_power.append(spectral_power(fiber[1][0].channels))

        return (out_temporal_power, out_spectral_power)

    return layout

# ----------------------------------------------------------------------
# Tests ----------------------------------------------------------------
# ----------------------------------------------------------------------

@pytest.mark.solvers
def test_all_solvers_single_pulse(gssn_fiber_layout):
    """Should fail if the output pulses are not almost equal."""
    outputs = gssn_fiber_layout(1, [0.5], [0.5], [1030.0])
    temporal_powers = outputs[0]
    spectral_powers = outputs[1]
    for i in range(1, len(temporal_powers)):
        assert_array_almost_equal(temporal_powers[i-1], temporal_powers[i], 5)
        assert_array_almost_equal(spectral_powers[i-1], spectral_powers[i], 5)


@pytest.mark.solvers
def test_all_solvers_multi_pulse_single_lambda(gssn_fiber_layout):
    """Should fail if the output pulses are not almost equal."""
    outputs = gssn_fiber_layout(2, [1.0, 0.5], [0.5, 0.8], [1030.0])
    temporal_powers = outputs[0]
    spectral_powers = outputs[1]
    for i in range(1, len(temporal_powers)):
        assert_array_almost_equal(temporal_powers[i-1], temporal_powers[i], 3)
        assert_array_almost_equal(spectral_powers[i-1], spectral_powers[i], 3)


@pytest.mark.solvers
def test_all_solvers_multi_pulse_multi_lambda(gssn_fiber_layout):
    """Should fail if the output pulses are not almost equal."""
    outputs = gssn_fiber_layout(2, [1.0, 0.5], [0.5, 0.8], [1030.0, 1028.0])
    temporal_powers = outputs[0]
    spectral_powers = outputs[1]
    for i in range(1, len(temporal_powers)):
        assert_array_almost_equal(temporal_powers[i-1], temporal_powers[i], 3)
        assert_array_almost_equal(spectral_powers[i-1], spectral_powers[i], 3)


@pytest.mark.solvers
def test_rk4ip_vs_opti_rk4ip_single_pulse(gssn_fiber_gnlse):
    """Should fail if the output pulses are not (almost) equal.
    Should be exactly equal, but noise induced by ifft(fft(.)) calculs.
    """
    outputs = gssn_fiber_gnlse(1, [1.0], [0.5], [1030.0])
    temporal_powers = outputs[0]
    spectral_powers = outputs[1]
    for i in range(1, len(temporal_powers)):
        assert_array_almost_equal(temporal_powers[i-1], temporal_powers[i], 15)
        assert_array_almost_equal(spectral_powers[i-1], spectral_powers[i], 15)


@pytest.mark.solvers
def test_rk4ip_vs_opti_rk4ip_multi_pulse_single_lambda(gssn_fiber_gnlse):
    """Should fail if the output pulses are not (almost) equal.
    Should be exactly equal, but noise induced by ifft(fft(.)) calculs.
    """
    outputs = gssn_fiber_gnlse(2, [1.0, 0.5], [0.5, 0.8], [1030.0])
    temporal_powers = outputs[0]
    spectral_powers = outputs[1]
    for i in range(1, len(temporal_powers)):
        assert_array_almost_equal(temporal_powers[i-1], temporal_powers[i], 3)
        assert_array_almost_equal(spectral_powers[i-1], spectral_powers[i], 3)


@pytest.mark.solvers
def test_rk4ip_vs_opti_rk4ip_multi_pulse_multi_lambda(gssn_fiber_gnlse):
    """Should fail if the output pulses are not (almost) equal.
    Should be exactly equal, but noise induced by ifft(fft(.)) calculs.
    """
    outputs = gssn_fiber_gnlse(2, [1.0, 0.5], [0.5, 0.8], [1030.0, 1031.5])
    temporal_powers = outputs[0]
    spectral_powers = outputs[1]
    for i in range(1, len(temporal_powers)):
        assert_array_almost_equal(temporal_powers[i-1], temporal_powers[i], 3)
        assert_array_almost_equal(spectral_powers[i-1], spectral_powers[i], 3)
