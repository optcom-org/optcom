import pytest
import numpy as np
from numpy.testing.utils import assert_array_equal, assert_array_almost_equal

from optcom.domain import Domain


nu_to_omega = Domain.nu_to_omega
nu_to_lambda = Domain.nu_to_lambda
omega_to_nu = Domain.omega_to_nu
omega_to_lambda = Domain.omega_to_lambda
lambda_to_nu = Domain.lambda_to_nu
lambda_to_omega = Domain.lambda_to_omega
nu_bw_to_lambda_bw = Domain.nu_bw_to_lambda_bw
lambda_bw_to_nu_bw = Domain.lambda_bw_to_nu_bw
omega_bw_to_lambda_bw = Domain.omega_bw_to_lambda_bw
lambda_bw_to_omega_bw = Domain.lambda_bw_to_omega_bw


# ----------------------------------------------------------------------
# Tests ----------------------------------------------------------------
# ----------------------------------------------------------------------

@pytest.mark.domain
def test_omega_nu_conversion():
    """Should fail if the forward and backward conversion are not equal.
    """
    array = np.arange(1., 1000., 1000, dtype=float)
    omegas = omega_to_nu(array)
    nus = nu_to_omega(omegas)
    assert_array_equal(array, nus)


def test_omega_lambda_conversion():
    """Should fail if the forward and backward conversion are not equal.
    """
    array = np.arange(1., 1000., 1000, dtype=float)
    omegas = omega_to_lambda(array)
    lambdas = lambda_to_omega(omegas)
    assert_array_equal(array, lambdas)


def test_lambda_nu_conversion():
    """Should fail if the forward and backward conversion are not equal.
    """
    array = np.arange(1., 1000., 1000, dtype=float)
    lambdas = lambda_to_nu(array)
    nus = nu_to_lambda(lambdas)
    assert_array_equal(array, nus)


def test_bw_lambda_omega():
    """Should fail if the forward and backward conversion are not equal.
    """
    array = np.arange(1., 1000., 1000, dtype=float)
    center_lambda = 500.
    center_omega = lambda_to_omega(center_lambda)
    omegas = lambda_bw_to_omega_bw(array, center_lambda)
    lambdas = omega_bw_to_lambda_bw(omegas, center_omega)
    # Almost equal to take into account rouding errors
    assert_array_almost_equal(array, lambdas, 15)


def test_bw_lambda_nu():
    """Should fail if the forward and backward conversion are not equal.
    """
    array = np.arange(1., 1000., 1000, dtype=float)
    center_lambda = 500.
    center_nu = lambda_to_nu(center_lambda)
    nus = lambda_bw_to_nu_bw(array, center_lambda)
    lambdas = nu_bw_to_lambda_bw(nus, center_nu)
    # Almost equal to take into account rouding errors
    assert_array_almost_equal(array, lambdas, 15)


def test_nested_conversion():
    """Should fail if the forward and backward conversion are not equal.
    """
    array = np.arange(1., 1000., 1000, dtype=float)
    omegas = lambda_to_omega(array)
    nus = omega_to_nu(omegas)
    lambdas = nu_to_lambda(nus)
    assert_array_equal(array, lambdas)
