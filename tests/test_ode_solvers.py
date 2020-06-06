import pytest
import numpy as np
from numpy.testing.utils import assert_array_almost_equal, assert_array_equal

import optcom.utils.constants as cst
from optcom.solvers.ode_solver import ODESolver

import matplotlib.pyplot as plt

ode_methods = ["euler", "rk1", "rk2", "rk3", "rk4"]

# ----------------------------------------------------------------------
# Fixtures -------------------------------------------------------------
# ----------------------------------------------------------------------

@pytest.fixture
def stepper_all_methods_and_exact_sol():
    """Return the ode calculation results for the specified number of
    steps and the specified function for all ode methods in ode_methods
    as well as the exact solution from f_sol.

    """
    def calc_ode(f, init_vec, minimum, maximum, steps, f_sol, methods):

        step_array, h = np.linspace(minimum, maximum, steps, False, True)
        step_array += h
        res = [[init_vec] for i in range(len(methods) +1)]
        # Exact solution
        for i in range(len(step_array)):
            res[0].append(f_sol(step_array[i]))
        # ODE solution
        for k, method in enumerate(methods):
            solver = ODESolver(f, method)
            for i in range(len(step_array)):
                res[k+1].append(solver(res[k+1][-1], step_array[i], h))

        return np.hstack((np.array([minimum]), step_array)), res

    return calc_ode

# ----------------------------------------------------------------------
# Tests ----------------------------------------------------------------
# ----------------------------------------------------------------------

@pytest.mark.solvers
def test_solvers_on_lode_1(stepper_all_methods_and_exact_sol):
    r"""Should fail if the output pulses are not almost equal.
    Test the solver on the following differential equation:

    .. math::
        \frac{dy(x)}{dx} = \alpha y \qquad \y_0(0) = \gamma

    which has solution:

    .. math::
        y(x) = -\gamma\exp(\alpha x)

    """
    alpha = -0.5
    gamma = np.asarray(2.0, dtype=float)
    f = lambda vec, z, h: alpha * vec
    f_sol = lambda z: gamma * np.exp(alpha * z)
    steps = 1000
    x, ys = stepper_all_methods_and_exact_sol(f, gamma, 0.0, 10.0, steps,
                                              f_sol, ode_methods)
    # Test
    exact_sol = ys[0]
    for i in range(1, len(ys)):
        assert_array_almost_equal(exact_sol, ys[i], 2)


@pytest.mark.solvers
def test_solvers_on_lode_2(stepper_all_methods_and_exact_sol):
    r"""Should fail if the output pulses are not almost equal.
    Test the solver on the following linear ordinary differential
    equation:

    .. math::
        \frac{dy(x)}{dx} = \alpha - \beta y \qquad y_0(0) = \gamma

    which has solution:

    .. math::
        y(x) = \frac{\alpha}{\beta}
               + \big(\gamma - \frac{\alpha}{\beta)}\big)\exp(-\beta x)

    """
    alpha = 0.5
    beta = 1.2
    gamma = np.asarray(2.0, dtype=float)
    f = lambda vec, z, h: alpha - (beta*vec)
    f_sol = lambda z: (alpha/beta) + (gamma-(alpha/beta)) * np.exp(-1*beta*z)
    steps = 1000
    x, ys = stepper_all_methods_and_exact_sol(f, gamma, 0.0, 10.0, steps,
                                              f_sol, ode_methods)
    # Test
    exact_sol = ys[0]
    for i in range(1, len(ys)):
        assert_array_almost_equal(exact_sol, ys[i], 2)
