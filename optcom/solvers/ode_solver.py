# This file is part of Optcom.
#
# Optcom is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Optcom is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Optcom.  If not, see <https://www.gnu.org/licenses/>.

""".. moduleauthor:: Sacha Medaer"""

import copy
from typing import Callable, List, Optional

import numpy as np
from nptyping import Array

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.equations.abstract_equation import AbstractEquation
from optcom.solvers.abstract_solver import AbstractSolver


class ODESolver(AbstractSolver):

    _default_method = cst.DFT_ODEMETHOD

    def __init__(self, f: AbstractEquation,
                 method: Optional[str] = cst.DFT_ODEMETHOD):
        """
        Parameters
        ----------
        f : AbstractEquation
            The function to compute.
        method :
            The computation method.
        """
        super().__init__(f, method)
    # ==================================================================
    @staticmethod
    def euler(f: AbstractEquation, vectors: Array, h: float, z: float
              ) -> Array:
        r"""Euler method to solve system of differential equations.

        Parameters
        ----------
        f :
            The functions which compute each equation of the system.
        vectors :
            The value of the unknown (waves) at the considered
            time/space step.
        h :
            The step size.
        z :
            The current value of the variable.

        Returns
        -------
        :
            The one step euler computation results.

        Notes
        -----
        Having the initial system of differential equations:

        .. math:: \begin{cases} x'(t) = f(t, x, y, z),
                  \quad x(t_0) = x_0 \\
                  y'(t) = f(t, x, y, z), \quad y(t_0) = y_0  \\
                  z'(t) = f(t, x, y, z), \quad z(t_0) = z_0  \\
                  \ldots \\ \end{cases}

        The Euler method solves it by the following iterative method:

        .. math:: \begin{cases} x_{k+1} = x_{k} + hf(t_k, x_k, y_k, z_k)\\
                  y_{k+1} = y_{k} + hf(t_k, x_{k+1}, y_k, z_k)\\
                  z_{k+1} = z_{k} + hf(t_k, x_{k+1}, y_{k+1}, z_k) \\
                  \ldots \\ \end{cases}

        """
        k_0 = np.zeros_like(vectors)
        for i in range(len(vectors)):
            k_0[i] = h * f.term_all(vectors, i, h)

        return vectors + k_0
    # ==================================================================
    @staticmethod
    def rk1(f: AbstractEquation, vectors: Array, h: float, z: float) -> Array:

        return ODESolver.euler(f,vectors,h,z)
    # ==================================================================
    @staticmethod
    def rk2(f: AbstractEquation, vectors: Array, h: float, z: float) -> Array:
        h_h = 0.5 * h
        k_0 = np.zeros_like(vectors)
        k_1 = np.zeros_like(vectors)
        for i in range(len(vectors)):
            k_0[i] = h * f.term_all(vectors, i, h)
        vectors_ = vectors + (0.5*k_0)
        for i in range(len(vectors)):
            k_1[i] = h * f.term_all(vectors_, i, h_h)

        return vectors + k_1
    # ==================================================================
    @staticmethod
    def rk3(f: AbstractEquation, vectors: Array, h: float, z: float) -> Array:
        h_h = 0.5 * h
        k_0 = np.zeros_like(vectors)
        k_1 = np.zeros_like(vectors)
        k_2 = np.zeros_like(vectors)
        for i in range(len(vectors)):
            k_0[i] = h * f.term_all(vectors, i, h)
        vectors_ = vectors + (0.5*k_0)
        for i in range(len(vectors)):
            k_1[i] = h * f.term_all(vectors_, i, h_h)
        vectors_ = vectors - k_0 + 2*k_1
        for i in range(len(vectors)):
            k_2[i] = h * f.term_all(vectors_, i, h)

        return vectors + (1/6)*k_0 + (2/3)*k_1 + (1/6)*k_2
    # ==================================================================
    @staticmethod
    def rk4(f: AbstractEquation, vectors: Array, h: float, z: float) -> Array:
        h_h = 0.5 * h
        k_0 = np.zeros_like(vectors)
        k_1 = np.zeros_like(vectors)
        k_2 = np.zeros_like(vectors)
        k_3 = np.zeros_like(vectors)
        for i in range(len(vectors)):
            k_0[i] = h * f.term_all(vectors, i, h)
        vectors_ = vectors + (0.5*k_0)
        for i in range(len(vectors)):
            k_1[i] = h * f.term_all(vectors_, i, h_h)
        vectors_ = vectors + (0.5*k_1)
        for i in range(len(vectors)):
            k_2[i] = h * f.term_all(vectors_, i, h_h)
        vectors_ = vectors + k_2
        for i in range(len(vectors)):
            k_3[i] = h * f.term_all(vectors_, i, h)

        return vectors + (1/6)*k_0 + (1/3)*k_1 + (1/3)*k_2 + (1/6)*k_3


if __name__ == "__main__":

    import math

    import optcom.utils.plot as plot
    import optcom.layout as layout
    import optcom.components.gaussian as gaussian
    import optcom.domain as domain
    from optcom.components.fiber_coupler import FiberCoupler
    from optcom.effects.coupling import Coupling
    from optcom.utils.utilities_user import temporal_power, spectral_power

    plot_groups = []
    plot_labels = []
    plot_titles = []
    x_datas = []
    y_datas = []

    ode_methods = ["euler", "rk1", "rk2", "rk3", "rk4"]

    # ---------------- NLSE solvers test -------------------------------
    lt = layout.Layout()

    Lambda = 1030.0
    pulse = gaussian.Gaussian(channels=1, peak_power=[1.0, 1.0],
                              center_lambda=[Lambda])

    steps = int(1e2)
    beta = [[1e5,10.0,-0.0],[1e5,10.0,-0.0]]
    V = [2.0]
    core_radius = [5.0]
    c2c_spacing = [[15.0]]
    n_0 = [1.02]
    omega = domain.Domain.lambda_to_omega(Lambda)
    kappa_ = Coupling.calc_kappa(omega, V[0], core_radius[0],
                                 c2c_spacing[0][0], n_0[0])
    kappa = None
    delta_a = 0.5*(beta[0][0] - beta[1][0])
    length_c = cst.PI/(2*math.sqrt(delta_a**2 + kappa_**2))
    length = length_c / 2

    for j, method in enumerate(ode_methods):
        coupler = FiberCoupler(length=length, kappa=kappa, V=V, n_0=n_0,
                               core_radius=core_radius,
                               c2c_spacing=c2c_spacing, ATT=False, DISP=False,
                               nl_approx=False, SPM=False, SS=False, RS=False,
                               XPM=True, ASYM=True, COUP=True, approx_type=1,
                               nlse_method = 'ssfm_super_sym', steps=steps,
                               ode_method = 'rk4', save=True, wait=False)
        lt.link((pulse[0], coupler[0]))
        lt.run(pulse)
        lt.reset()
        # Plot parameters and get waves
        x_datas.append(coupler[2][0].time)
        y_datas.append(temporal_power(coupler[2][0].channels))
        plot_groups.append(0)

    plot_labels.extend(ode_methods)
    plot_titles.extend(["ODE solvers test with n={}".format(str(steps))])
    # -------------------- Plotting results ------------------------
    plot.plot2d(x_datas, y_datas, plot_groups=plot_groups,
                plot_titles=plot_titles, x_labels=['t'], y_labels=['P_t'],
                plot_labels=plot_labels, opacity=0.3)
