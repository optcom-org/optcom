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

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.solvers.abstract_solver import AbstractSolver
from optcom.solvers.abstract_solver import SOLVER_CALLABLE_TYPE


class ODESolver(AbstractSolver):

    _default_method = cst.DFT_ODEMETHOD

    def __init__(self, f: SOLVER_CALLABLE_TYPE,
                 method: Optional[str] = cst.DFT_ODEMETHOD) -> None:
        """
        Parameters
        ----------
        f :
            The function to compute.
        method :
            The computation method.

        """
        super().__init__(f, method)
    # ==================================================================
    @staticmethod
    def euler(f: SOLVER_CALLABLE_TYPE, vectors: np.ndarray, z: float, h: float
              ) -> np.ndarray:
        r"""Euler method to solve system of differential equations.

        Parameters
        ----------
        f :
            The functions which compute each equation of the system.
        vectors :
            The value of the unknown (waves) at the considered
            time/space step.
        z :
            The current value of the space variable.
        h :
            The step size.

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
        k_0 = h * f(vectors, z, h)

        return vectors + k_0
    # ==================================================================
    @staticmethod
    def rk1(f: SOLVER_CALLABLE_TYPE, vectors: np.ndarray, z: float, h: float
            ) -> np.ndarray:

        return ODESolver.euler(f, vectors, z, h)
    # ==================================================================
    @staticmethod
    def rk2(f: SOLVER_CALLABLE_TYPE, vectors: np.ndarray, z: float, h: float
            ) -> np.ndarray:
        h_h = 0.5 * h
        k_0 = np.zeros_like(vectors)
        k_1 = np.zeros_like(vectors)
        k_0 = h * f(vectors, z, 0.0)
        vectors_ = vectors + (0.5*k_0)
        k_1 = h * f(vectors_, z+h_h, h_h)

        return vectors + k_1
    # ==================================================================
    @staticmethod
    def rk3(f: SOLVER_CALLABLE_TYPE, vectors: np.ndarray, z: float, h: float
            ) -> np.ndarray:
        h_h = 0.5 * h
        k_0 = np.zeros_like(vectors)
        k_1 = np.zeros_like(vectors)
        k_2 = np.zeros_like(vectors)
        k_0 = h * f(vectors, z, 0.0)
        vectors_ = vectors + (0.5*k_0)
        k_1 = h * f(vectors_, z+h_h, h_h)
        vectors_ = vectors - k_0 + 2*k_1
        k_2 = h * f(vectors_, z+h, h)

        return vectors + ((1/6)*k_0) + ((2/3)*k_1) + ((1/6)*k_2)
    # ==================================================================
    @staticmethod
    def rk4(f: SOLVER_CALLABLE_TYPE, vectors: np.ndarray, z: float, h: float
            ) -> np.ndarray:
        h_h = 0.5 * h
        k_0 = np.zeros_like(vectors)
        k_1 = np.zeros_like(vectors)
        k_2 = np.zeros_like(vectors)
        k_3 = np.zeros_like(vectors)
        k_0 = h * f(vectors, z, 0.0)
        vectors_ = vectors + (0.5*k_0)
        k_1 = h * f(vectors_, z+h_h, h_h)
        vectors_ = vectors + (0.5*k_1)
        k_2 = h * f(vectors_, z+h_h, h_h)
        vectors_ = vectors + k_2
        k_3 = h * f(vectors_, z+h, h)

        return vectors + ((1/6)*k_0) + ((1/3)*k_1) + ((1/3)*k_2) + ((1/6)*k_3)


if __name__ == "__main__":

    import math
    from typing import Callable, List, Optional, Union

    import numpy as np

    import optcom.utils.plot as plot
    from optcom.components.gaussian import Gaussian
    from optcom.components.fiber_coupler import FiberCoupler
    from optcom.domain import Domain
    from optcom.effects.coupling import Coupling
    from optcom.layout import Layout
    from optcom.parameters.fiber.coupling_coeff import CouplingCoeff
    from optcom.field import Field

    plot_groups: List[int] = []
    plot_labels: List[Optional[str]] = []
    plot_titles: List[str] = []
    x_datas: List[np.ndarray] = []
    y_datas: List[np.ndarray] = []

    ode_methods: List[str] = ["euler", "rk1", "rk2", "rk3", "rk4"]

    # ---------------- NLSE solvers test -------------------------------
    lt: Layout = Layout()

    Lambda: float = 1030.0
    pulse: Gaussian = Gaussian(channels=1, peak_power=[1.0, 1.0],
                              center_lambda=[Lambda])

    steps: int = int(1e1)
    beta_01: float = 1e5
    beta_02: float = 1e5
    beta: List[Union[List[float], Callable, None]] =\
        [[beta_01,10.0,-0.0],[beta_02,10.0,-0.0]]
    v_nbr_value = 2.0
    v_nbr: List[Union[float, Callable, None]] = [v_nbr_value]
    core_radius: List[float] = [5.0]
    c2c_spacing: List[List[float]] = [[15.0]]
    n_clad: float = 1.02
    omega: float = Domain.lambda_to_omega(Lambda)
    kappa_: Union[float, Callable]
    kappa_ = CouplingCoeff.calc_kappa(omega, v_nbr_value, core_radius[0],
                                      c2c_spacing[0][0], n_clad)
    kappa: List[List[Union[List[float], Callable, None]]] = [[None]]
    delta_a: float = 0.5*(beta_01 - beta_02)
    length_c: float = cst.PI/(2*math.sqrt(delta_a**2 + kappa_**2))
    length: float = length_c / 2


    for j, method in enumerate(ode_methods):
        coupler = FiberCoupler(length=length, kappa=kappa, v_nbr=v_nbr,
                               core_radius=core_radius, n_clad=n_clad,
                               c2c_spacing=c2c_spacing, ATT=False, DISP=False,
                               nl_approx=False, SPM=False, SS=False, RS=False,
                               XPM=False, ASYM=True, COUP=True, approx_type=1,
                               nlse_method='ssfm_super_sym', steps=steps,
                               ode_method=method, save=True, wait=False)
        lt.link((pulse[0], coupler[0]))
        lt.run(pulse)
        lt.reset()
        # Plot parameters and get waves
        x_datas.append(coupler[2][0].time)
        y_datas.append(Field.temporal_power(coupler[2][0].channels))
        plot_groups.append(0)

    plot_labels.extend(ode_methods)
    plot_titles.extend(["ODE solvers test with n={}".format(str(steps))])
    # -------------------- Plotting results ------------------------
    plot.plot2d(x_datas, y_datas, plot_groups=plot_groups,
                plot_titles=plot_titles, x_labels=['t'], y_labels=['P_t'],
                plot_labels=plot_labels, opacity=[0.1])
