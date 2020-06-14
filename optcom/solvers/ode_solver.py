# Copyright 2019 The Optcom Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""".. moduleauthor:: Sacha Medaer"""

import copy
from typing import Callable, List, Optional

import numpy as np

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.solvers.abstract_solver import AbstractSolver
from optcom.solvers.abstract_solver import SOLVER_CALLABLE_TYPE


class ODESolver(AbstractSolver):
    r"""Solve a system of ordinary differential equations.

    Notes
    -----
    A system of ordinary differential equations is of the form:

    .. math:: \begin{cases}
                \frac{dy_0(x)}{dx} = f_0(x, y_0, \ldots, y_n ),
                \quad &y_0(x_0) = y_{00} \\
                \ldots \\
                \frac{dy_i(x)}{dx} = f_i(x, y_0, \ldots, y_i,
                \ldots, y_n), \quad &y_i(x_0) = y_{i0}  \\
                \ldots \\
                \frac{dy_n(x)}{dx} = f_n(x, y_0, \ldots, y_n ),
                \quad &y_n(x_0) = y_{n0} \\
              \end{cases}

    """

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
        r"""Euler method to solve a system of differential equations.

        Parameters
        ----------
        f :
            The functions which compute each equation of the system.
        vectors :
            The value of the unknowns at the considered step.
        z :
            The current value of the variable.
        h :
            The step size.

        Returns
        -------
        :
            The one step Euler method results.

        Notes
        -----
        The Euler method solves the system of differential equations
        by the following iterative method:

        .. math:: \begin{split}
                    y_{0,k+1} &= y_{0,k} + hf_0(x_k, y_{0,k}, \ldots,
                    y_{n,k} )\\
                    \ldots\\
                    y_{i,k+1} &= y_{i,k} + hf_i(x_k, y_{0,k+1}, \ldots,
                    y_{i-1,k+1}, y_{i,k}, \ldots, y_{n,k})\\
                    \ldots\\
                    y_{n,k+1} &= y_{n,k} + hf_n(x_k, y_{0,k+1}, \ldots,
                    y_{n-1,k+1}, y_{n,k})
                   \end{split}

        """
        k_0 = np.zeros_like(vectors)
        for i in range(len(vectors)):
            k_0[i] = h * f(vectors, z, h, i)

        return vectors + k_0
    # ==================================================================
    @staticmethod
    def rk1(f: SOLVER_CALLABLE_TYPE, vectors: np.ndarray, z: float, h: float
            ) -> np.ndarray:
        r"""Runge-Kutta 1st order method to solve a system of
        differential equations.

        Parameters
        ----------
        f :
            The functions which compute each equation of the system.
        vectors :
            The value of the unknowns at the considered step.
        z :
            The current value of the variable.
        h :
            The step size.

        Returns
        -------
        :
            The one step Runge-Kutta 1st order method results.

        Notes
        -----
        The explicit Runge-Kutta 1st order method is equivalent to the
        explicit forward Euler method.

        """

        return ODESolver.euler(f, vectors, z, h)
    # ==================================================================
    @staticmethod
    def rk2(f: SOLVER_CALLABLE_TYPE, vectors: np.ndarray, z: float, h: float
            ) -> np.ndarray:
        r"""Runge-Kutta 2nd order method to solve a system of
        differential equations.

        Parameters
        ----------
        f :
            The functions which compute each equation of the system.
        vectors :
            The value of the unknowns at the considered step.
        z :
            The current value of the variable.
        h :
            The step size.

        Returns
        -------
        :
            The one step Runge-Kutta 2nd order method results.

        Notes
        -----
        The Runge-Kutta 2nd order method solves the system of
        differential equations by the following iterative method:

        .. math:: \begin{aligned}
                    k_{0i} &= hf_i(x_k, y_{0,k}, \ldots, y_{n,k})
                    \quad \quad && \forall i=1,\ldots,n\\
                    k_{1i} &= hf_i(x_k + \frac{h}{2}, y_{0,k}
                    + \frac{k_{00}}{2}, \ldots, y_{n,k}
                    + \frac{k_{0n}}{2}) \quad \quad
                    && \forall i=1,\ldots,n\\
                    y_{i,k+1} &= y_{i,k} + k_{1i} + O(h^3) \quad
                    \quad && \forall i=1,\ldots,n
                   \end{aligned}

        """
        h_h = 0.5 * h
        k_0 = np.zeros_like(vectors)
        k_1 = np.zeros_like(vectors)
        for i in range(len(vectors)):
            k_0[i] = h * f(vectors, z, 0.0, i)
        vectors_ = vectors + (0.5*k_0)
        for i in range(len(vectors)):
            k_1[i] = h * f(vectors_, z+h_h, h_h, i)

        return vectors + k_1
    # ==================================================================
    @staticmethod
    def rk3(f: SOLVER_CALLABLE_TYPE, vectors: np.ndarray, z: float, h: float
            ) -> np.ndarray:
        r"""Runge-Kutta 3rd order method to solve a system of
        differential equations.

        Parameters
        ----------
        f :
            The functions which compute each equation of the system.
        vectors :
            The value of the unknowns at the considered step.
        z :
            The current value of the variable.
        h :
            The step size.

        Returns
        -------
        :
            The one step Runge-Kutta 3rd order method results.

        Notes
        -----
        The Runge-Kutta 3rd order method solves the system of
        differential equations by the following iterative method:

        .. math:: \begin{aligned}
                    k_{0i} &= hf_i(x_k, y_{0,k}, \ldots, y_{n,k})
                    \quad \quad && \forall i=1,\ldots,n\\
                    k_{1i} &= hf_i(x_k + \frac{h}{2}, y_{0,k}
                    + \frac{k_{00}}{2}, \ldots, y_{n,k}
                    + \frac{k_{0n}}{2}) \quad \quad
                    && \forall i=1,\ldots,n\\
                    k_{2i} &= hf_i(x_k + h, y_{0,k} - k_{00} + 2k_{10},
                    \ldots, y_{n,k} - k_{0n} + 2k_{1n}) \quad \quad
                    && \forall i=1,\ldots,n\\
                    y_{i,k+1} &= y_{i,k} + \frac{1}{6} \big[k_{0i}
                    + 4k_{1i} + k_{2i} \big] + O(h^4) \quad \quad
                    && \forall i=1,\ldots,n
                   \end{aligned}

        """
        h_h = 0.5 * h
        k_0 = np.zeros_like(vectors)
        k_1 = np.zeros_like(vectors)
        k_2 = np.zeros_like(vectors)
        for i in range(len(vectors)):
            k_0[i] = h * f(vectors, z, 0.0, i)
        vectors_ = vectors + (0.5*k_0)
        for i in range(len(vectors)):
            k_1[i] = h * f(vectors_, z+h_h, h_h, i)
        vectors_ = vectors - k_0 + 2*k_1
        for i in range(len(vectors)):
            k_2[i] = h * f(vectors_, z+h, h, i)

        return vectors + ((1/6)*k_0) + ((2/3)*k_1) + ((1/6)*k_2)
    # ==================================================================
    @staticmethod
    def rk4(f: SOLVER_CALLABLE_TYPE, vectors: np.ndarray, z: float, h: float
            ) -> np.ndarray:
        r"""Runge-Kutta 4th order method to solve a system of
        differential equations.

        Parameters
        ----------
        f :
            The functions which compute each equation of the system.
        vectors :
            The value of the unknowns at the considered step.
        z :
            The current value of the variable.
        h :
            The step size.

        Returns
        -------
        :
            The one step Runge-Kutta 4th order method results.

        Notes
        -----
        The Runge-Kutta 4th order method solves the system of
        differential equations by the following iterative method:

        .. math:: \begin{aligned}
                    k_{0i} &= hf_i(x_k, y_{0,k}, \ldots, y_{n,k})
                    \quad \quad && \forall i=1,\ldots,n\\
                    k_{1i} &= hf_i(x_k + \frac{h}{2}, y_{0,k}
                    + \frac{k_{00}}{2}, \ldots, y_{n,k}
                    + \frac{k_{0n}}{2}) \quad \quad
                    && \forall i=1,\ldots,n\\
                    k_{2i} &= hf_i(x_k + \frac{h}{2}, y_{0,k}
                    + \frac{k_{10}}{2}, \ldots, y_{n,k}
                    + \frac{k_{1n}}{2}) \quad \quad
                    && \forall i=1,\ldots,n\\
                    k_{3i} &= hf_i(x_k + h, y_{0,k} + k_{20},
                    \ldots, y_{n,k} + k_{2n}) \quad \quad
                    && \forall i=1,\ldots,n\\
                    y_{i,k+1} &= y_{i,k} + \frac{1}{6} \big[ k_{0i}
                    + 2 k_{1i} + 2 k_{2i} + k_{3i} \big] + O(h^5)
                    \quad \quad && \forall i=1,\ldots,n
                  \end{aligned}

        """
        h_h = 0.5 * h
        k_0 = np.zeros_like(vectors)
        k_1 = np.zeros_like(vectors)
        k_2 = np.zeros_like(vectors)
        k_3 = np.zeros_like(vectors)
        for i in range(len(vectors)):
            k_0[i] = h * f(vectors, z, 0.0, i)
        vectors_ = vectors + (0.5*k_0)
        for i in range(len(vectors)):
            k_1[i] = h * f(vectors_, z+h_h, h_h, i)
        vectors_ = vectors + (0.5*k_1)
        for i in range(len(vectors)):
            k_2[i] = h * f(vectors_, z+h_h, h_h, i)
        vectors_ = vectors + k_2
        for i in range(len(vectors)):
            k_3[i] = h * f(vectors_, z+h, h, i)

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
    line_labels: List[Optional[str]] = []
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
                               ode_method=method, save=True, wait=False, NOISE=True)
        lt.add_link(pulse[0], coupler[0])
        lt.run(pulse)
        lt.reset()
        # Plot parameters and get waves
        x_datas.append(coupler[2][0].time)
        y_datas.append(Field.temporal_power(coupler[2][0].channels))
        plot_groups.append(0)

    line_labels.extend(ode_methods)
    plot_titles.extend(["ODE solvers test with n={}".format(str(steps))])
    # -------------------- Plotting results ------------------------
    plot.plot2d(x_datas, y_datas, plot_groups=plot_groups,
                plot_titles=plot_titles, x_labels=['t'], y_labels=['P_t'],
                line_labels=line_labels, line_opacities=[0.1])
