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

# Optcom's development comments:
#
# - Make conventional DE solver

import copy
from typing import Callable, List, Optional

import numpy as np
from nptyping import Array

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.equations.abstract_equation import AbstractEquation
from optcom.utils.fft import FFT


class Solver(object):

    def __init__(self, f: AbstractEquation,
                 method: str = cst.DEFAULT_SOLVER):
        """
        Parameters
        ----------
        f : AbstractEquation
            The function to compute.
        method :
            The computation method.
        """
        self._method: Callable
        if (not method):    # analytical solution, no need numerical
            self._method = getattr(Solver, 'f_call')
        elif (hasattr(Solver, method.lower())):   # Force to be static method
            self._method = getattr(Solver, method.lower())
        else:
            util.warning_terminal("This solver method does not exist, default "
                "method will be use")
            self._method = getattr(Solver, cst.DEFAULT_SOLVER)
        self.f: AbstractEquation = f
    # ==================================================================
    def __call__(self, waves: Array[cst.NPFT], h: float, z: float
                 ) -> Array[cst.NPFT]:
        """
        Parameters
        ----------
        waves :
            The value of the unknown (waves) at the considered time/
            space step.
        h :
            The step size.
        z :
            The variable value. (time, space, ...)

        """
        self.f.set(waves, h, z)
        res = self._method(self.f, waves, h, z)
        self.f.update(waves, h, z)

        return res
    # ==================================================================
    @staticmethod
    def f_call(f: AbstractEquation, vectors: Array[cst.NPFT], h: float,
               z: float) -> Array[cst.NPFT]:
        """Call the __call__ method of the equation f."""

        return f(vectors, h, z)
    # ==================================================================
    # Conventional PDE solvers =========================================
    # ==================================================================
    @staticmethod
    def euler(f: AbstractEquation, vectors: Array[cst.NPFT], h: float,
              z: float) -> Array[cst.NPFT]:
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
        for i in range(len(vectors)):
            vectors[i] = vectors[i] + f.term_all(vectors, i, h)*h

        return vectors
    # ==================================================================
    # NLSE like eq. solvers ============================================
    # ==================================================================
    @staticmethod
    def ssfm(f: AbstractEquation, waves: Array[cst.NPFT], h: float,
             z: float) -> Array[cst.NPFT]:
        r"""Split step Fourier method.

        Parameters
        ----------
        f :
            The function to compute.
        waves :
            The value of the unknown (waves) at the considered
            space step.
        h :
            The step size.
        z :
            The current value of the space variable.

        Returns
        -------
        :
            The one step euler computation results.

        Notes
        -----
        Having the initial system of differential equations:

        .. math:: \begin{alignat}{1}
                    A^{L} &= \mathcal{F}^{-1}\big\{\exp
                    \big(h\hat\mathcal{{D}}\big)\mathcal{F}\{A(z,T)\}
                    \big\}\\
                    A(z+h,T) &= \exp\big(h\hat\mathcal{{N}}(A(z,T))
                    \big)A^{L}
                  \end{alignat}

        for :math:`j = 1, \ldots, K` where :math:`K` is the number of
        channels.

        """
        old_waves = copy.deepcopy(waves)
        for i in range(len(waves)):
            waves[i] = f.exp_term_lin(waves, i, h)
            waves[i] = f.exp_term_non_lin(old_waves, i, h, waves[i])

        return waves
    # ==================================================================
    @staticmethod
    def ssfm_symmetric(f: AbstractEquation, waves: Array[cst.NPFT], h: float,
                       z: float) -> Array[cst.NPFT]:
        r"""Symmetric split step Fourier method.

        Parameters
        ----------
        f :
            The function to compute.
        waves :
            The value of the unknown (waves) at the considered
            space step.
        h :
            The step size.
        z :
            The current value of the space variable.

        Returns
        -------
        :
            The one step euler computation results.

        Notes
        -----
        Having the initial system of differential equations:

        .. math:: \begin{alignat}{1}
                    A_j^{L} &= \mathcal{F}^{-1}\Big\{\exp\Big(\frac{h}{2}
                    \hat\mathcal{{D}}\Big)\mathcal{F}\{A_j(z)\}\Big\}\\
                    A_j^{N} &= \exp\Big(h\hat\mathcal{{N}}
                    \big(A_1(z), \ldots, A_K(z)\big)\Big)  A_j^{L}\\
                    A_j(z+h) &=  \mathcal{F}^{-1}\Big\{\exp\Big(\frac{h}{2}
                    \hat\mathcal{{D}}\Big)\mathcal{F}\{A_j^{N}\}\Big\}
                  \end{alignat}

        for :math:`j = 1, \ldots, K` where :math:`K` is the number of
        channels.

        """
        h_h = 0.5 * h
        old_waves = copy.deepcopy(waves)
        for i in range(len(waves)):
            waves[i] = f.exp_term_lin(waves, i, h_h)
        for i in range(len(waves)):
            waves[i] = f.exp_term_non_lin(old_waves, i, h, waves[i])
            waves[i] = f.exp_term_lin(waves, i, h_h)

        return waves
    # ==================================================================
    @staticmethod
    def ssfm_reduced(f: AbstractEquation, waves: Array[cst.NPFT], h: float,
                     z: float) -> Array[cst.NPFT]:
        r"""Reduced split step Fourier method.

        Parameters
        ----------
        f :
            The function to compute.
        waves :
            The value of the unknown (waves) at the considered
            space step.
        h :
            The step size.
        z :
            The current value of the space variable.

        Returns
        -------
        :
            The one step euler computation results.

        Notes
        -----
        Having the initial system of differential equations:

        .. math:: \begin{alignat}{1}
                    A^{L} &= \mathcal{F}^{-1}\Big\{\exp\Big(\frac{h}{2}
                    \hat\mathcal{{D}}\Big)\mathcal{F}\{A(z,T)\}\Big\}\\
                    A^{N} &= \exp\big(h\hat\mathcal{{N}}(A^{L})\big) A^{L}\\
                    A(z+h,T) &=  \mathcal{F}^{-1}\Big\{\exp\Big(\frac{h}{2}
                    \hat\mathcal{{D}}\Big)\mathcal{F}\{A^{N}\}\Big\}
                  \end{alignat}

        for :math:`j = 1, \ldots, K` where :math:`K` is the number of
        channels.

        """
        h_h = 0.5 * h
        A_L = np.zeros(waves.shape, dtype=cst.NPFT)
        for i in range(len(waves)):
            A_L[i] = f.exp_term_lin(waves, i, h_h)
        for i in range(len(waves)):
            waves[i] = f.exp_term_non_lin(A_L, i, h, A_L[i])
            waves[i] = f.exp_term_lin(waves, i, h_h)

        return waves
    # ==================================================================
    @staticmethod
    def ssfm_opti_reduced(f: AbstractEquation, waves: Array[cst.NPFT],
                          h: float, z: float) -> Array[cst.NPFT]:
        r"""Optimized reduced split step Fourier method.

        Parameters
        ----------
        f :
            The function to compute.
        waves :
            The value of the unknown (waves) at the considered
            space step.
        h :
            The step size.
        z :
            The current value of the space variable.

        Returns
        -------
        :
            The one step euler computation results.

        Notes
        -----
        Having the initial system of differential equations:

        .. math:: \begin{alignat}{1}
                    A_j^{L} &= \mathcal{F}^{-1}\Big\{\exp\Big(\frac{h}{2}
                        \hat\mathcal{{D}}\Big)\mathcal{F}\{A_j(z)\}\Big\}\\
                    A_j^{N} &= \exp\Big(h\hat\mathcal{{N}}\big(A_1^{N}, \ldots,
                    A_{j-1}^{N}, A_{j}^{L}, \ldots, A_K^{L}\big)\Big)
                    A_j^{L}\\
                    A_j(z+h) &=  \mathcal{F}^{-1}\Big\{\exp\Big(\frac{h}{2}
                    \hat\mathcal{{D}}\Big)\mathcal{F}\{A_j^{N}\}\Big\}
                  \end{alignat}

        for :math:`j = 1, \ldots, K` where :math:`K` is the number of
        channels.

        """
        h_h = 0.5 * h
        for i in range(len(waves)):
            waves[i] = f.exp_term_lin(waves, i, h_h)
        for i in range(len(waves)):
            waves[i] = f.exp_term_non_lin(waves, i, h, waves[i])
        for i in range(len(waves)):
            waves[i] = f.exp_term_lin(waves, i, h_h)

        return waves
    # ==================================================================
    @staticmethod
    def ssfm_super_sym(f: AbstractEquation, waves: Array[cst.NPFT], h: float,
                       z: float) -> Array[cst.NPFT]:
        r"""Super symmetric split step Fourier method.

        Parameters
        ----------
        f :
            The function to compute.
        waves :
            The value of the unknown (waves) at the considered
            space step.
        h :
            The step size.
        z :
            The current value of the space variable.

        Returns
        -------
        :
            The one step euler computation results.

        Notes
        -----
        Having the initial system of differential equations:

        .. math:: \begin{alignat}{1}
                    A_j^{L} &= \mathcal{F}^{-1}\Big\{\exp\Big(\frac{h}{2}
                    \hat\mathcal{{D}}\Big)\mathcal{F}\{A_j(z)\}\Big\}\\
                    A_j^{N'} &= \exp\Big(\frac{h}{2}\hat\mathcal{{N}}
                    \big(A_1^{L}, \ldots, A_K^{L}\big)\Big)  A_j^{L}\\
                    A_i^{N} &= \exp\Big(\frac{h}{2}\hat\mathcal{{N}}
                    \big(A_1^{N'}, \ldots, A_{i}^{N}, A_{i+1}^{N},
                    \ldots, A_K^{N}\big)\Big)  A_j^{L}\\
                    A_j(z+h) &=  \mathcal{F}^{-1}\Big\{\exp\Big(\frac{h}{2}
                    \hat\mathcal{{D}}\Big)\mathcal{F}\{A_j^{N}\}\Big\}
                  \end{alignat}

        for :math:`j = 1, \ldots, K` and :math:`i = K, \ldots, 1`
        where :math:`K` is the number of channels.

        """
        h_h = 0.5 * h
        A_L = np.zeros(waves.shape, dtype=cst.NPFT)
        for i in range(len(waves)):
            A_L[i] = f.exp_term_lin(waves, i, h_h)
        for i in range(len(waves)-1):
            waves[i] = f.exp_term_non_lin(A_L, i, h_h, A_L[i])
        waves[-1] = f.exp_term_non_lin(A_L, len(A_L)-1, h, A_L[-1])
        for i in range(len(waves)-2, -1, -1):
            waves[i] = f.exp_term_non_lin(waves, i, h_h, waves[i])
        for i in range(len(waves)):
            waves[i] = f.exp_term_lin(waves, i, h_h)

        return waves
    # ==================================================================
    @staticmethod
    def ssfm_opti_super_sym(f: AbstractEquation, waves: Array[cst.NPFT],
                            h: float, z: float) -> Array[cst.NPFT]:
        r"""Optimized super symmetric split step Fourier method.

        Parameters
        ----------
        f :
            The function to compute.
        waves :
            The value of the unknown (waves) at the considered
            space step.
        h :
            The step size.
        z :
            The current value of the space variable.

        Returns
        -------
        :
            The one step euler computation results.

        Notes
        -----
        Having the initial system of differential equations:

        .. math:: \begin{alignat}{1}
                    A_j^{L} &= \mathcal{F}^{-1}\Big\{\exp\Big(\frac{h}{2}
                    \hat\mathcal{{D}}\Big)\mathcal{F}\{A_j(z)\}\Big\}\\
                    A_j^{N'} &= \exp\Big(\frac{h}{2}\hat\mathcal{{N}}
                    \big(A_1^{N'}, \ldots, A_{j-1}^{N'}, A_{j}^{L},
                    \ldots, A_K^{L}\big)\Big)  A_j^{L}\\
                    A_i^{N} &= \exp\Big(\frac{h}{2}\hat\mathcal{{N}}
                    \big(A_1^{N'}, \ldots, A_{i}^{N'}, A_{i+1}^{N},
                    \ldots, A_K^{N}\big)\Big)  A_j^{L} \\
                    A_j(z+h) &=  \mathcal{F}^{-1}\Big\{\exp\Big(\frac{h}{2}
                    \hat\mathcal{{D}}\Big)\mathcal{F}\{A_j^{N}\}\Big\}
                  \end{alignat}

        for :math:`j = 1, \ldots, K` and :math:`i = K, \ldots, 1`
        where :math:`K` is the number of channels.

        """
        h_h = 0.5 * h
        for i in range(len(waves)):
            waves[i] = f.exp_term_lin(waves, i, h_h)
        for i in range(len(waves)-1):
            waves[i] = f.exp_term_non_lin(waves, i, h_h, waves[i])
        waves[-1] = f.exp_term_non_lin(waves, len(waves)-1, h, waves[-1])
        for i in range(len(waves)-2, -1, -1):
            waves[i] = f.exp_term_non_lin(waves, i, h_h, waves[i])
        for i in range(len(waves)):
            waves[i] = f.exp_term_lin(waves, i, h_h)

        return waves
    # ==================================================================
    @staticmethod
    def ssfm_truncated(A, h, f):
        pass
    # ==================================================================
    @staticmethod
    def ssfm_central_diff(f: AbstractEquation, waves: Array[cst.NPFT],
                          h: float, z: float) -> Array[cst.NPFT]:
        pass
    # ==================================================================
    @staticmethod
    def ssfm_upwind(f: AbstractEquation, waves: Array[cst.NPFT], h: float,
                    z: float) -> Array[cst.NPFT]:
        pass
    # ==================================================================
    @staticmethod
    def rk4ip(f: AbstractEquation, waves: Array[cst.NPFT], h: float, z: float
              ) -> Array[cst.NPFT]:

        if (len(waves) == 1):
            A = copy.deepcopy(waves[0])
            h_h = 0.5 * h
            A_lin = f.exp_term_lin(waves, 0, h_h)
            waves[0] = f.term_non_lin(waves, 0)
            k_0 = h * f.exp_term_lin(waves, 0, h_h)
            waves[0] = A + 0.5*k_0
            k_1 = h * f.term_non_lin(waves, 0)
            waves[0] = A + 0.5*k_1
            k_2 = h * f.term_non_lin(waves, 0)
            waves[0] = A_lin + k_2
            waves[0] = f.exp_term_lin(waves, 0, h_h)
            k_3 = h * f.term_non_lin(waves, 0)
            waves[0] = A_lin + k_0/6 + (k_1+k_2)/3
            waves[0] = k_3/6 + f.exp_term_lin(waves, 0, h_h)
            return waves

        else:
            util.warning_terminal("rk4ip with more than one field "
                "currently not supported")

        return waves
    # ==================================================================
    @staticmethod
    def rk4ip_gnlse(f: AbstractEquation, waves: Array[cst.NPFT], h: float,
                    z: float) -> Array[cst.NPFT]:
        if (len(waves) == 1):
            h_h = 0.5 * h
            A = copy.deepcopy(waves[0])
            exp_op_lin = f.exp_op_lin(waves, 0, h_h)
            #if (Solver.rk4ip_gnlse.first_rk4ip_gnlse_iter):
            A_lin = exp_op_lin * FFT.fft(A)
            #    Solver.rk4ip_gnlse.first_rk4ip_gnlse_iter = False
            #else:
            #    A_lin = exp_op_lin * Solver.rk4ip_gnlse.fft_A_next
            k_0 = h * exp_op_lin * f.op_non_lin_rk4ip(waves, 0)
            waves[0] = FFT.ifft(A_lin + k_0/2)
            k_1 = h * f.op_non_lin_rk4ip(waves, 0)
            waves[0] = FFT.ifft(A_lin + k_1/2)
            k_2 = h * f.op_non_lin_rk4ip(waves, 0)
            waves[0] = FFT.ifft(exp_op_lin * (A_lin + k_0/2))
            k_3 = h * f.op_non_lin_rk4ip(waves, 0)

            #Solver.rk4ip_gnlse.fft_A_next = k_3/6 + (exp_op_lin
            #                                * (A_lin + k_0/6 + (k_1+k_2)/3))
            waves[0] = FFT.ifft(k_3/6 + (exp_op_lin
                                         * (A_lin + k_0/6 + (k_1+k_2)/3)))

            return waves

        else:
            util.warning_terminal("rk4ip with more than two fields "
                "currently not supported")

        return waves

if __name__ == "__main__":

    import optcom.utils.plot as plot
    import optcom.layout as layout
    import optcom.components.gaussian as gaussian
    import optcom.domain as domain
    from optcom.components.fiber import Fiber
    from optcom.utils.utilities_user import temporal_power, spectral_power

    plot_groups = []
    plot_labels = []
    plot_titles = []
    times = []
    fields = []

    pde_methods = ["euler", "rk4"]

    nlse_methods = ["ssfm", "ssfm_reduced", "ssfm_symmetric",
                    "ssfm_opti_reduced", "ssfm_super_sym",
                    "ssfm_opti_super_sym", "rk4ip", "rk4ip"]

    # ---------------- PDE solvers test --------------------------------

    # to do

    # ---------------- NLSE solvers test -------------------------------
    lt = layout.Layout()

    pulse = gaussian.Gaussian(channels=1, peak_power=[1.0])

    steps = int(10e3)
    for j, method in enumerate(nlse_methods):
        if (j == (len(nlse_methods)-1)):
            nl_approx = False   # To compute rk4ip_gnlse
        else:
            nl_approx = True
        # Propagation
        fiber = Fiber(length=2.0, method=method, alpha=[0.046],
                      beta=[0.0, 1.0, -19.83, 0.031], gamma=4.3,
                      nl_approx=nl_approx, SPM=True, XPM=False, SS=False,
                      RS=False, approx_type=1, steps=steps, save=True)
        lt.link((pulse[0], fiber[0]))
        lt.run(pulse)
        lt.reset()
        # Plot parameters and get waves
        times.append(fiber.fields[1].time)
        fields.append(temporal_power(fiber.fields[1].channels))
        plot_groups.append(0)

    plot_labels.extend(nlse_methods)
    plot_titles.extend(["NLSE pde solvers test with n={}"
                       .format(str(steps))])
    # -------------------- Plotting results ----------------------------
    plot.plot(times, fields, plot_groups=plot_groups,
         plot_titles=plot_titles, x_labels=['t'], y_labels=['P_t'],
         plot_labels=plot_labels, opacity=0.3)
