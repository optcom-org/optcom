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

import optcom.config as cfg
import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.equations.abstract_field_equation import AbstractFieldEquation
from optcom.equations.ampgnlse import AmpGNLSE
from optcom.equations.cgnlse import CGNLSE
from optcom.equations.gnlse import GNLSE
from optcom.utils.fft import FFT
from optcom.solvers.abstract_solver import AbstractSolver


# Exceptions
class NLSESolverError(Exception):
    pass

class RK4IPGNLSEError(NLSESolverError):
    pass


class NLSESolver(AbstractSolver):

    _default_method = cst.DFT_NLSEMETHOD

    def __init__(self, f: AbstractFieldEquation,
                 method: Optional[str] = cst.DFT_NLSEMETHOD) -> None:
        """
        Parameters
        ----------
        f : AbstractFieldEquation
            The function to compute.
        method :
            The computation method.

        """
        # Special case for gnlse and rk4ip method
        method_: Optional[str] = ''
        is_gnlse_eq: bool = (isinstance(f, GNLSE) or isinstance(f, AmpGNLSE))
        if (is_gnlse_eq and (method == "rk4ip") and cfg.RK4IP_OPTI_GNLSE):
            method_ = "rk4ip_gnlse"
        else:
            method_ = method
        super().__init__(f, method_)
    # ==================================================================
    def __call__(self, vectors: np.ndarray, z: float, h: float) -> np.ndarray:
        """
        Parameters
        ----------
        vectors :
            The value of the variables at the considered time/
            space step.
        h :
            The step size.
        z :
            The variable value. (time, space, ...)

        """
        # If not change, put this in parent __call__ method
        vectors_ = np.array([vectors]) if (vectors.ndim == 1) else vectors
        res = self._method(self.f, vectors_, z, h)
        res_ = res[0] if (vectors.ndim == 1) else res

        return res_
    # ==================================================================
    @staticmethod
    def ssfm(f: AbstractFieldEquation, waves: np.ndarray, z: float, h: float
             ) -> np.ndarray:
        r"""Split step Fourier method.

        Parameters
        ----------
        f :
            The function to compute.
        waves :
            The value of the unknown (waves) at the considered
            space step.
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

        Equations:

        .. math::
              \begin{align}
                A_j^{N} &= \exp\Big(h\hat{\mathcal{N}}\big(A_1(z,T),
                    \ldots, A_j(z,T), \ldots, A_K(z,T)\big)\Big)A_j(z,T)
                    &\forall j=1,\ldots,K\\
                A_j(z+h,T) &= \exp\big(h\mathcal{D}\big)A_j^{N} &\forall
                    j=1,\ldots,K
              \end{align}

        Implementation:

        .. math::
              \begin{align}
                A_j^{N} &= \exp\Big(h\hat{\mathcal{N}\big(A_1(z),\ldots,
                    A_j(z), \ldots, A_K(z)\big)\Big)  A_j(z) &\forall
                    j=1,\ldots,K\\
                A_j(z+h) &= \mathcal{F}^{-1}\Big\{\exp\Big(h
                    \hat{\mathcal{D}}\Big)\mathcal{F}
                    \{A_j^{N}(z)\}\Big\} &\forall j=1,\ldots,K
              \end{align}

        where :math:`K` is the number of channels.

        """
        A_N = np.zeros_like(waves)
        for i in range(len(waves)):
            A_N[i] = f.exp_term_non_lin(waves, i, z, h, waves[i])
        for i in range(len(waves)):
            waves[i] = f.exp_term_lin(A_N, i, z, h)

        return waves
    # ==================================================================
    @staticmethod
    def ssfm_symmetric(f: AbstractFieldEquation, waves: np.ndarray, z: float,
                       h: float) -> np.ndarray:
        r"""Symmetric split step Fourier method.

        Parameters
        ----------
        f :
            The function to compute.
        waves :
            The value of the unknown (waves) at the considered
            space step.
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

        Equations:

        .. math::
              \begin{align}
                A_j^{L} &= \exp\Big(\frac{h}{2}\mathcal{D}\Big)A_j(z,T)
                    &\forall j=1,\ldots,K\\
                A_j^{N} &= A_j^{L} + h\bar{\mathcal{N}}\big(A_1(z, T),
                    \ldots, A_j(z, T), \ldots, A_K(z, T)\big)
                    &\forall j=1,\ldots,K\\
                A_j(z+h,T) &=  \exp\Big(\frac{h}{2}\mathcal{D}\Big)
                    A_j^{N} &\forall j=1,\ldots,K
              \end{align}

        Implementation:

        .. math::
              \begin{align}
                A_j^{L} &= \mathcal{F}^{-1}\Big\{\exp\Big(\frac{h}{2}
                    \hat{\mathcal{D}}\Big)\mathcal{F}\{A_j(z)\}\Big\}
                    &\forall j=1,\ldots,K\\
                A_j^{N} &= \exp\Big(h\hat{\mathcal{N}}\big(A_1(z, T),
                    \ldots, A_j(z), \ldots, A_K(z)\big)\Big)  A_j^{L}
                    &\forall j=1,\ldots,K \\
                A_j(z+h) &=  \mathcal{F}^{-1}\Big\{\exp\Big(\frac{h}{2}
                    \hat{\mathcal{D}}\Big)\mathcal{F}\{A_j^{N}\}\Big\}
                    &\forall j=1,\ldots,K
              \end{align}

        where :math:`K` is the number of channels.

        """
        h_h = 0.5 * h
        A_L = np.zeros_like(waves)
        A_N = np.zeros_like(waves)
        for i in range(len(waves)):
            A_L[i] = f.exp_term_lin(waves, i, z, h_h)
            A_N[i] = f.exp_term_non_lin(waves, i, z, h, A_L[i])
        for i in range(len(waves)):
            waves[i] = f.exp_term_lin(A_N, i, z, h_h)

        return waves
    # ==================================================================
    @staticmethod
    def ssfm_reduced(f: AbstractFieldEquation, waves: np.ndarray, z: float,
                     h: float) -> np.ndarray:
        r"""Reduced split step Fourier method.

        Parameters
        ----------
        f :
            The function to compute.
        waves :
            The value of the unknown (waves) at the considered
            space step.
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

        Equations:

        .. math::
              \begin{align}
                A_j^{L} &= \exp\Big(\frac{h}{2}\mathcal{D}\Big)A_j(z,T)
                    &\forall j=1,\ldots,K\\
                A_j^{N} &= A_j^{L} + h\bar{\mathcal{N}}(A_1^{L},\ldots,
                    A_j^{L}, \ldots, A_K^{L}) &\forall j=1,\ldots,K\\
                A_j(z+h,T) &=  \exp\Big(\frac{h}{2}\mathcal{D}\Big)
                    A_j^{N} &\forall j=1,\ldots,K
              \end{align}

        Implementation:

        .. math::
              \begin{align}
                A_j^{L} &= \mathcal{F}^{-1}\Big\{\exp\Big(\frac{h}{2}
                    \hat{\mathcal{D}}\Big)\mathcal{F}\{A_j(z)\}\Big\}
                    &\forall j=1,\ldots,K\\
                A_j^{N} &= \exp\Big(h\hat{\mathcal{N}}\big(A_1^{L},
                    \ldots, A_j^{L}, \ldots, A_K^{L}\big)\Big)  A_j^{L}
                    &\forall j=1,\ldots,K\\
                A_j(z+h) &=  \mathcal{F}^{-1}\Big\{\exp\Big(\frac{h}{2}
                    \hat{\mathcal{D}}\Big)\mathcal{F}\{A_j^{N}\}\Big\}
                    &\forall j=1,\ldots,K
              \end{align}

        where :math:`K` is the number of channels.

        """
        h_h = 0.5 * h
        A_L = np.zeros_like(waves)
        A_N = np.zeros_like(waves)
        for i in range(len(waves)):
            A_L[i] = f.exp_term_lin(waves, i, z, h_h)
        for i in range(len(waves)):
            A_N[i] = f.exp_term_non_lin(A_L, i, z, h, A_L[i])
        for i in range(len(waves)):
            waves[i] = f.exp_term_lin(A_N, i, z, h_h)

        return waves
    # ==================================================================
    @staticmethod
    def ssfm_opti_reduced(f: AbstractFieldEquation, waves: np.ndarray,
                          z: float, h: float) -> np.ndarray:
        r"""Optimized reduced split step Fourier method.

        Parameters
        ----------
        f :
            The function to compute.
        waves :
            The value of the unknown (waves) at the considered
            space step.
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

        Equations:

        .. math::
              \begin{align}
                A_j^{L} &= \exp\Big(\frac{h}{2}\mathcal{D}\Big)A_j(z,T)
                    &\forall j=1,\ldots,K\\
                A_j^{N} &= A_j^{L} + h\bar{\mathcal{N}}(A_1^{N},\ldots,
                    A_{j-1}^{N}, A_{j}^{L}, \ldots, A_K^{L})
                    &\forall j=1,\ldots,K\\
                A_j(z+h,T) &=  \exp\Big(\frac{h}{2}\mathcal{D}\Big)
                    A_j^{N} &\forall j=1,\ldots,K
              \end{align}

        Implementation:

        .. math::
              \begin{align}
                A_j^{L} &= \mathcal{F}^{-1}\Big\{\exp\Big(\frac{h}{2}
                    \hat{\mathcal{D}}\Big)\mathcal{F}\{A_j(z)\}\Big\}
                    &\forall j=1,\ldots,K\\
                A_j^{N} &= \exp\Big(h\hat{\mathcal{N}}\big(A_1^{N},
                    \ldots, A_{j-1}^{N}, A_{j}^{L}, \ldots,
                    A_K^{L}\big)\Big) A_j^{L} &\forall j=1,\ldots,K\\
                A_j(z+h) &=  \mathcal{F}^{-1}\Big\{\exp\Big(\frac{h}{2}
                    \hat{\mathcal{D}} \Big)\mathcal{F}\{A_j^{N}\}\Big\}
                    &\forall j=1,\ldots,K
              \end{align}

        where :math:`K` is the number of channels.

        """
        h_h = 0.5 * h
        for i in range(len(waves)):
            waves[i] = f.exp_term_lin(waves, i, z, h_h)
        for i in range(len(waves)):
            waves[i] = f.exp_term_non_lin(waves, i, z, h, waves[i])
        for i in range(len(waves)):
            waves[i] = f.exp_term_lin(waves, i, z, h_h)

        return waves
    # ==================================================================
    @staticmethod
    def ssfm_super_sym(f: AbstractFieldEquation, waves: np.ndarray, z: float,
                       h: float) -> np.ndarray:
        r"""Super symmetric split step Fourier method.

        Parameters
        ----------
        f :
            The function to compute.
        waves :
            The value of the unknown (waves) at the considered
            space step.
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

        Equations:

        .. math::
              \begin{align}
                A_j^{L} &= \exp\Big(\frac{h}{2}\mathcal{D}\Big)A_j(z, T)
                    &\forall j=1,\ldots,K\\
                A_j^{N'} &= A_j^{L} + \frac{h}{2}\bar{\mathcal{N}}
                    \big(A_1^{L}, \ldots, A_K^{L}\big)
                    &\forall j=1,\ldots,K\\
                A_j^{N} &= A_j^{N'} + \frac{h}{2}\bar{\mathcal{N}}\big(
                    A_1^{N'}, \ldots, A_{j}^{N'}, A_{j+1}^{N},
                    \ldots, A_K^{N}\big) &\forall j=K,\ldots,1\\
                A_j(z+h, T) &=  \exp\Big(\frac{h}{2}\hat{\mathcal{D}}
                    \Big)A_j^{N} &\forall j=1,\ldots,K
              \end{align}

        Implementation:

        .. math::
              \begin{align}
                A_j^{L} &= \mathcal{F}^{-1}\Big\{\exp\Big(\frac{h}{2}
                    \hat{\mathcal{D}}\Big)\mathcal{F}\{A_j(z)\}\Big\}
                    &\forall j=1,\ldots,K\\
                A_j^{N'} &= \exp\Big(\frac{h}{2}\hat{\mathcal{N}}
                    \big(A_1^{L}, \ldots, A_K^{L}\big)\Big)  A_j^{L}
                    &\forall j=1,\ldots,K\\
                A_j^{N} &= \exp\Big(\frac{h}{2}\hat{\mathcal{N}}\big(
                    A_1^{N'}, \ldots, A_{j}^{N'}, A_{j+1}^{N},\ldots,
                    A_K^{N}\big)\Big)  A_j^{N'} &\forall j=K,\ldots,1\\
                A_j(z+h) &=  \mathcal{F}^{-1}\Big\{\exp\Big(\frac{h}{2}
                    \hat{\mathcal{D}}\Big)\mathcal{F}\{A_j^{N}\}\Big\}
                    &\forall j=1,\ldots,K
              \end{align}

        where :math:`K` is the number of channels.

        """
        h_h = 0.5 * h
        A_L = np.zeros_like(waves)
        for i in range(len(waves)):
            A_L[i] = f.exp_term_lin(waves, i, z, h_h)
        for i in range(len(waves)-1):
            waves[i] = f.exp_term_non_lin(A_L, i, z, h_h, A_L[i])
        waves[-1] = f.exp_term_non_lin(A_L, len(A_L)-1, z, h, A_L[-1])
        for i in range(len(waves)-2, -1, -1):
            waves[i] = f.exp_term_non_lin(waves, i, z, h_h, waves[i])
        for i in range(len(waves)):
            waves[i] = f.exp_term_lin(waves, i, z, h_h)

        return waves
    # ==================================================================
    @staticmethod
    def ssfm_opti_super_sym(f: AbstractFieldEquation, waves: np.ndarray,
                            z: float, h: float) -> np.ndarray:
        r"""Optimized super symmetric split step Fourier method.

        Parameters
        ----------
        f :
            The function to compute.
        waves :
            The value of the unknown (waves) at the considered
            space step.
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

        Equations:

        .. math::
              \begin{align}
                A_j^{L} &= \exp\Big(\frac{h}{2}\mathcal{D}\Big)A_j(z, T)
                    &\forall j=1,\ldots,K\\
                A_j^{N'} &= A_j^{L} +\frac{h}{2}\bar{\mathcal{N}}\big(
                    A_1^{N'},\ldots, A_{j-1}^{N'}, A_{j}^{L}, \ldots,
                    A_K^{L}\big) &\forall j=1,\ldots,K\\
                A_j^{N} &= A_j^{N'} +\frac{h}{2}\bar{\mathcal{N}}\big(
                    A_1^{N'},\ldots, A_{j}^{N'}, A_{j+1}^{N}, \ldots,
                    A_K^{N}\big)&\forall j=K,\ldots,1\\
                A_j(z+h, T) &=  \exp\Big(\frac{h}{2}\mathcal{D}\Big)
                    A_j^{N} &\forall j=1,\ldots,K
              \end{align}

        Implementation:

        .. math::
              \begin{align}
                A_j^{L} &= \mathcal{F}^{-1}\Big\{\exp\Big(\frac{h}{2}
                    \hat{\mathcal{D}}\Big)\mathcal{F}\{A_j(z)\}\Big\}
                    &\forall j=1,\ldots,K\\
                A_j^{N'} &= \exp\Big(\frac{h}{2}\hat{\mathcal{N}}
                    \big(A_1^{N'},\ldots, A_{j-1}^{N'}, A_{j}^{L},
                    \ldots, A_K^{L}\big)\Big)  A_j^{L}
                    &\forall j=1,\ldots,K\\
                A_j^{N} &= \exp\Big(\frac{h}{2}\hat{\mathcal{N}}
                    \big(A_1^{N'},\ldots, A_{j}^{N'}, A_{j+1}^{N},
                    \ldots, A_K^{N}\big)\Big)  A_j^{N'}
                    &\forall j=K,\ldots,1 \\
                A_j(z+h) &=  \mathcal{F}^{-1}\Big\{\exp\Big(\frac{h}{2}
                    \hat{\mathcal{D}}\Big)\mathcal{F}\{A_j^{N}\}\Big\}
                    &\forall j=1,\ldots,K
              \end{align}

        where :math:`K` is the number of channels.

        """
        h_h = 0.5 * h
        for i in range(len(waves)):
            waves[i] = f.exp_term_lin(waves, i, z, h_h)
        for i in range(len(waves)-1):
            waves[i] = f.exp_term_non_lin(waves, i, z, h_h, waves[i])
        waves[-1] = f.exp_term_non_lin(waves, len(waves)-1, z, h, waves[-1])
        for i in range(len(waves)-2, -1, -1):
            waves[i] = f.exp_term_non_lin(waves, i, z, h_h, waves[i])
        for i in range(len(waves)):
            waves[i] = f.exp_term_lin(waves, i, z, h_h)

        return waves
    # ==================================================================
    @staticmethod
    def ssfm_truncated(f: AbstractFieldEquation, waves: np.ndarray,
                       z: float, h: float) -> np.ndarray:
        pass
    # ==================================================================
    @staticmethod
    def ssfm_central_diff(f: AbstractFieldEquation, waves: np.ndarray,
                          z: float, h: float) -> np.ndarray:
        pass
    # ==================================================================
    @staticmethod
    def ssfm_upwind(f: AbstractFieldEquation, waves: np.ndarray, z: float,
                    h: float) -> np.ndarray:
        pass
    # ==================================================================
    @staticmethod
    def rk4ip(f: AbstractFieldEquation, waves: np.ndarray, z: float,
              h: float) -> np.ndarray:
        r"""Runge-Kutta interaction picture method.

        Parameters
        ----------
        f :
            The function to compute.
        waves :
            The value of the unknown (waves) at the considered
            space step.
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

        Equations:

        .. math::
              \begin{align}
                A^L_j &=\exp\Big(\frac{h}{2}\mathcal{D}\Big)A_j(z,T)
                    &\forall j=1,\ldots,K\\
                k_{0,j} &=\exp\Big(\frac{h}{2}\mathcal{D}\Big) \Big(h
                    \bar{\mathcal{N}}\big(A_1(z,T), \ldots, A_j(z,T),
                    \ldots,A_K(z,T)\big)\Big) &\forall j=1,\ldots,K\\
                k_{1,j} &=h \bar{\mathcal{N}}\Big(A^L_{1}
                    + \frac{k_{0,1}}{2},\ldots, A^L_{j}
                    + \frac{k_{0,j}}{2}, \ldots, A^L_{K}
                    + \frac{k_{0,K}}{2}\Big) &\forall j=1,\ldots,K\\
                k_{2,j} &=h \bar{\mathcal{N}}\Big(A^L_{1}
                    + \frac{k_{1,1}}{2},\ldots, A^L_{j}
                    + \frac{k_{1,j}}{2}, \ldots, A^L_{K}
                    + \frac{k_{1,K}}{2}\Big) &\forall j=1,\ldots,K\\
                k_{3,j} &=h \bar{\mathcal{N}}\Big(\exp\Big(\frac{h}{2}
                    \mathcal{D}\Big)(A^L_{1} + k_{2,1}, \ldots, A^L_{j}
                    + k_{2,j}, \ldots, A^L_{K} + k_{2,K})\Big)
                    &\forall j=1,\ldots,K\\
                A_j(z+h,T) &=\frac{k_{3,j}}{6}+\exp\Big(\frac{h}{2}
                    \mathcal{D}\Big)\Big(A^L_{j}+\frac{k_{0,j}}{6}
                    +\frac{k_{1,j}}{3}+\frac{k_{2,j}}{3}\Big)
                    \quad &\forall j=1,\ldots,K
              \end{align}

        Implementation:

        .. math::
              \begin{align}
                A^L_{j} &= \mathcal{F}^{-1}\Big\{\exp\Big(\frac{h}{2}
                    \hat{\mathcal{D}}\Big)\mathcal{F}\{A_j(z)\}\Big\}
                    &\forall j=1,\ldots,K \\
                k_{0,j} &= \mathcal{F}^{-1}\Big\{\exp\Big(\frac{h}{2}
                    \hat{\mathcal{D}}\Big)\mathcal{F}\big\{h
                    \hat{\mathcal{N}}\big(A_1(z), \ldots, A_j(z),
                    \ldots,A_K(z)\big)\big\}\Big\}
                    &\forall j=1,\ldots,K\\
                k_{1,j} &= h \hat{\mathcal{N}}\Big(A^L_{1}
                    + \frac{k_{0,1}}{2}, \ldots, A^L_{j}
                    + \frac{k_{0,j}}{2}, \ldots, A^L_{K}
                    + \frac{k_{0,K}}{2} \Big) &\forall j=1,\ldots,K\\
                k_{2,j} &= h \hat{\mathcal{N}}\Big(A^L_{1}
                    + \frac{k_{1,1}}{2}, \ldots, A^L_{j}
                    + \frac{k_{1,j}}{2}, \ldots, A^L_{K}
                    + \frac{k_{1,K}}{2} \Big) &\forall j=1,\ldots,K\\
                k_{3,j} &= h \hat{\mathcal{N}}\Big(\mathcal{F}^{-1}
                    \Big\{\exp\Big(\frac{h}{2}\hat{\mathcal{D}}\Big)
                    \mathcal{F}\{A^L_{1} + k_{2,1}\}
                    \Big\} , \ldots, \nonumber \\
                    & \qquad \qquad \mathcal{F}^{-1}\Big\{\exp\Big(
                    \frac{h}{2}\hat{\mathcal{D}}\Big)\mathcal{F}
                    \{A^L_{j} + k_{2,j}\}\Big\} ,\ldots, \nonumber \\
                & \qquad \qquad \mathcal{F}^{-1}\Big\{\exp\Big(
                    \frac{h}{2}\hat{\mathcal{D}} \Big)\mathcal{F}
                    \{A^L_{K} + k_{2,K}\}\Big\} \Big)
                    &\forall j=1,\ldots,K\\
                A_j(z+h) &= \frac{k_{3,j}}{6}+\mathcal{F}^{-1}\Big\{
                    \exp\Big(\frac{h}{2}\hat{\mathcal{D}}\Big)
                    \mathcal{F}\big\{A^L_{j}
                    +\frac{k_{0,j}}{6}+\frac{k_{1,j}}{3}
                    +\frac{k_{2,j}}{3}\big\}\Big\} \quad
                    &\forall j=1,\ldots,K
              \end{align}

        where :math:`K` is the number of channels.

        """
        h_h = 0.5 * h
        A_L = np.zeros_like(waves)
        k_0 = np.zeros_like(waves)
        k_1 = np.zeros_like(waves)
        k_2 = np.zeros_like(waves)
        k_3 = np.zeros_like(waves)
        for i in range(len(waves)):
            A_L[i] = f.exp_term_lin(waves, i, z, h_h)
        for i in range(len(waves)):
            waves[i] = h * f.term_non_lin(waves, i, z)
        for i in range(len(waves)):
            k_0[i] = f.exp_term_lin(waves, i, z, h_h)
        for i in range(len(waves)):
            waves[i] = A_L[i] + 0.5*k_0[i]
        for i in range(len(waves)):
            k_1[i] = h * f.term_non_lin(waves, i, z)
        for i in range(len(waves)):
            waves[i] = A_L[i] + 0.5*k_1[i]
        for i in range(len(waves)):
            k_2[i] = h * f.term_non_lin(waves, i, z)
        for i in range(len(waves)):
            waves[i] = A_L[i] + k_2[i]
            waves[i] = f.exp_term_lin(waves, i, z, h_h)
        for i in range(len(waves)):
            k_3[i] = h * f.term_non_lin(waves, i, z)
        for i in range(len(waves)):
            waves[i] = A_L[i] + (k_0[i]/6.0) + ((k_1[i]+k_2[i])/3.0)
            waves[i] = (k_3[i]/6.0) + f.exp_term_lin(waves, i, z, h_h)

        return waves
    # ==================================================================
    @staticmethod
    def rk4ip_gnlse(f: AbstractFieldEquation, waves: np.ndarray, z: float,
                    h: float) -> np.ndarray:
        r"""Optimized Runge-Kutta interaction picture method.

        Parameters
        ----------
        f :
            The function to compute.
        waves :
            The value of the unknown (waves) at the considered
            space step.
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

        Implementation:

        .. math::
              \begin{align}
                &A^L_j = \exp\Big(\frac{h}{2}\hat{\mathcal{D}}\Big)
                    \mathcal{F}\{A_j(z)\} &\forall j=1,\ldots,K\\
                &k_0 = h \exp\Big(\frac{h}{2}\hat{\mathcal{D}}\Big)
                    \hat{\mathcal{N}}_0\big(A_1(z), \ldots, A_j(z),
                    \ldots, A_K(z)\big)&\forall j=1,\ldots,K\\
                &k_1 = h \hat{\mathcal{N}}_0\Big(\mathcal{F}^{-1}
                    \Big\{A^L_{1} +\frac{k_{0,1}}{2},\ldots,
                    A^L_{j}+\frac{k_{0,j}}{2},\ldots, A^L_{K}
                    + \frac{k_{0,K}}{2}\Big\} \Big)
                    &\forall j=1,\ldots,K\\
                &k_2 = h \hat{\mathcal{N}}_0\Big(\mathcal{F}^{-1}
                    \Big\{A^L_{1} +\frac{k_{1,1}}{2},\ldots, A^L_{j}
                    +\frac{k_{1,j}}{2},\ldots, A^L_{K}
                    + \frac{k_{1,K}}{2}\Big\} \Big)
                    &\forall j=1,\ldots,K\\
                &k_3 = h \hat{\mathcal{N}}_0\Big(\mathcal{F}^{-1}
                    \Big\{\exp\Big(\frac{h}{2}\hat{\mathcal{D}}\Big)
                    (A^L_1 + k_{2, 1}) \Big\},\ldots, \nonumber \\
                & \qquad \qquad \quad \mathcal{F}^{-1}\Big\{\exp
                    \Big(\frac{h}{2}\hat{\mathcal{D}}\Big)(A^L_j
                    + k_{2,j}) \Big\}, \ldots,\nonumber\\
                & \qquad \qquad \quad \mathcal{F}^{-1}\Big\{\exp\Big(
                    \frac{h}{2}\hat{\mathcal{D}}\Big)(A^L_K + k_{2,K})
                    \Big\}\Big)&\forall j=1,\ldots,K\\
                &A(z+h) = \mathcal{F}^{-1}\Big\{\frac{k_{3,j}}{6}
                    +\exp\Big(\frac{h}{2}\hat{\mathcal{D}}\Big)
                    \big(A^L_{j}+\frac{k_{0,j}}{6}+\frac{k_{1,j}}{3}
                    +\frac{k_{2,j}}{3}\big)\Big\} &\forall j=1,\ldots,K
              \end{align}

        where :math:`K` is the number of channels.

        """
        if (isinstance(f, GNLSE) or isinstance(f, AmpGNLSE)):
            h_h = 0.5 * h
            exp_op_lin = np.zeros_like(waves)
            A_L = np.zeros_like(waves)
            k_0 = np.zeros_like(waves)
            k_1 = np.zeros_like(waves)
            k_2 = np.zeros_like(waves)
            k_3 = np.zeros_like(waves)
            for i in range(len(waves)):
                exp_op_lin[i] = f.exp_op_lin(waves, i, h_h)
            for i in range(len(waves)):
                A_L[i] = exp_op_lin[i] * FFT.fft(waves[i])
            for i in range(len(waves)):
                k_0[i] = h * exp_op_lin[i] * f.term_rk4ip_non_lin(waves, i, z)
            for i in range(len(waves)):
                waves[i] = FFT.ifft(A_L[i] + 0.5*k_0[i])
            for i in range(len(waves)):
                k_1[i] = h * f.term_rk4ip_non_lin(waves, i, z)
            for i in range(len(waves)):
                waves[i] = FFT.ifft(A_L[i] + 0.5*k_1[i])
            for i in range(len(waves)):
                k_2[i] = h * f.term_rk4ip_non_lin(waves, i, z)
            for i in range(len(waves)):
                waves[i] = FFT.ifft(exp_op_lin[i] * (A_L[i] + k_2[i]))
            for i in range(len(waves)):
                k_3[i] = h * f.term_rk4ip_non_lin(waves, i, z)
            for i in range(len(waves)):
                waves[i] = ((k_3[i]/6.0)
                            + (exp_op_lin[i] * (A_L[i] + k_0[i]/6.0
                                                + (k_1[i]+k_2[i])/3.0)))
                waves[i] = FFT.ifft(waves[i])
        else:

            raise RK4IPGNLSEError("Only the the gnlse can be computed with "
                "the rk4ip_gnlse method.")

        return waves



if __name__ == "__main__":

    from typing import List, Optional

    import numpy as np

    import optcom as oc

    plot_groups: List[int] = []
    line_labels: List[Optional[str]] = []
    plot_titles: List[str] = []
    x_datas: List[np.ndarray] = []
    y_datas: List[np.ndarray] = []

    nlse_methods: List[str] = ["ssfm", "ssfm_reduced", "ssfm_symmetric",
                               "ssfm_opti_reduced", "ssfm_super_sym",
                               "ssfm_opti_super_sym", "rk4ip", "rk4ip"]
    # ---------------- NLSE solvers test -------------------------------
    lt: oc.Layout = oc.Layout(oc.Domain(bit_width=100.0, samples_per_bit=4096))

    pulse: oc.Gaussian = oc.Gaussian(channels=2, peak_power=[0.5, 1.0], width=[0.5, 0.8])

    steps: int = int(5e3)
    fiber: oc.Fiber
    SS: bool = True
    for j, nlse_method in enumerate(nlse_methods):
        if (j == len(nlse_methods)-2):  # To compute rk4ip and rk4ip_gnlse
            oc.set_rk4ip_opti_gnlse(False)   # Can make slighty diff. output
        else:
            oc.set_rk4ip_opti_gnlse(True)
        # Propagation
        fiber = oc.Fiber(length=0.2, nlse_method=nlse_method, alpha=[0.5],
                      beta_order=3, gamma=4.0, nl_approx=False, SPM=True,
                      XPM=True, SS=True, RS=True, steps=steps, save=True)
        lt.add_link(pulse[0], fiber[0])
        lt.run(pulse)
        lt.reset()
        # Plot parameters and get waves
        x_datas.append(fiber[1][0].time)
        y_datas.append(oc.temporal_power(fiber[1][0].channels))
        plot_groups.append(0)

    line_labels.extend(nlse_methods[:-1] + ["rk4ip_gnlse"])
    plot_titles.extend(["NLSE solvers test with n={}".format(str(steps))])
    # -------------------- Plotting results ------------------------
    oc.plot2d(x_datas, y_datas, plot_groups=plot_groups,
              plot_titles=plot_titles, x_labels=['t'], y_labels=['P_t'],
              line_labels=line_labels, line_opacities=[0.3])
