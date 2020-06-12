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
import math
from typing import Callable, Dict, List, Optional, overload, Tuple,\
                   Union

import numpy as np

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.domain import Domain
from optcom.parameters.abstract_parameter import AbstractParameter
from optcom.utils.callable_container import CallableContainer

# ref for f values and UV: Arkwright, J.W., Elango, P., Atkins, G.R.,
# Whitbread,  T. and Digonnet, M.J., 1998. Experimental and theoretical
# analysis of  the resonant nonlinearity in ytterbium-doped fiber.
# Journal of lightwave technology, 16(5), p.803.
MEDIA: List[str] = ["yb"]
# esa_values = {medium: (f_ESA, UV ESA), ...} # in nm
esa_values: Dict[str, Tuple[float, List[float]]]
esa_values = {"yb": (0.529, [135.0, 123.0, 118.1, 117.6])}
# sa_values = {medium: (f_GSA, UV GSA), ...} # in nm
gsa_values: Dict[str, Tuple[float, List[float]]]
gsa_values = {"yb": (0.666, [118.6, 109.3, 105.4, 105.0])}
# g_values = {medium: (g_1, g_2)}
# N.B.: ground and metastable state degeneracy factor
g_values: Dict[str, Tuple[float, float]] = {"yb": (8., 6.)}
# main_trans_values = {medium: (f_12, \lambda_12, FWHM), ...} # in nm
# N.B.: FWHM is at \lambda_12 transition (ground-metastable states)
main_trans_values: Dict[str, Tuple[float, float, float]]
main_trans_values = {"yb": (5.08e-6, 980.0, 16.0)}


class ResonantIndex(AbstractParameter):
    r"""Compute the resonant index change. [11]_

    References
    ----------
    .. [11] Digonnet, M.J., Sadowski, R.W., Shaw, H.J. and Pantell,
            R.H., 1997. Experimental evidence for strong UV transition
            contribution in the resonant nonlinearity of doped fibers.
            Journal of lightwave technology, 15(2), pp.299-303.

    """

    def __init__(self, medium: str, n_0: Union[float, Callable],
                 N: Union[float, Callable]) -> None:
        r"""
        Parameters
        ----------
        medium :
            The medium in which the wave propagates.
        n_0 :
            The raw refratvie index of the medium.  If a
            callable is provided, the variable must be angular
            frequency. :math:`[ps^{-1}]`
        N :
            The population of the metastable level. :math:`[nm^{-3}]`
            If a callable is provided, the variable must be angular
            frequency. :math:`[ps^{-1}]`

        """
        self._medium: str = util.check_attr_value(medium.lower(), MEDIA,
                                                  cst.DEF_FIBER_DOPANT)
        self._n_0: Union[float, Callable] = n_0
        self._N: Union[float, Callable] = N
        self._main_trans = main_trans_values[self._medium]
        self._gs = g_values[self._medium]
        self._esa = esa_values[self._medium]
        self._gsa = gsa_values[self._medium]
    # ==================================================================
    @property
    def n_0(self) -> Union[float, Callable]:

        return self._n_0
    # ------------------------------------------------------------------
    @n_0.setter
    def n_0(self, n_0: Union[float, Callable]) -> None:

        self._n_0 = n_0
    # ==================================================================
    @property
    def N(self) -> Union[float, Callable]:

        return self._N
    # ------------------------------------------------------------------
    @N.setter
    def N(self, N: Union[float, Callable]) -> None:
        self._N = N
    # ==================================================================
    @overload
    def __call__(self, omega: float) -> float: ...
    # ------------------------------------------------------------------
    @overload
    def __call__(self, omega: np.ndarray) -> np.ndarray: ...
    # ------------------------------------------------------------------
    def __call__(self, omega):
        r"""Compute the resonant index.

        Parameters
        ----------
        omega :
            The angular frequency. :math:`[rad\cdot ps^{-1}]`

        Returns
        -------
        :
            Value of the resonant index.

        """
        #print(self._N)
        fct = CallableContainer(ResonantIndex.calc_res_index,
                                [omega, self._n_0, self._N, self._main_trans,
                                 self._gs, self._esa, self._gsa])

        return fct(omega)
    # ==================================================================
    @staticmethod
    @overload
    def calc_Q(n_0: float) -> float: ...
    # ------------------------------------------------------------------
    @staticmethod
    @overload
    def calc_Q(n_0: np.ndarray) -> np.ndarray: ...
    # ------------------------------------------------------------------
    @staticmethod
    def calc_Q(n_0):
        r"""Compute the Q factor.

        Parameters
        ----------
        n_0 :
            The refractive index.

        Returns
        -------
        :
            The Q factor. :math:`[ps^{-1} nm^{2}]`

        Notes
        -----

        .. math:: Q = \frac{e^2 (n_0^2+2)^2}{72 n_0 m \epsilon_0 c}

        """
        if (isinstance(n_0, float)):
            num = cst.C_E**2 * (n_0**2 + 2.0)**2 / 9.0
        else:
            num = np.square(cst.C_E) * np.square((np.square(n_0) + 2.0)) / 9.0
        den = 8.0 * n_0 * cst.M_E * cst.EPS_0 * cst.C

        return num / den
    # ==================================================================
    @staticmethod
    @overload
    def lorentzian_lineshape(lambda_s: float, Lambda: float, lambda_bw: float
                             ) -> float: ...
    # ------------------------------------------------------------------
    @staticmethod
    @overload
    def lorentzian_lineshape(lambda_s: np.ndarray, Lambda: np.ndarray,
                             lambda_bw: float) -> np.ndarray: ...
    # ------------------------------------------------------------------
    @staticmethod
    def lorentzian_lineshape(lambda_s, Lambda, lambda_bw):
        r"""Compute the lorentzian lineshape.

        Parameters
        ----------
        lambda_s :
            The center wavelength of the wave signal. :math:`[nm]`
        Lambda :
            The reference wavelength. :math:`[nm]`
            (depends on the material)
        lambda_bw :
            The wavelength bandwidth. :math:`[nm]`

        Returns
        -------
        :
            The lorentzian lineshape. :math:`[ps]`

        Notes
        -----

        .. math::  g'_{ij}(\lambda_s) = \frac{1}{\pi c}
                   \frac{ \frac{1}{\lambda_{ij}} \Big(\frac{1}
                   {\lambda_{ij}^2}-\frac{1}{\lambda_{s}^2}\Big)}
                   {\Big(\frac{1}{\lambda_{ij}^2}
                   -\frac{1}{\lambda_{s}^2}\Big)
                   + \Big(\frac{\Delta\lambda_{ij}}
                   {\lambda_s \lambda_{ij}^2}\Big)}

        """
        factor = 1.0 / (cst.C*cst.PI)
        if (isinstance(Lambda, float)):
            diff = ((1.0 / Lambda**2) - (1.0 / lambda_s**2))
            den = (diff**2 + (lambda_bw/(Lambda**2)/lambda_s)**2)
        else:
            diff = ((1.0/np.square(Lambda)) - (1.0/np.square(lambda_s)))
            den = (np.square(diff)
                   + np.square(lambda_bw / np.square(Lambda) / lambda_s))
        num = (1.0 / Lambda) * diff

        return factor * num / den
    # ==================================================================
    @staticmethod
    @overload
    def calc_res_index(omega: float, n_0: float, N: float,
                       main_trans: Tuple[float, float, float],
                       gs: Tuple[float, float], esa: Tuple[float, List[float]],
                       gsa: Tuple[float, List[float]]) -> float: ...
    # ------------------------------------------------------------------
    @staticmethod
    @overload
    def calc_res_index(omega: np.ndarray, n_0: np.ndarray, N: float,
                       main_trans: Tuple[float, float, float],
                       gs: Tuple[float, float], esa: Tuple[float, List[float]],
                       gsa: Tuple[float, List[float]]) -> np.ndarray: ...
    # ------------------------------------------------------------------
    @staticmethod
    def calc_res_index(omega, n_0, N, main_trans, gs, esa, gsa):
        r"""Compute the resonant index change.

        Parameters
        ----------
        omega :
            The angular frequency. :math:`[rad\cdot ps^{-1}]`
        n_0 :
            The raw refratvie index of the medium.
        N :
            The population of the metastable level. :math:`[nm^{-3}]`
        main_trans :
            The main transition parameters. :math:`[nm]`
            [:math:`f_{12}`, :math:`\lambda_{12}`, FWHM]
        gs :
            The degeneracy factor of the metastable and ground state.
            (:math:`g_{ground}, g_{metastable}`)
        esa :
            The transition strength of the metastable state and the
            relevant UV levels. :math:`[nm]`
            (:math:`f_{esa}`, [:math:`UV_{esa,1}`, ...])
        gsa :
            The transition strength of the ground state and the
            relevant UV levels. :math:`[nm]`
            (:math:`f_{gsa}`, [:math:`UV_{gsa,1}`, ...])


        Returns
        -------
        :
            The refractive index change.

        Notes
        -----

        .. math:: \Delta n = Q N S

        where:

        .. math:: S = -f_{12}\lambda_{12}g'_{12}(\lambda_s)
                  \big(1 + \frac{g_1}{g_2}\big) - \sum_{j>2} f_{1j}
                  \lambda_{1j}g'_{1j}(\lambda_s)
                  + \sum_{j>2} f_{2j}\lambda_{2j}g'_{2j}(\lambda_s)

        """
        if (isinstance(omega, float)):
            gsa_ = 0.0
            esa_ = 0.0
        else:
            gsa_ = np.zeros_like(omega)
            esa_ = np.zeros_like(omega)

        lambda_s = Domain.omega_to_lambda(omega)
        ln = ResonantIndex.lorentzian_lineshape(lambda_s, main_trans[1],
                                                main_trans[2])
        main_trans_ = main_trans[0] * main_trans[1] * ln * (1 + (gs[0]/gs[1]))

        # Need to verify for the lambda bw to use here
        # should not change a lot -> very small term
        for uv_gsa in gsa[1]:
            ln = ResonantIndex.lorentzian_lineshape(lambda_s, uv_gsa,
                                                    main_trans[2])
            gsa_ += gsa[0] * uv_gsa * ln
        for uv_esa in esa[1]:
            ln = ResonantIndex.lorentzian_lineshape(lambda_s, uv_esa,
                                                    main_trans[2])
            esa_ += esa[0] * uv_esa * ln

        res = ResonantIndex.calc_Q(n_0) * N * (esa_ - gsa_ - main_trans_)

        return res


if __name__ == "__main__":
    """Plot the resonant index as a function of the wavelength.
    This piece of code is standalone, i.e. can be used in a separate
    file as an example.
    """

    from typing import List

    import numpy as np

    import optcom as oc

    # With float
    n_0: float = 1.43  #
    N: float = 0.03  # nm^-3
    omega: float = oc.lambda_to_omega(1050.0)
    resonant: oc.ResonantIndex = oc.ResonantIndex("yb", n_0, N)
    print(resonant(omega))
    # With np.ndarray
    lambdas: np.ndarray = np.linspace(500., 1500., 1000)
    omegas: np.ndarray = oc.lambda_to_omega(lambdas)
    delta_n: np.ndarray = resonant(omegas)
    delta_n_data: List[np.ndarray] = [delta_n]
    x_labels: List[str] = ['Lambda']
    y_labels: List[str] = ['Index change']
    plot_titles: List[str] = ["Resonant index change with constant n_0 = {} "
                              "(pumped)".format(n_0)]
    sellmeier: oc.Sellmeier = oc.Sellmeier("siO2")
    resonant = oc.ResonantIndex("yb", sellmeier, N)
    delta_n = resonant(omegas)
    delta_n_data.append(delta_n)
    x_labels.append('Lambda')
    y_labels.append('Index change')
    plot_titles.append("Resonant index change with n_0 from Silica Sellmeier "
                       "equations. (pumped)")

    oc.plot2d([lambdas, lambdas], delta_n_data, x_labels=x_labels,
              y_labels=y_labels, plot_titles=plot_titles, split=True,
              line_opacities=[0.0])
