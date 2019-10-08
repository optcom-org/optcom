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
import math
from typing import Dict, List, Optional, overload, Tuple, Union

import numpy as np
from nptyping import Array

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.domain import Domain
from optcom.equations.abstract_refractive_index import AbstractRefractiveIndex

# ref for f values and UV: Arkwright, J.W., Elango, P., Atkins, G.R.,
# Whitbread,  T. and Digonnet, M.J., 1998. Experimental and theoretical
# analysis of  the resonant nonlinearity in ytterbium-doped fiber.
# Journal of lightwave technology, 16(5), p.803.
media = ["yb"]
# esa_values = {medium: (f_ESA, UV ESA), ...} # in nm
esa_values: Dict[str, Tuple[float, List[float]]] =\
    {"yb": (0.529, [135.0, 123.0, 118.1, 117.6])}
# sa_values = {medium: (f_GSA, UV GSA), ...} # in nm
gsa_values: Dict[str, Tuple[float, List[float]]] =\
    {"yb": (0.666, [118.6, 109.3, 105.4, 105.0])}
# g_values = {medium: (g_1, g_2)}
# N.B.: ground and metastable state degeneracy factor
g_values: Dict[str, Tuple[float, float]] = {"yb": (8, 6)}
# main_trans_values = {medium: (f_12, \lambda_12, FWHM), ...} # in nm
# N.B.: FWHM is at \lambda_12 transition (ground-metastable states)
main_trans_values: Dict[str, Tuple[float, float, float]] =\
    {"yb": (5.08e-6, 980.0, 16.0)}


class ResonantIndex(AbstractRefractiveIndex):
    r"""Compute the resonant index change. [7]_

    References
    ----------
    .. [7] Digonnet, M.J., Sadowski, R.W., Shaw, H.J. and Pantell, R.H.,
           1997. Experimental evidence for strong UV transition
           contribution in the resonant nonlinearity of doped fibers.
           Journal of lightwave technology, 15(2), pp.299-303.

    """

    def __init__(self, medium: str = "yb"):
        super().__init__(medium)
        if (self._medium in media):
            self._main_trans = main_trans_values[self._medium]
            self._gs = g_values[self._medium]
            self._esa = esa_values[self._medium]
            self._gsa = gsa_values[self._medium]
    # ==================================================================
    @overload
    def calc_Q(self, n_0: float) -> float: ...
    # ------------------------------------------------------------------
    @overload
    def calc_Q(self, n_0: Array[float]) -> Array[float]: ...
    # ------------------------------------------------------------------
    def calc_Q(self, n_0):
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
        den = 8.0 * n_0 * cst.M_E * cst.EPS_0 * cst.LIGHT_SPEED

        return num / den
    # ==================================================================
    @overload
    def lorentzian_lineshape(self, lambda_s: float, Lambda: float,
                              lambda_bw: float) -> float: ...
    # ------------------------------------------------------------------
    @overload
    def lorentzian_lineshape(self, lambda_s: Array[float],
                              Lambda: Array[float], lambda_bw: float
                              ) -> Array[float]: ...
    # ------------------------------------------------------------------
    def lorentzian_lineshape(self, lambda_s, Lambda, lambda_bw):
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
        factor = 1.0 / (cst.LIGHT_SPEED*cst.PI)
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
    @overload
    def n(self, omega: float, n_0: float, N_1: float) -> float: ...
    # ------------------------------------------------------------------
    @overload
    def n(self, omega: Array[float], n_0: Array[float], N_1: float
          ) -> Array[float]: ...
    # ------------------------------------------------------------------
    def n(self, omega, n_0, N_1):
        r"""Compute the resonant index change.

        Parameters
        ----------
        omega :
            The angular frequency. :math:`[rad\cdot ps^{-1}]`

        Returns
        -------
        :
            The refractive index change.

        Notes
        -----

        .. math:: \Delta n = Q N_1 S

        where:

        .. math:: S = -f_{12}\lambda_{12}g'_{12}(\lambda_s)
                  \big(1 + \frac{g_1}{g_2}\big) - \sum_{j>2} f_{1j}
                  \lambda_{1j}g'_{1j}(\lambda_s)
                  + \sum_{j>2} f_{2j}\lambda_{2j}g'_{2j}(\lambda_s)

        """
        if (isinstance(omega, float)):
            gsa = 0.0
            esa = 0.0
            res = 0.0
        else:
            gsa = np.zeros_like(omega)
            esa = np.zeros_like(omega)
            res = np.zeros_like(omega)

        lambda_s = Domain.omega_to_lambda(omega)
        main_trans = (self._main_trans[0] * self._main_trans[1]
                      * self.lorentzian_lineshape(lambda_s,
                                                  self._main_trans[1],
                                                  self._main_trans[2])
                      * (1 + (self._gs[0]/self._gs[1])))

        # Need to verify for the lambda bw to use here
        # should not change a lot -> very small term
        for uv_gsa in self._gsa[1]:
            gsa += (self._gsa[0] * uv_gsa
                    * self.lorentzian_lineshape(lambda_s, uv_gsa,
                                                self._main_trans[2]))
        for uv_esa in self._esa[1]:
            esa += (self._esa[0] * uv_esa
                    * self.lorentzian_lineshape(lambda_s, uv_esa,
                                                self._main_trans[2]))

        res = self.calc_Q(n_0) * N_1 * (esa - gsa - main_trans)

        return res


if __name__ == "__main__":

    import optcom.utils.plot as plot
    from optcom.domain import Domain
    from optcom.equations.sellmeier import Sellmeier

    n_0 = 1.43  #
    N_1 = 0.03  # nm^-3
    omega = Domain.lambda_to_omega(1050.0)

    resonant = ResonantIndex("yb")
    print(resonant.n(omega, n_0, N_1))

    Lambda = np.linspace(500, 1500, 1000)
    omega = Domain.lambda_to_omega(Lambda)
    delta_n = resonant.n(omega, n_0, N_1)
    delta_n_data = [delta_n]

    x_labels = ['Lambda']
    y_labels = ['Index change']
    plot_titles = ['Resonant index change with constant n_0 = {} (pumped)'
                   .format(n_0)]

    sellmeier = Sellmeier("siO2")
    n_0 = sellmeier.n(omega)
    delta_n = resonant.n(omega, n_0, N_1)
    delta_n_data.append(delta_n)

    x_labels.append('Lambda')
    y_labels.append('Index change')
    plot_titles.append("Resonant index change with n_0 from Silica Sellmeier "
                       "equations. (pumped)")

    plot.plot([Lambda, Lambda], delta_n_data, x_labels=x_labels,
              y_labels=y_labels, plot_titles=plot_titles, split=True,
              opacity=0.0)
