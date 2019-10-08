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

import math
from typing import List, Optional

import numpy as np
from nptyping import Array

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.effects.abstract_effect import AbstractEffect
from optcom.utils.taylor import Taylor


class Coupling(AbstractEffect):
    r"""Generic class for effect object.

    Attributes
    ----------
    omega : numpy.ndarray of float
        The angular frequency array. :math:`[ps^{-1}]`
    time : numpy.ndarray of float
        The time array. :math:`[ps]`
    center_omega : numpy.ndarray of float
        The center angular frequency. :math:`[ps^{-1}]`

    """

    def __init__(self, kappa: Optional[List[float]] = None,
                 V: float = cst.V, a: float = cst.CORE_RADIUS,
                 d: float = cst.C2C_SPACING,
                 lambda_0: float = cst.DEF_LAMBDA,
                 n_0: float = cst.REF_INDEX,
                 omega: Optional[Array[float]] = None) -> None:
        r"""
        Parameters
        ----------
        kappa :
            The coupling coefficients. :math:`[km^{-1}]`
        V :
            The fiber parameter.
        a :
            The core radius. :math:`[\mu m]`
        d :
            The center to center spacing between the two cores.
            :math:`[\mu m]`
        lambda_0 :
            The wavelength in the vacuum for the considered wave.
            :math:`[nm]`
        n_0 :
            The refractive index outside of the two fiber cores.
        omega :
            The angular frequency. :math:`[ps^{-1}]`

        """
        super().__init__(omega)
        self._kappa: List[float]
        if (kappa is None):
            self._kappa = [Coupling.calc_kappa(V, a, d, lambda_0, n_0)]
        else:
            self._kappa = util.make_list(kappa)
    # ==================================================================
    @property
    def kappa(self) -> List[float]:

        return self._kappa
    # ==================================================================
    @kappa.setter
    def kappa(self, kappa: List[float]) -> None:
        self._kappa = kappa
    # ==================================================================
    def __getitem__(self, key: int) -> float:

        return self._kappa[key]
    # ==================================================================
    def __setitem__(self, key: int, kappa_key: float) -> None:
        self._kappa[key] = kappa_key
    # ==================================================================
    def __delitem__(self, key: int) -> None:
        self._kappa[key] = 0.0
    # ==================================================================
    def __len__(self) -> int:

        return len(self._kappa)
    # ==================================================================
    def op(self, waves: Array[cst.NPFT], id: int,
           corr_wave: Optional[Array[cst.NPFT]] = None) -> Array[cst.NPFT]:
        """The operator of the dispersion effect."""

        op = Taylor.series(self._kappa, self._omega)
        print(self._kappa)
        return 1j * op
    # ==================================================================
    def term(self, waves: Array[cst.NPFT], id: int,
             corr_wave: Optional[Array[cst.NPFT]] = None) -> Array[cst.NPFT]:
        res = np.zeros(waves[id].shape, dtype=cst.NPFT)
        for i in range(len(waves)):
            if (i != id):
                res += waves[i]

        return self.op(waves, id, corr_wave) * res
    # ==================================================================
    @staticmethod
    def calc_kappa(V: float = cst.V, a: float = cst.CORE_RADIUS,
                   d: float = cst.C2C_SPACING,
                   lambda_0: float = cst.DEF_LAMBDA,
                   n_0: float = cst.REF_INDEX) -> float:
        r"""Calculate the coupling coefficient for the parameters given
        for two waveguides. (assuming the two waveguides are
        symmetric) [7]_

        Parameters
        ----------
        V :
            The fiber parameter.
        a :
            The core radius. :math:`[\mu m]`
        d :
            The center to center spacing between the two cores.
            :math:`[\mu m]`
        lambda_0 :
            The wavelength in the vacuum for the considered wave.
            :math:`[nm]`
        n_0 :
            The refractive index outside of the two fiber cores.

        Returns
        -------
        :
            Value of the coupling coefficient. :math:`[km^{-1}]`

        Notes
        -----

        .. math:: \kappa = \frac{\pi V}{2 k_0 n_0 a^2}
                            e^{-(c_0 + c_1 d + c_2 d^2)}

        This equation is accurate to within 1% for values of the fiber
        parameter :math:`1.5\leq V\leq 2.5` and of the normalized
        center-to-center spacing :math:`\bar{d} = d/a`,
        :math:`2\leq \bar{d}\leq 4.5` .

        References
        ----------
        .. [7] Govind Agrawal, Chapter 2: Fibers Couplers,
           Applications of Nonlinear Fiber Optics (Second Edition),
           Academic Press, 2008, Page 59.

        """
        a *= 1e-9
        d *= 1e-9
        lambda_0 *= 1e-12
        norm_d = a/d
        k_0  = 2*cst.PI/lambda_0
        c_0 = 5.2789 - 3.663*V + 0.3841*V**2
        c_1 = -0.7769 + 1.2252*V - 0.0152*V**2
        c_2 = -0.0175 - 0.0064*V - 0.0009*V**2

        kappa = (cst.PI*V / (2*k_0*n_0*a**2)
                 * math.exp(-(c_0 + c_1*norm_d + c_2*norm_d**2)))

        return kappa
