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
from typing import Callable, List, Optional, overload

import numpy as np
from nptyping import Array

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.domain import Domain
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
                 n_0: float = cst.REF_INDEX) -> None:
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
        n_0 :
            The refractive index outside of the two fiber cores.

        """
        super().__init__()
        self._kappa: List[float]
        self._predict: Optional[Callable] = None
        self._V: float = V
        self._a: float = a
        self._d: float = d
        self._n_0: float = n_0
        if (kappa is not None):
            kappa = util.make_list(kappa)   # make sure is list
            self._kappa = np.asarray(kappa).reshape((1,-1))
        else:
            self._predict = self.get_kappa
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
    @property
    def center_omega(self) -> Optional[Array[float]]:

        return self._center_omega
    # ==================================================================
    @center_omega.setter
    def center_omega(self, center_omega: Array[float]) -> None:
        # Overloading to upload the betas(\omega)
        self.update(center_omega)
        self._center_omega = center_omega
    # ==================================================================
    def update(self, center_omega: Optional[Array[float]] = None) -> None:
        if (center_omega is None):
            center_omega = self._center_omega
        if (self._predict is not None):
            self._kappa = np.zeros((len(center_omega), 1))
            self._kappa = self._predict(center_omega).reshape((-1,1))
        else:
            if (len(center_omega) < len(self._kappa)):
                self._kappa = self._kappa[:len(center_omega)]
            else:
                for i in range(len(self._kappa), len(center_omega)):
                    self._kappa = np.vstack((self._kappa, self._kappa[-1]))
    # ==================================================================
    def op(self, waves: Array[cst.NPFT], id: int,
           corr_wave: Optional[Array[cst.NPFT]] = None) -> Array[cst.NPFT]:
        """The operator of the dispersion effect."""

        op = Taylor.series(self._kappa[id], self._omega)

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
    @overload
    def get_kappa(self, omega: float) -> float: ...
    # ------------------------------------------------------------------
    @overload
    def get_kappa(self, omega: Array[float]) -> Array[float]: ...
    # ------------------------------------------------------------------
    def get_kappa(self, omega):

        return Coupling.calc_kappa(omega, self._V, self._a, self._d, self._n_0)
    # ==================================================================
    @overload
    @staticmethod
    def calc_kappa(omega: float, V: float, a: float, d: float, n_0: float
                   ) -> float: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def calc_kappa(omega: Array[float], V: float, a: float, d: float,
                   n_0: float) -> Array[float]: ...
    # ------------------------------------------------------------------
    @staticmethod
    def calc_kappa(omega, V = cst.V, a = cst.CORE_RADIUS, d = cst.C2C_SPACING,
                   n_0 = cst.REF_INDEX):
        r"""Calculate the coupling coefficient for the parameters given
        for two waveguides. (assuming the two waveguides are
        symmetric) [7]_

        Parameters
        ----------
        omega :
            The angular frequency.  :math:`[rad\cdot ps^{-1}]`
        V :
            The fiber parameter.
        a :
            The core radius. :math:`[\mu m]`
        d :
            The center to center spacing between the two cores.
            :math:`[\mu m]`
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
        lambda_0 = Domain.omega_to_lambda(omega)
        a *= 1e-9   # um -> km
        d *= 1e-9   # um -> km
        lambda_0 *= 1e-12   # nm -> km
        norm_d = d/a
        k_0  = 2*cst.PI/lambda_0
        c_0 = 5.2789 - 3.663*V + 0.3841*V**2
        c_1 = -0.7769 + 1.2252*V - 0.0152*V**2
        c_2 = -0.0175 - 0.0064*V - 0.0009*V**2
        # Check validity of formula paramater --------------------------
        if (V < 1.5 or V > 2.5):
            util.warning_terminal("Value of the fiber parameter is V = {}, "
                "kappa fitting formula is valid only for : 1.5 <= V <= 2.5, "
                "might lead to unrealistic results.".format(V))
        if (norm_d < 2.0 or norm_d > 4.5):
            util.warning_terminal("Value of the normalized spacing is d = {}, "
                "kappa fitting formula is valid only for : 2.0 <= d <= 4.5, "
                "might lead to unrealistic results.".format(norm_d))
        # Formula ------------------------------------------------------
        if (isinstance(omega, float)):
            res = (cst.PI*V / (2*k_0*n_0*a**2)
                   * math.exp(-(c_0 + c_1*norm_d + c_2*norm_d**2)))
        else:
            res = (cst.PI*V / (2*k_0*n_0*a**2)
                   * np.exp(-(c_0 + c_1*norm_d + c_2*norm_d**2)))

        return res


if __name__ == "__main__":

    import optcom.utils.plot as plot
    from optcom.domain import Domain

    V = 2.0
    a = 5.0
    d = 15.0
    n_0 = 1.02
    norm_d = d/a
    cpl = Coupling(V=V, a=a, d=d, n_0=n_0)

    Lambda = 1550.0
    omega = Domain.lambda_to_omega(Lambda)
    print(cpl.get_kappa(omega))

    Lambda = np.linspace(900, 1600, 3)
    omega = Domain.lambda_to_omega(Lambda)
    print(cpl.get_kappa(omega))


    Lambda = np.linspace(900, 1600, 1000)
    omega = Domain.lambda_to_omega(Lambda)
    kappas = cpl.get_kappa(omega)

    plot_titles = ['Coupling coefficient as a function of the wavelength '
                   'for norm. spacing = {}'.format(norm_d)]

    plot.plot(Lambda, kappas, x_labels=['Lambda'], y_labels=['Kappa km^{-1}'],
              plot_titles=plot_titles, split=True, opacity=0.0)
