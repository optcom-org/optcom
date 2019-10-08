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

from typing import Callable, List, Optional, overload, Union

import numpy as np
from nptyping import Array
from scipy import interpolate

from optcom.utils.fft import FFT
import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.effects.abstract_effect import AbstractEffect
from optcom.equations.abstract_refractive_index import AbstractRefractiveIndex
from optcom.equations.sellmeier import Sellmeier
from optcom.utils.taylor import Taylor


class Dispersion(AbstractEffect):
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

    def __init__(self, beta: Optional[Union[List[float], Callable]] = None,
                 order: int = 2, medium: str = cst.DEF_FIBER_MEDIUM,
                 start_taylor: int = 2) -> None:
        r"""
        Parameters
        ----------
        beta :
            The derivatives of the propagation constant.
            :math:`[km^{-1}, ps\cdot km^{-1}, ps^2\cdot km^{-1},
            ps^3\cdot km^{-1}, \ldots]` If a callable is provided,
            variable must be angular frequency. :math:`[ps^{-1}]`
        order :
            The order of beta coefficients to take into account. (will
            be ignored if beta values are provided - no file)
        medium :
            The medium in which the dispersion is considered.
        start_taylor :
            The order of the derivative from which to start the Taylor
            Series expansion.

        """
        super().__init__()
        self._order: int =  order
        self._medium: str = medium
        self._start_taylor: int = start_taylor # 2 due to change of var
        self._predict: Optional[Callable] = None
        self._beta: Array[float]
        self._class_n: Optional[Callable] = None
        if (beta is not None):
            if (callable(beta)):
                self._predict = beta
            else:
                beta = util.make_list(beta)   # make sure is list
                self._beta = np.asarray(beta).reshape((1,-1))
                self._order = self._beta.shape[1] - 1
        else:
            self._predict = self.calc_beta_coeffs
            self._class_n = Sellmeier(medium)

    # ==================================================================
    def __call__(self, omega):
        return None
    # ==================================================================
    @property
    def beta(self) -> List[float]:

        return self._beta
    # ==================================================================
    @beta.setter
    def beta(self, beta: List[float]) -> None:
        self._beta = beta
    # ==================================================================
    def __getitem__(self, key: int) -> float:

        return self._beta[key]
    # ==================================================================
    def __setitem__(self, key: int, disp: float) -> None:
        self._beta[key] = disp
    # ==================================================================
    def __delitem__(self, key: int) -> None:
        self._beta = np.delete(self._beta, key, axis=0)
    # ==================================================================
    def __len__(self) -> int:

        return self._beta.shape[1]
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
    def update(self, center_omega: Optional[Array[float]] = None,
               class_n: Optional[Callable] = None) -> None:
        # Do no test if beta is None to be able to do multi pass
        if (center_omega is None):
            center_omega = self._center_omega
        if (self._predict is not None):
            self._beta = np.zeros((len(center_omega), self._order+1))
            if (class_n is not None):
                self._beta = self._predict(center_omega, self._order,
                                           class_n).T
            else:
                if (self._class_n is None):
                    self._beta = self._predict(center_omega, self._order).T
                else:
                    self._beta = self._predict(center_omega, self._order,
                                               self._class_n).T
        else:
            if (len(center_omega) < len(self._beta)):
                self._beta = self._beta[:len(center_omega)]
            else:
                for i in range(len(self._beta), len(center_omega)):
                    self._beta = np.vstack((self._beta, self._beta[-1]))
    # ==================================================================
    def op(self, waves: Array[cst.NPFT], id: int,
           corr_wave: Optional[Array[cst.NPFT]] = None) -> Array[cst.NPFT]:
        """The operator of the dispersion effect."""

        op = Taylor.series(self._beta[id], self._omega, self._start_taylor)

        return 1j * op
    # ==================================================================
    @overload
    @staticmethod
    def calc_beta_coeffs(omega: float, order: int,
                         class_n: Optional[AbstractRefractiveIndex]
                         ) -> List[float]: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def calc_beta_coeffs(omega: Array[float], order: int,
                         class_n: Optional[AbstractRefractiveIndex]
                         ) -> Array[float]: ...
    # ------------------------------------------------------------------
    @staticmethod
    def calc_beta_coeffs(omega, order, class_n):
        r"""Calcul the nth first derivatives of the propagation
        constant. (valid only for TEM)

        Parameters
        ----------
        omega :
            The angular frequency. :math:`[rad\cdot ps^{-1}]`
        order :
            The order of the highest dispersion coefficient.
        class_n : AbstractRefractiveIndex
            A function which calculate the refractive index.

        Returns
        -------
        :
            The nth derivative of the propagation constant.
            :math:`[ps^{i}\cdot km^{-1}]`

        Notes
        -----

        .. math:: \beta_i(\omega) = \frac{d^i \beta(\omega)}{d\omega^i}
                  = \frac{1}{c}\bigg(i
                  \frac{d^{(i-1)} n(\omega)}{d\omega^{(i-1)}}
                  + \omega \frac{d^i n(\omega)}{d\omega^i}\bigg)

        for :math:`i = 0, \ldots, \text{order}`

        """
        LIGHT_SPEED = cst.LIGHT_SPEED * 1e-12   # nm/ps -> km/ps
        if (isinstance(omega, float)):
            res = [0.0 for i in range(order+1)]
        else:
            res = np.zeros((order+1, len(omega)))
        prec_n_deriv = class_n.n(omega)
        res[0] = omega / LIGHT_SPEED * prec_n_deriv
        for i in range(1, order+1):
            current_n_deriv = class_n.n_deriv(omega, i)
            to_append = (i*prec_n_deriv + omega*current_n_deriv)/LIGHT_SPEED
            prec_n_deriv = current_n_deriv
            res[i] = to_append

        return res
    # ==================================================================
    @overload
    @staticmethod
    def calc_beta(omega: float, medium: str, n: Optional[float]) -> float: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def calc_beta(omega: Array[float], medium: str, n: Optional[Array[float]],
                  ) -> Array[float]: ...
    # ------------------------------------------------------------------
    @staticmethod
    def calc_beta(omega, medium, n=None):
        r"""Calcul the propagation constant. (valid only for TEM)

        Parameters
        ----------
        omega :
            The angular frequency. :math:`[rad\cdot ps^{-1}]`
        medium :
            The medium in which the wave propagates (used for Sellmeier
            equations if n not provided).
        n :
            The refractive index at angular frequency :math:`\omega`.
            Calcul :math:`n(\omega)` from Sellmeier equation if None.

        Returns
        -------
        :
            The propagation constant.
            :math:`[km^{-1}]`

        Notes
        -----

        .. math:: \beta(\omega) = k(\omega) = k_0 n(\omega)
                  = \frac{\omega}{c} n(\omega)

        """
        LIGHT_SPEED = cst.LIGHT_SPEED * 1e-12   # nm/ps -> km/ps
        if (isinstance(omega, float)):
            res = 0.0
        else:
            res = np.zeros_like(omega)
        if (n is None):
            sellmeier = Sellmeier(medium)
            res = omega / LIGHT_SPEED * sellmeier.n(omega)
        else:
            res = omega / LIGHT_SPEED * n

        return res
    # ==================================================================
    @overload
    @staticmethod
    def calc_beta_deriv(omega: float, order: int, medium: str) -> float: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def calc_beta_deriv(omega: Array[float], order: int, medium: str
                        ) -> Array[float]: ...
    # ------------------------------------------------------------------
    @staticmethod
    def calc_beta_deriv(omega, order, medium):
        r"""Calcul the nth derivative of the propagation constant.
        (valid only for TEM)

        Parameters
        ----------
        omega :
            The angular frequency. :math:`[rad\cdot ps^{-1}]`
        order :
            The order of the derivative.
        medium :
            The medium in which the wave propagates, used for Sellmeier
            equations in order to calculate the refractive index.

        Returns
        -------
        :
            The nth derivative of the propagation constant.
            :math:`[ps^{i}\cdot km^{-1}]`

        Notes
        -----

        .. math:: \beta_i(\omega) = \frac{d^i \beta(\omega)}{d\omega^i}
                  = \frac{1}{c}\bigg(i
                  \frac{d^{(i-1)} n(\omega)}{d\omega^{(i-1)}}
                  + \omega \frac{d^i n(\omega)}{d\omega^i}\bigg)

        for :math:`i = 0, \ldots, \text{order}`

        """
        LIGHT_SPEED = cst.LIGHT_SPEED * 1e-12   # nm/ps -> km/ps
        sellmeier_eq = Sellmeier(medium)
        if (order):
            res = (order*sellmeier_eq.n_deriv(omega, order-1)
                   + omega*sellmeier_eq.n_deriv(omega, order)) / LIGHT_SPEED
        else:
            res = omega / LIGHT_SPEED * sellmeier_eq.n(omega)

        return res
    # ==================================================================
    @overload
    @staticmethod
    def calc_dispersion(Lambda: float, beta_2: float) -> float: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def calc_dispersion(Lambda: Array[float], beta_2: Array[float]
                        ) -> Array[float]: ...
    # ------------------------------------------------------------------
    @staticmethod
    def calc_dispersion(Lambda, beta_2):
        r"""Calculate the dispersion parameter.

        Parameters
        ----------
        Lambda :
            The wavelength. :math:`[nm]`
        beta_2 :
            The GVD term of the dispersion. :math:`[ps^2\cdot km^{-1}]`

        Returns
        -------
        :
            Value of the dispersion parameter.
            :math:`[ps \cdot nm^{-1} \cdot km^{-1}]`

        Notes
        -----

        .. math::  D = \frac{d}{d\lambda}\Big(\frac{1}{v_g}\Big)
                   = \frac{d}{d\lambda} \beta_1
                   = -\frac{2\pi c}{\lambda^2} \beta_2

        """
        if (isinstance(Lambda, float)):
            factor = (2.0 * cst.PI * cst.LIGHT_SPEED) / (Lambda**2)
        else:
            factor = (2.0 * cst.PI * cst.LIGHT_SPEED) / np.square(Lambda)

        return -1 * factor * beta_2
    # ==================================================================
    @overload
    @staticmethod
    def calc_dispersion_slope(Lambda: float, beta_2: float, beta_3: float
                              ) -> float: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def calc_dispersion_slope(Lambda: Array[float], beta_2: Array[float],
                              beta_3: Array[float]) -> Array[float]: ...
    # ------------------------------------------------------------------
    @staticmethod
    def calc_dispersion_slope(Lambda, beta_2, beta_3):
        r"""Calculate the dispersio slope.

        Parameters
        ----------
        Lambda :
            The wavelength. :math:`[nm]`
        beta_2 :
            The GVD term of the dispersion. :math:`[ps^2\cdot km^{-1}]`
        beta_3 :
            The third order dispersion term. :math:`[ps^3\cdot km^{-1}]`

        Returns
        -------
        :
            Value of the dispersion parameter.
            :math:`[ps \cdot nm^{-2} \cdot km^{-1}]`

        Notes
        -----

        .. math::  S = \frac{d D}{d\lambda}
                   = \beta_2 \frac{d}{d\lambda} \Big(-\frac{2\pi c}
                   {\lambda^2}\Big) - \frac{2\pi c}
                   {\lambda^2} \frac{d\beta_2}{d\lambda}
                   = \frac{4\pi c}{\lambda^3} \beta_2
                   + \Big(\frac{2\pi c}{\lambda^2}\Big)^2 \beta_3

        """
        if (isinstance(Lambda, float)):
            factor = (2.0 * cst.PI * cst.LIGHT_SPEED) / (Lambda**2)
        else:
            factor = (2.0 * cst.PI * cst.LIGHT_SPEED) / np.square(Lambda)

        return (2 * factor / Lambda * beta_2) + (factor * factor * beta_3)
    # ==================================================================
    @overload
    @staticmethod
    def calc_RDS(Lambda: float, beta_2: float, beta_3: float) -> float: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def calc_RDS(Lambda: Array[float], beta_2: Array[float],
                 beta_3: Array[float]) -> Array[float]: ...
    # ------------------------------------------------------------------
    @staticmethod
    def calc_RDS(Lambda, beta_2, beta_3):
        r"""Calculate the relative dispersion slope.

        Parameters
        ----------
        Lambda :
            The wavelength. :math:`[nm]`
        beta_2 :
            The GVD term of the dispersion. :math:`[ps^2\cdot km^{-1}]`
        beta_3 :
            The third order dispersion term. :math:`[ps^3\cdot km^{-1}]`

        Returns
        -------
        :
            Value of the dispersion parameter.
            :math:`[nm^{-1}]`

        Notes
        -----

        .. math::  RDS = \frac{S}{D}

        """

        return (Dispersion.calc_dispersion_slope(beta_2, beta_3, Lambda)
                / Dispersion.calc_dispersion(beta_2, Lambda))
    # ==================================================================
    @overload
    @staticmethod
    def calc_accumulated_dispersion(Lambda: float, beta_2: float, length: float
                                    ) -> float: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def calc_accumulated_dispersion(Lambda: Array[float], beta_2: Array[float],
                                    length: float) -> Array[float]: ...
    # ------------------------------------------------------------------
    @staticmethod
    def calc_accumulated_dispersion(Lambda, beta_2, length):
        r"""Calculate the dispersion parameter.

        Parameters
        ----------
        Lambda :
            The wavelength. :math:`[nm]`
        beta_2 :
            The GVD term of the dispersion. :math:`[ps^2\cdot km^{-1}]`
        length :
            The length over which dispersion is considered. :math:`[km]`

        Returns
        -------
        :
            Value of the dispersion parameter.
            :math:`[ps \cdot nm^{-1}]`

        Notes
        -----

        .. math::  D_{acc} = D \cdot L

        """

        return Dispersion.calc_dispersion(beta_2, Lambda) * length


if __name__ == "__main__":

    import optcom.utils.plot as plot
    from optcom.domain import Domain
    from optcom.equations.sellmeier import Sellmeier

    center_omega = 1929.97086814#Domain.lambda_to_omega(976.0)
    sellmeier = Sellmeier("Sio2")
    betas = Dispersion.calc_beta_coeffs(center_omega, 13, sellmeier)
    print('betas: ', betas)

    Lambda = np.linspace(900, 1600, 1000)
    omega = Domain.lambda_to_omega(Lambda)
    beta_2 = Dispersion.calc_beta_deriv(omega, 2, "SiO2")
    x_data = [Lambda]
    y_data = [beta_2]

    x_labels = ['Lambda']
    y_labels = ['beta2']
    plot_titles = ["Group velocity dispersion coefficients in Silica"]

    disp = Dispersion.calc_dispersion(Lambda, beta_2)
    x_data.append(Lambda)
    y_data.append(disp)
    x_labels.append('Lambda')
    y_labels.append('dispersion')
    plot_titles.append('Dispersion of Silica')

    beta_3 = Dispersion.calc_beta_deriv(omega, 3, "Sio2")
    slope = Dispersion.calc_dispersion_slope(Lambda, beta_2, beta_3)
    x_data.append(Lambda)
    y_data.append(slope)
    x_labels.append('Lambda')
    y_labels.append('dispersion_slope')
    plot_titles.append('Dispersion slope of Silica')

    plot.plot(x_data, y_data, x_labels=x_labels, y_labels=y_labels,
              plot_titles=plot_titles, split=True, opacity=0.0)
