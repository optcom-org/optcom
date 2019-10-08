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
from scipy import interpolate

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.domain import Domain
from optcom.effects.abstract_effect import AbstractEffect



dopants = ["yb", "er"]
# dopant_range = {dopant:(bottom, up), \ldots} # in nm
dopant_range = {"yb": (900, 1050), "er": (1400, 1650)}
# ref: Valley, G.C., 2001. Modeling cladding-pumped Er/Yb fiber
# amplifiers.  Optical Fiber Technology, 7(1), pp.21-44.
# dopant_fit_coeff = { dopant:[[coef, l_c, l_w, n],\ldots], \ldots}
dopant_fit_coeff = {"yb": [[0.09,913.0,8.0,2.0], [0.13,950.0,40.0,4.0],
                           [0.2,968.0,40.0,2.4], [1.08,975.8,3.0,1.5]],
                    "er": [[0.221,1493.0,16.5,2.2], [0.342,1534.0,4.5,1.4],
                           [0.158,1534.0,30.0,4.0], [0.037,1534.0,85.0,4.0],
                           [0.132,1541.0,8.0,0.8]]}


class Absorption(AbstractEffect):
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

    def __init__(self, dopant: str = cst.DEF_FIBER_DOPANT,
                 predict: Optional[Callable] = None) -> None:
        """
        Parameters
        ----------
        dopant :
            The doping agent.
        predict :
            A callable object to predict the cross sections.
            The variable must be the wavelength. :math:`[nm]`

        """
        self._dopant: str = ''
        self._predict: Optional[Callable] = predict
        if (predict is None):
            if (dopant not in dopants):
                self._dopant = cst.DEF_FIBER_DOPANT
                util.warning_terminal("The dopant specified for the "
                    "Absorption effect is not supported, default dopant {} set"
                    .format(cst.DEF_FIBER_DOPANT))
            else:
                self._dopant = dopant
    # ==================================================================
    @overload
    def get_cross_section(self, omega: float) -> float: ...
    # ------------------------------------------------------------------
    @overload
    def get_cross_section(self, omega: Array[float]) -> Array[float]: ...
    # ------------------------------------------------------------------
    def get_cross_section(self, omega):

        return Absorption.calc_cross_section(omega, self._dopant,
                                             self._predict)
    # ==================================================================
    # N.B.: No analytical fct for emission cross sections.  Can use
    # however the McCumber relations to draw emission from absorption.
    @overload
    @staticmethod
    def calc_cross_section(omega: float, dopant: str, predict: Callable
                           ) -> float: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def calc_cross_section(omega: Array[float], dopant: str,
                           predict: Callable) -> Array[float]: ...
    # ------------------------------------------------------------------
    @staticmethod
    def calc_cross_section(omega, dopant ="Yb", predict=None):
        r"""Calculate the absorption cross section. Calculate with the
        fitting formula from ref. [4]_ .

        Parameters
        ----------
        omega :
            The angular frequency.  :math:`[rad\cdot ps^{-1}]`
        dopant :
            The doping agent.
        predict :
            A callable object to predict the cross sections.
            The variable must be the wavelength. :math:`[nm]`

        Returns
        -------
        float or numpy.ndarray of float
            Value of the absorption cross section. :math:`[nm^2]`

        Notes
        -----
        Considering:

        .. math:: f(\lambda, \lambda_c, \lambda_w, n)
                  = \exp\Big[-\Big\lvert
                  \frac{\lambda-\lambda_c}{\lambda_w}\Big\rvert^n \Big]

        With :math:`\lambda`, :math:`\lambda_c` and :math:`\lambda_w`
        in :math:`[nm]`

        For Ytterbium:

        .. math:: \sigma_a^{Yb}(\lambda) = 0.09f(\lambda,913,8,2)
                  + 0.13 f(\lambda,950,40,4)
                  + 0.2 f(\lambda, 968, 40, 2.4)
                  + 1.08 f(\lambda, 975.8,3,1.5)

        For Erbium:

        .. math:: \begin{split}
                    \sigma_a^{Er}(\lambda) &= 0.221
                    f(\lambda, 1493, 16.5, 2.2)
                    + 0.342 f(\lambda, 1534, 4.5, 1.4)\\
                    &\quad + 0.158f(\lambda, 1534, 30,4)
                    + 0.037 f(\lambda, 1534, 85, 4)
                    + 0.132 f(\lambda, 1541, 8,0.8)
                  \end{split}

        References
        ----------
        .. [4] Valley, G.C., 2001. Modeling cladding-pumped Er/Yb fiber
               amplifiers. Optical Fiber Technology, 7(1), pp.21-44.

        """
        Lambda = Domain.omega_to_lambda(omega)
        if (predict is None):
            dopant = dopant.lower()
            if (dopant in dopants):
                if (isinstance(Lambda, float)):
                    res = 0.0
                    if (Lambda < dopant_range[dopant][1]
                            and Lambda > dopant_range[dopant][0]):
                        res = Absorption._formula(Lambda,
                                                  dopant_fit_coeff[dopant])
                else:
                    Lambda_down = Lambda[Lambda < dopant_range[dopant][0]]
                    Lambda_up = Lambda[Lambda > dopant_range[dopant][1]]
                    if (len(Lambda_up)):
                        Lambda_in = Lambda[len(Lambda_down):-len(Lambda_up)]
                    else:
                        Lambda_in = Lambda[len(Lambda_down):]
                    res = np.array([])
                    if (Lambda_in.size):
                        res = Absorption._formula(Lambda_in,
                                                  dopant_fit_coeff[dopant])
                    res = np.hstack((np.zeros_like(Lambda_down),
                                     res, np.zeros_like(Lambda_up)))
                res *= 1e-6      # 10^-20 cm^2 -> nm^2
            else:
                util.warning_terminal("The specified doped material is not "
                    "supported, return 0")
        else:   # Interpolate the data in csv file with scipy fct
            res = predict(Lambda)[0]    # Take only zeroth deriv.

        return res
    # ==================================================================
    @overload
    @staticmethod
    def _formula(Lambda: float, coefficients: List[float]) -> float: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def _formula(Lambda: Array[float], coefficients: List[float]
                ) -> Array[float]: ...
    # ------------------------------------------------------------------
    @staticmethod
    def _formula(Lambda, coefficients):

        def f(l, lc, lw, n):
            if (isinstance(l, float)):

                return math.exp(-(abs((l-lc)/lw)**n))
            else:

                return np.exp(-(np.abs((l-lc)/lw)**n))

        if (isinstance(Lambda, float)):
            res = 0.0
        else:
            res = np.zeros(Lambda.shape)

        for coeff in coefficients:
            res += coeff[0] * f(Lambda, coeff[1], coeff[2], coeff[3])

        return res



if __name__ == "__main__":

    import optcom.utils.plot as plot
    from optcom.domain import Domain
    from optcom.utils.utilities_user import CSVFit

    folder = './data/fiber_amp/cross_section/absorption/'
    files = ['yb']
    file_range = [(900.0,1050.0)]

    x_data = []
    y_data = []
    plot_titles = []

    for dopant in dopants:
        nbr_samples = 1000
        lambdas = np.linspace(dopant_range[dopant][0], dopant_range[dopant][1],
                              nbr_samples)
        omegas = Domain.omega_to_lambda(lambdas)
        absorp = Absorption(dopant=dopant)
        sigmas = absorp.get_cross_section(omegas)
        x_data.append(lambdas)
        y_data.append(sigmas)
        dopant_name = dopant[0].upper()+dopant[1:]
        plot_titles.append('Cross sections {} from formula'
                           .format(dopant_name))

    for i, file in enumerate(files):
        nbr_samples = 1000
        lambdas = np.linspace(file_range[i][0], file_range[i][1],
                              nbr_samples)
        omegas = Domain.omega_to_lambda(lambdas)
        file_name = folder + file + '.txt'
        predict = CSVFit(file_name=file_name, conv_factor=[1e9, 1e18]) # in nm
        absorp = Absorption(predict=predict)
        sigmas = absorp.get_cross_section(omegas)
        x_data.append(lambdas)
        y_data.append(sigmas)
        dopant_name = file[0].upper()+file[1:]
        plot_titles.append('Absorption cross sections {} from data file'
                           .format(dopant_name))

    plot.plot(x_data, y_data, x_labels=['Lambda'], y_labels=['sigma_a'],
              split=True, plot_colors='red', plot_titles=plot_titles,
              plot_linestyles='-.', opacity=0.0)
