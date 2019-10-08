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
from optcom.effects.absorption import Absorption
from optcom.effects.abstract_effect import AbstractEffect
from optcom.equations.mccumber import McCumber


dopants = ["yb", "er"]


class StimulatedEmission(AbstractEffect):
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
                    "Emission effect is not supported, default dopant {} set"
                    .format(cst.DEF_FIBER_DOPANT))
            else:
                self._dopant = dopant
    # ==================================================================
    @overload
    def get_cross_section(self, omega: float, center_omega: Optional[float],
                          N_0: Optional[float], N_1: Optional[float],
                          T: Optional[float]) -> float: ...
    # ------------------------------------------------------------------
    @overload
    def get_cross_section(self, omega: Array[float],
                          center_omega: Optional[float], N_0: Optional[float],
                          N_1: Optional[float], T: Optional[float]
                          ) -> Array[float]: ...
    # ------------------------------------------------------------------
    def get_cross_section(self, omega, center_omega=None, N_0=None, N_1=None,
                          T=None):

        return StimulatedEmission.calc_cross_section(omega, self._dopant,
                                                     self._predict,
                                                     center_omega, N_0, N_1, T)
    # ==================================================================
    # N.B.: No analytical fct for emission cross sections.  Can use
    # however the McCumber relations to draw emission from absorption.
    @overload
    @staticmethod
    def calc_cross_section(omega: float, dopant: str, predict: Callable,
                           center_omega: Optional[float], N_0: Optional[float],
                           N_1: Optional[float], T: Optional[float]
                           ) -> float: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def calc_cross_section(omega: Array[float], dopant: str, predict: Callable,
                           center_omega: Optional[float], N_0: Optional[float],
                           N_1: Optional[float], T: Optional[float]
                           ) -> Array[float]: ...
    # ------------------------------------------------------------------
    @staticmethod
    def calc_cross_section(omega, dopant ="Yb", predict=None,
                           center_omega=None, N_0=None, N_1=None, T=None):
        r"""Calculate the emission cross section. There is no analytical
        formula for the emission cross section. Calculate first the
        absorption cross sections with the fitting formula from ref.
        [5]_ and then use the McCumber relations to deduce the emission
        cross sections.

        Parameters
        ----------
        omega :
            The angular frequency.  :math:`[rad\cdot ps^{-1}]`
        dopant :
            The type of the doped medium.
        predict :
            A callable object to predict the cross sections.
            The variable must be the wavelength. :math:`[nm]`
        center_omega :
            The center angular frequency.  :math:`[rad\cdot ps^{-1}]`
        N_0 :
            The population in ground state. :math:`[nm^{-3}]`
        N_1 :
            The population in the excited state. :math:`[nm^{-3}]`
        T :
            The temperature. :math:`[K]`

        Returns
        -------
        :
            Value of the emission cross section. :math:`[nm^2]`

        References
        ----------
        .. [5] Valley, G.C., 2001. Modeling cladding-pumped Er/Yb fiber
               amplifiers. Optical Fiber Technology, 7(1), pp.21-44.

        """

        if (isinstance(omega, float)):
            res = 0.0
        else:
            res = np.zeros_like(omega)
        Lambda = Domain.omega_to_lambda(omega)
        if (predict is None):
            if (N_0 is not None and N_1 is not None and T is not None
                    and center_omega is not None):
                sigma_a = Absorption.calc_cross_section(omega, dopant)
                res = McCumber.calc_cross_section_emission(sigma_a, omega,
                                                           center_omega, N_0,
                                                           N_1, T)
            else:
                util.warning_terminal("Not enough information to calculate "
                    "the value of the emission cross sections, will return "
                    "null values.")

        else:   # Interpolate the data from csv file with scipy fct
            res = predict(Lambda)[0]    # Take only zeroth deriv.

        return res


if __name__ == "__main__":

    import optcom.utils.plot as plot
    from optcom.domain import Domain
    from optcom.utils.utilities_user import CSVFit
    omegas: Array[float]
    dopants = ["yb", "er"]
    # dopant_range = {dopant:(bottom, up), \ldots} # in nm
    dopant_range = {"yb": (900, 1050), "er": (1400, 1650)}

    folder = './data/fiber_amp/cross_section/emission/'
    files = ['yb']
    file_range = [(900.0,1050.0)]

    x_data = []
    y_data = []
    plot_titles = []

    for dopant in dopants:
        nbr_samples = 1000
        lambdas = np.linspace(dopant_range[dopant][0], dopant_range[dopant][1],
                              nbr_samples)
        omegas = Domain.lambda_to_omega(lambdas)
        center_lambda = (dopant_range[dopant][0] + dopant_range[dopant][1]) / 2
        center_omega = Domain.lambda_to_omega(center_lambda)
        N_0 = 0.01   # nm^-3
        N_1 = 0.01 * 1.5   # nm^-3
        T = 293.5
        stimu = StimulatedEmission(dopant=dopant)
        sigmas = stimu.get_cross_section(omegas, center_omega, N_0, N_1, T)
        x_data.append(lambdas)
        y_data.append(sigmas)
        dopant_name = dopant[0].upper()+dopant[1:]
        plot_titles.append("Cross sections {} from formula and McCumber "
                           "relations".format(dopant_name))

    for i, file in enumerate(files):
        nbr_samples = 1000
        lambdas = np.linspace(file_range[i][0], file_range[i][1],
                              nbr_samples)
        omegas = Domain.lambda_to_omega(lambdas)
        file_name = folder + file + '.txt'
        predict = CSVFit(file_name=file_name, conv_factor=[1e9, 1e18]) # in nm
        stimu = StimulatedEmission(predict=predict)
        sigmas = stimu.get_cross_section(omegas)
        x_data.append(lambdas)
        y_data.append(sigmas)
        dopant_name = file[0].upper()+file[1:]
        plot_titles.append("Emission ross sections {} from data file"
                           .format(dopant_name))

    plot.plot(x_data, y_data, x_labels=['Lambda'], y_labels=['sigma_e'],
              split=True, plot_colors='red', plot_titles=plot_titles,
              plot_linestyles='-.', opacity=0.0)
