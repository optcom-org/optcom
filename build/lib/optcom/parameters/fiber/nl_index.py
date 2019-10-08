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
from typing import List, Optional, overload, Union

import numpy as np
from nptyping import Array

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.domain import Domain
from optcom.parameters.abstract_parameter import AbstractParameter


# Ref: Salceda-Delgado, G., Martinez-Rios, A., Ilan, B. and
# Monzon-Hernandez, D., 2012.  Raman response function an Raman fraction
# of phosphosilicate fibers.  Optical and Quantum Electronics, 44(14),
# pp.657-671.
# poly_coeff = {medium: [(coeff, exponant)], ...}
POLY_FACTOR = {"sio2": 1.000055}
POLY_COEFF = {"sio2": [(8.30608, 0.0), (-27.79971, 1.0), (59.66014, 2.0),
              (-69.24258, 3.0), (45.22437, 4.0), (-15.63666, 5.0),
              (2.22585, 6.0)]}


class NLIndex(AbstractParameter):

    def __init__(self, medium: str = cst.DEF_FIBER_MEDIUM) -> None:
        """The non linear index.

        Parameters
        ----------
        core_radius :
            The radius of the core. :math:`[\mu m]`

        """
        self._medium = medium.lower()
    # ==================================================================
    def is_medium_recognised(self, medium: str) -> bool:

        return (POLY_COEFF.get(medium.lower()) is not None)
    # ==================================================================
    @overload
    def __call__(self, omega: float) -> float: ...
    # ------------------------------------------------------------------
    @overload
    def __call__(self, omega: Array[float]) -> float: ...
    # ------------------------------------------------------------------
    def __call__(self, omega):
        """Compute the non linear index.

        Parameters
        ----------
        omega :
            The angular frequency.  :math:`[rad\cdot ps^{-1}]`

        Returns
        -------
        :
            Value of the non linear index. :math:`[m^2\cdot W^{-1}]`

        """

        return NLIndex.calc_nl_index(omega, self._medium)

    # ==================================================================
    @overload
    @staticmethod
    def calc_nl_index(omega: float, medium: str) -> float: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def calc_nl_index(omega: Array[float], medium: str) -> Array[float]: ...
    # ------------------------------------------------------------------
    @staticmethod
    def calc_nl_index(omega, medium):
        r"""Calculate the non linear index by help of fitting formula.
        The non linear is assumed to be the sum of the electronic and
        Raman contributions at zero frequency shift. [5]_

        Parameters
        ----------
        omega :
            The angular frequency.  :math:`[rad\cdot ps^{-1}]`
        medium :
            The medium in which to consider the non linear index.

        Returns
        -------
        :
            Value of the non linear index. :math:`[m^2\cdot W^{-1}]`

        Notes
        -----
        Considering:

        .. math:: n_2(\lambda) = 1.000055 (8.30608 - 27.79971\lambda
                  + 59.66014\lambda^2 - 69.24258\lambda^3
                  + 45.22437\lambda^4 - 15.63666\lambda^5
                  + 2.22585\lambda^6)

        with :math:`\lambda` in :math:`nm` and return with factor
        :math:`10^{-20}`

        References
        ----------
        .. [5] Salceda-Delgado, G., Martinez-Rios, A., Ilan, B. and
               Monzon-Hernandez, D., 2012. Raman response function an
               Raman fraction of phosphosilicate fibers. Optical and
               Quantum Electronics, 44(14), pp.657-671.

        """
        medium = medium.lower()
        if (isinstance(omega, float)):
            res = 0.0
        else:
            res = np.zeros_like(omega)
        Lambda = Domain.omega_to_lambda(omega)
        Lambda *= 1e-3  # nm -> um
        coeff = POLY_COEFF.get(medium)
        if (coeff is not None):
            if (isinstance(omega, float)):
                for elem in coeff:
                    res += elem[0] * Lambda**elem[1]
            else:
                for elem in coeff:
                    res += elem[0] * np.power(Lambda, elem[1])
            factor = POLY_FACTOR.get(medium)
            if (factor is not None):
                res *= factor
        else:
            util.warning_terminal("The medium provided to calculate the "
                "non linear index is not recognised. Return null values.")

        return res * 1e-20  # from fitting formula to m^2 w^-1


if __name__ == "__main__":

    import numpy as np

    import optcom.utils.plot as plot
    from optcom.domain import Domain

    # With float
    omega = Domain.lambda_to_omega(1552.0)

    nl_index = NLIndex(medium="SiO2")
    print(nl_index(omega))
    print(nl_index.calc_nl_index(omega, medium="SiO2"))

    # With numpy ndarray
    lambdas = np.linspace(500, 1600, 1000)
    omegas = Domain.lambda_to_omega(lambdas)

    nl_index = NLIndex(medium="SiO2")
    res = nl_index(omegas)

    x_labels = ['Lambda']
    y_labels = ['n_2']
    plot_titles = ["Non linear index of Silica from fitting formula"]

    plot.plot(lambdas, res, x_labels=x_labels, y_labels=y_labels,
              plot_titles=plot_titles, opacity=0.0,
              y_ranges=[2.5*1e-20, 3.5*1e-20])
