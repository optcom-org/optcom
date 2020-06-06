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
from typing import Callable, List, Optional, overload, Union

import numpy as np

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.parameters.abstract_parameter import AbstractParameter
from optcom.utils.callable_container import CallableContainer


class VNumber(AbstractParameter):

    def __init__(self, NA: Union[float, Callable], core_radius: float
                 ) -> None:
        """
        Parameters
        ----------
        NA :
            The numerical_aperture. If a callable is provided, the
            variable must be angular frequency. :math:`[ps^{-1}]`
        core_radius :
            The radius of the core. :math:`[\mu m]`

        """
        self._core_radius: float = core_radius
        self._NA: Union[float, Callable] = NA
    # ==================================================================
    @property
    def NA(self) -> Union[float, Callable]:

        return self._NA
    # ------------------------------------------------------------------
    @NA.setter
    def NA(self, NA: Union[float, Callable]) -> None:

        self._NA = NA
    # ==================================================================
    @overload
    def __call__(self, omega: float) -> float: ...
    # ------------------------------------------------------------------
    @overload
    def __call__(self, omega: np.ndarray) -> np.ndarray: ...
    # ------------------------------------------------------------------
    def __call__(self, omega):
        """Compute the V number.

        Parameters
        ----------
        omega :
            The angular frequency. :math:`[rad\cdot ps^{-1}]`

        Returns
        -------
        :
            Value of the V parameter.
        """
        fct = CallableContainer(VNumber.calc_v_number,
                                [omega, self._NA, self._core_radius])

        return fct(omega)
    # ==================================================================
    @overload
    @staticmethod
    def calc_v_number(omega: float, NA: float, core_radius: float
                      ) -> float: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def calc_v_number(omega: np.ndarray, NA: np.ndarray, core_radius: float
                      ) -> np.ndarray: ...
    # ------------------------------------------------------------------
    @staticmethod
    def calc_v_number(omega, NA, core_radius):
        r"""Calculate the V parameter.

        Parameters
        ----------
        omega :
            The angular frequency.  :math:`[rad\cdot ps^{-1}]`
        NA :
            The numerical aperture.
        core_radius :
            The radius of the core. :math:`[\mu m]`

        Returns
        -------
        :
            Value of the V parameter.

        Notes
        -----
        Considering:

        .. math:: V = k_0 a \text{NA} = \frac{\omega_0}{c} a \text{NA}

        """
        # Unit conversion
        core_radius *= 1e3  # um -> nm

        return NA * core_radius * omega / cst.LIGHT_SPEED


if __name__ == "__main__":
    """Plot the V fiber number as a function of the wavelength.
    This piece of code is standalone, i.e. can be used in a separate
    file as an example.
    """

    import math
    from typing import List

    import numpy as np

    import optcom.utils.plot as plot
    from optcom.domain import Domain
    from optcom.parameters.fiber.numerical_aperture import NumericalAperture
    from optcom.parameters.fiber.v_number import VNumber
    from optcom.parameters.refractive_index.sellmeier import Sellmeier

    # With float
    omega: float = Domain.lambda_to_omega(1552.0)
    core_radius: float = 5.0
    n_clad: float = 1.44
    sellmeier: Sellmeier = Sellmeier("sio2")
    NA_inst: NumericalAperture = NumericalAperture(sellmeier, n_clad)
    v_nbr: VNumber = VNumber(NA_inst, core_radius)
    print(v_nbr(omega))
    NA: float = NA_inst(omega)
    v_nbr = VNumber(NA, core_radius)
    print(v_nbr(omega))
    print(VNumber.calc_v_number(omega, NA, core_radius))
    # With np.ndarray
    lambdas: np.ndarray = np.linspace(900, 1550, 10)
    omegas: np.ndarray = Domain.lambda_to_omega(lambdas)
    NA_inst = NumericalAperture(sellmeier, n_clad)
    v_nbr = VNumber(NA_inst, core_radius)
    res: np.ndarray = v_nbr(omegas)
    x_labels: List[str] = ['Lambda']
    y_labels: List[str] = ['V Number']
    plot_titles: List[str] = ["V number as a function of the wavelength \n"
                              "for Silica core with constant cladding "
                              "refractive index."]

    plot.plot2d([lambdas], [res], x_labels=x_labels, y_labels=y_labels,
                plot_titles=plot_titles, opacity=[0.0], y_ranges=[(2., 7.)])