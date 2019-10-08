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
from nptyping import Array

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.parameters.abstract_parameter import AbstractParameter
from optcom.parameters.fiber.v_para import V


# Ref: Govind Agrawal, Chapter 2: Pulse Propagation in Fibers,
# Nonlinear Fiber Optics (Fifth Edition), Academic Press, 2013, Page 34.
# coefficients = [(coeff, exponant)]
w_coefficients =[(0.616, 0.0), (1.66, -1.5), (0.987, -6.0)]


class EffectiveArea(AbstractParameter):

    def __init__(self, core_radius: float = cst.CORE_RADIUS,
                 NA: Union[float, Array[float], Callable] = cst.NA) -> None:
        """Effective area, currently only for single mode.

        Parameters
        ----------
        core_radius :
            The radius of the core. :math:`[\mu m]`

        """
        self._core_radius: float = core_radius
        self._NA: Union[float, Array[float], Callable] = NA
    # ==================================================================
    @overload
    def __call__(self, omega: float) -> float: ...
    # ------------------------------------------------------------------
    @overload
    def __call__(self, omega: Array[float]) -> Array[float]: ...
    # ------------------------------------------------------------------
    def __call__(self, omega):
        """Compute the effective area.

        Parameters
        ----------
        omega :
            The angular frequency.  :math:`[rad\cdot ps^{-1}]`

        Returns
        -------
        :
            Value of the effective area. :math:`[\mu m^2]`

        """
        NA = self._NA(omega) if callable(self._NA) else self._NA

        return EffectiveArea.calc_effective_area(omega, self._core_radius, NA)

    # ==================================================================
    @overload
    @staticmethod
    def calc_effective_area(omega: float, core_radius: float,
                            NA: Optional[float], n_core: Optional[float],
                            n_clad: Optional[float]
                            ) -> float: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def calc_effective_area(omega: Array[float], core_radius: float,
                            NA: Optional[Array[float]],
                            n_core: Optional[Array[float]],
                            n_clad: Optional[Array[float]]
                            ) -> Array[float]: ...
    # ------------------------------------------------------------------
    @staticmethod
    def calc_effective_area(omega, core_radius=cst.CORE_RADIUS, NA=None,
                            n_core=None, n_clad=None):
        r"""Calculate the effective area (for now only single mode).
        [4]_

        Parameters
        ----------
        omega :
            The angular frequency.  :math:`[rad\cdot ps^{-1}]`
        core_radius :
            The radius of the core. :math:`[\mu m]`
        NA :
            The numerical aperture.
        n_core :
            The refractive index of the core.
        n_clad :
            The refractive index of the cladding.

        Returns
        -------
        :
            Value of the effective area. :math:`[\mu m^2]`

        Notes
        -----
        Considering:

        .. math:: A_{eff} = \pi w^2 = \pi \big(a(0.616
                  + 1.66V^{-\frac{2}{3}} + 0.987V^{-6})\big)^2

        References
        ----------
        .. [4] Govind Agrawal, Chapter 2: Pulse Propagation in Fibers,
           Nonlinear Fiber Optics (Fifth Edition), Academic Press, 2013,
           Page 34.

        """
        v_para = V.calc_V(omega, core_radius, NA, n_core, n_clad)

        if (isinstance(omega, float)):
            res = 0.0
            for coeff in w_coefficients:
                res += coeff[0] * v_para**(coeff[1])
            res = cst.PI * (core_radius*res)**2

        else:
            res = np.zeros(omega.shape)
            for coeff in w_coefficients:
                res += coeff[0] * np.power(v_para, coeff[1])
            res = cst.PI * np.square(core_radius*res)

        return res


if __name__ == "__main__":

    import math
    import numpy as np

    import optcom.utils.plot as plot
    from optcom.domain import Domain
    from optcom.parameters.fiber.numerical_aperture import NumericalAperture

    # With float
    omega = Domain.lambda_to_omega(1552.0)
    core_radius = 5.0
    n_core = 1.43
    n_clad = 1.425
    NA = math.sqrt(n_core**2 - n_clad**2)

    effective_area = EffectiveArea(core_radius=core_radius, NA=NA)
    print(effective_area(omega))
    print(EffectiveArea.calc_effective_area(omega, core_radius=core_radius,
                                            n_core=n_core, n_clad=n_clad))

    effective_area = EffectiveArea(core_radius=core_radius, NA=NA)
    print(effective_area(omega))
    print(EffectiveArea.calc_effective_area(omega, core_radius=core_radius,
                                            NA=NA))

    # With numpy ndarray
    lambdas = np.linspace(900, 1550, 100)
    omegas = Domain.lambda_to_omega(lambdas)
    n_core = np.linspace(1.42, 1.43, 100)
    n_clad = np.linspace(1.415, 1.425, 100)
    NA = NumericalAperture.calc_NA(n_core, n_clad)

    effective_area = EffectiveArea(core_radius=core_radius, NA=NA)
    res = effective_area(omegas)

    x_labels = ['Lambda']
    y_labels = ['Effective Area']
    plot_titles = ["Effective Area as a function of the wavelength"]

    plot.plot(lambdas, res, x_labels=x_labels, y_labels=y_labels,
              plot_titles=plot_titles, opacity=0.0,
              y_ranges=[2.5*1e-20, 3.5*1e-20])
