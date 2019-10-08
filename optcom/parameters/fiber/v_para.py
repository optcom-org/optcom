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
from typing import List, Optional, overload

import numpy as np
from nptyping import Array

import optcom.utils.constants as cst
import optcom.utils.utilities as util
from optcom.parameters.abstract_parameter import AbstractParameter



class V(AbstractParameter):

    def __init__(self, core_radius: float = cst.CORE_RADIUS,
                 NA: Optional[float] = None, n_core: Optional[float] = None,
                 n_clad: Optional[float] = None):
        """
        Parameters
        ----------
        omega :
            The angular frequency.  :math:`[rad\cdot ps^{-1}]`
        core_radius :
            The radius of the core. :math:`[\mu m]`
        NA :
            The numerical aperture.
        n_core : float or numpy.ndarray of float
            The refractive index of the core.
        n_clad :
            The refractive index of the cladding.

        """
        self._core_radius: float = core_radius
        self._NA: Optional[float] = NA
        self._n_core: Optional[float] = n_core
        self._n_clad: Optional[float] = n_clad
    # ==================================================================
    @overload
    def __call__(self, omega: float) -> float: ...
    # ------------------------------------------------------------------
    @overload
    def __call__(self, omega: Array[float]) -> Array[float]: ...
    # ------------------------------------------------------------------
    def __call__(self, omega: float) -> float:
        r"""Calculate the V parameter.

        Parameters
        ----------
        omega :
            The angular frequency.  :math:`[rad\cdot ps^{-1}]`

        Returns
        -------
        :
            Value of the V parameter.
        """
        return V.calc_V(omega, self._core_radius, self._NA, self._n_core,
                        self._n_clad)
    # ==================================================================
    @overload
    @staticmethod
    def calc_V(omega: float, core_radius: float, NA: Optional[float],
               n_core: Optional[float], n_clad: Optional[float]
               ) -> float: ...
    # ------------------------------------------------------------------
    @overload
    @staticmethod
    def calc_V(omega: Array[float], core_radius: float,
               NA: Optional[Array[float]],
               n_core: Optional[Array[float]],
               n_clad: Optional[Array[float]]
               ) -> Array[float]: ...
    # ------------------------------------------------------------------
    @staticmethod
    def calc_V(omega, core_radius=cst.CORE_RADIUS, NA=None, n_core=None,
               n_clad=None):
        r"""Calculate the V parameter.

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
            Value of the V parameter.

        Notes
        -----
        Considering:

        .. math:: V = k_0 a \text{NA} = \frac{\omega_0}{c} a \text{NA}
                    = \frac{\omega_0}{c} a
                      (n_{co}^2 - n_{cl}^2)^{\frac{1}{2}}

        """
        # Unit conversion
        core_radius *= 1e3  # um -> nm

        if (isinstance(omega, float)):
            res = 0.0
        else:
            res = np.zeros(omega.shape)

        factor = core_radius * omega / cst.LIGHT_SPEED

        if (NA is not None):
            res = factor * NA

        elif ((n_core is not None) and (n_clad is not None)):
            if (isinstance(omega, float)):
                res = factor * math.sqrt(n_core**2 - n_clad**2)
            else:
                res = factor * np.sqrt(np.square(n_core) - np.square(n_clad))
        else:
            util.warning_terminal("Not enough information to calculate the "
                "V parameter, must specified at least the Numerical Aperture "
                "or the refractive index of the core and the cladding. Will "
                "return 0.")

        return res


if __name__ == "__main__":

    import math
    import numpy as np
    from optcom.domain import Domain

    # With float
    omega = Domain.lambda_to_omega(1552.0)
    core_radius = 5.0
    n_core = 1.43
    n_clad = 1.425
    NA = math.sqrt(n_core**2 - n_clad**2)

    v_para = V(core_radius=core_radius, n_core=n_core, n_clad=n_clad)
    print(v_para(omega))
    print(V.calc_V(omega, core_radius=core_radius, n_core=n_core,
                   n_clad=n_clad))

    v_para = V(core_radius=core_radius, NA=NA)
    print(v_para(omega))
    print(V.calc_V(omega, core_radius=core_radius, NA=NA))

    # With numpy ndarray
    lambdas = np.linspace(900, 1550, 10)
    omegas = Domain.lambda_to_omega(lambdas)
    n_core = np.linspace(1.42, 1.43, 10)
    n_clad = np.linspace(1.415, 1.425, 10)
    NA = np.sqrt(np.square(n_core)-np.square(n_clad))

    print(V.calc_V(omegas, core_radius=core_radius, n_core=n_core,
                   n_clad=n_clad))
    print(V.calc_V(omegas, core_radius=core_radius, NA=NA))
